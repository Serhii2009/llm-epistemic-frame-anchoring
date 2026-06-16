"""
EFA falsification — scoring.

For every generated response, decide whether each ground-truth outside-frame concept
was SUBSTANTIVELY surfaced, using two independent signals:

  1. LLM judge (primary): a model DIFFERENT from the generator reads the response and
     rules, per concept, present/absent with a supporting quote. "Substantive invocation,
     not a passing allusion." Cross-model judging avoids self-grading.
  2. BGE-M3 chunk similarity (secondary): max cosine between the concept and any sentence
     of the response. Reported at thresholds tau in {0.5,0.6,0.7} for sensitivity and to
     measure judge<->embedding agreement.

Input : results/raw/gen_*.json (+ eca_*.json)
Output: results/scored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.embeddings import embed
from efa.llm import LLMClient

BENCH = ROOT / "bench" / "efa_bench.json"
RAW = ROOT / "results" / "raw"
SCORED = ROOT / "results" / "scored.json"

# Generator -> judge (must differ from generator).
JUDGE_FOR = {
    "anthropic/claude-sonnet-4-5": "openai/gpt-4o",
    "openai/gpt-4o": "anthropic/claude-sonnet-4-5",
    "anthropic/claude-opus-4-8": "openai/gpt-4o",
}

JUDGE_SYSTEM = (
    "You are a strict, literal evaluator of whether a response invokes specific concepts.\n"
    "For each concept listed, decide whether the RESPONSE substantively invokes or applies "
    "it — meaning the response raises that idea in a meaningful way, either by name, by a "
    "well-known synonym, or by an unmistakable description of its mechanism. A vague, "
    "generic allusion that could mean many things does NOT count. Be strict but fair: the "
    "exact wording need not match if the specific idea is clearly present.\n\n"
    "Return ONLY JSON: {\"results\": [{\"concept\": \"<verbatim concept>\", "
    "\"present\": true|false, \"evidence\": \"<short quote from response, or 'none'>\"}]}"
)

TAUS = [0.5, 0.6, 0.7]


def load_bench() -> dict[str, dict]:
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in data["problems"]}


def judge_one(judge: LLMClient, problem: dict, response_text: str) -> dict[str, dict]:
    """Return {concept: {'present': bool, 'evidence': str}} for one response."""
    concepts = problem["ground_truth"]
    concept_lines = "\n".join(
        f'- "{c["concept"]}" (from {c["from_domain"]})' for c in concepts
    )
    prompt = (
        f"PROBLEM:\n{problem['problem_statement']}\n\n"
        f"RESPONSE TO EVALUATE:\n{response_text[:6000]}\n\n"
        f"CONCEPTS TO CHECK:\n{concept_lines}\n\n"
        "For each concept, is it substantively present in the response?"
    )
    out: dict[str, dict] = {}
    try:
        parsed = judge.complete_json(prompt, system=JUDGE_SYSTEM)
        rows = parsed.get("results", []) if isinstance(parsed, dict) else []
        by_name = {r.get("concept", "").strip().lower(): r for r in rows}
        for c in concepts:
            r = by_name.get(c["concept"].strip().lower())
            if r is None:
                # fall back: fuzzy index by position if names drift
                r = next((rr for rr in rows if c["concept"][:20].lower() in
                          str(rr.get("concept", "")).lower()), None)
            out[c["concept"]] = {
                "present": bool(r.get("present", False)) if r else False,
                "evidence": (r.get("evidence", "") if r else "")[:200],
            }
    except Exception as e:
        for c in concepts:
            out[c["concept"]] = {"present": False, "evidence": f"[judge error: {e}]"}
    return out


def embed_scores(problem: dict, response_text: str) -> dict[str, float]:
    return {c["concept"]: embed.best_chunk_sim(c["concept"], response_text)
            for c in problem["ground_truth"]}


def score_gen_file(path: Path, bench: dict[str, dict]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gen_model = data["model"]
    judge_id = JUDGE_FOR[gen_model]
    judge = LLMClient(model=judge_id)
    gens = data["generations"]
    print(f"  scoring {path.name}: {len(gens)} responses (judge={judge_id})", flush=True)

    # Judge in parallel: build one judge prompt per response.
    concept_lines_for = {}
    prompts, meta = [], []
    for g in gens:
        prob = bench[g["problem_id"]]
        concepts = prob["ground_truth"]
        cl = "\n".join(f'- "{c["concept"]}" (from {c["from_domain"]})' for c in concepts)
        concept_lines_for[g["problem_id"]] = cl
        prompts.append(
            f"PROBLEM:\n{prob['problem_statement']}\n\n"
            f"RESPONSE TO EVALUATE:\n{(g['text'] or '')[:6000]}\n\n"
            f"CONCEPTS TO CHECK:\n{cl}\n\n"
            "For each concept, is it substantively present in the response?"
        )
        meta.append(g)

    raw_judge = judge.complete_parallel(prompts, system=JUDGE_SYSTEM, temperature=0.0,
                                        max_workers=8)

    rows: list[dict] = []
    for g, rj in zip(meta, raw_judge):
        prob = bench[g["problem_id"]]
        # parse judge json
        present_map: dict[str, dict] = {}
        try:
            parsed = json.loads(LLMClient._extract_json(rj))
            jrows = parsed.get("results", []) if isinstance(parsed, dict) else []
            by_name = {str(r.get("concept", "")).strip().lower(): r for r in jrows}
        except Exception:
            by_name = {}
        # embedding scores (chunk-level)
        esc = embed_scores(prob, g["text"] or "")
        for c in prob["ground_truth"]:
            jr = by_name.get(c["concept"].strip().lower())
            if jr is None:
                jr = next((rr for rr in by_name.values()
                           if c["concept"][:20].lower() in str(rr.get("concept", "")).lower()),
                          None)
            present = bool(jr.get("present", False)) if jr else False
            evidence = (jr.get("evidence", "") if jr else "")[:200]
            rows.append({
                "model": gen_model, "problem_id": g["problem_id"],
                "condition": g["condition"], "sample": g["sample"],
                "concept": c["concept"], "from_domain": c["from_domain"],
                "judge_present": present, "judge_evidence": evidence,
                "emb_sim": round(esc[c["concept"]], 4),
            })
    return rows


def score_eca_file(path: Path, bench: dict[str, dict]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gen_model = data["model"]
    judge = LLMClient(model=JUDGE_FOR[gen_model])
    recs = [r for r in data["records"] if "error" not in r]
    print(f"  scoring {path.name}: {len(recs)} ECA responses", flush=True)
    rows: list[dict] = []
    for r in recs:
        prob = bench[r["problem_id"]]
        # ECA value = synthesis text + explicit surfaced concept list
        combined = (r.get("response", "") or "") + "\nSURFACED: " + \
            "; ".join(r.get("surfaced_concepts", []))
        jmap = judge_one(judge, prob, combined)
        esc = embed_scores(prob, combined)
        for c in prob["ground_truth"]:
            rows.append({
                "model": gen_model, "problem_id": r["problem_id"],
                "condition": "C_ECA", "sample": r["sample"],
                "concept": c["concept"], "from_domain": c["from_domain"],
                "judge_present": jmap[c["concept"]]["present"],
                "judge_evidence": jmap[c["concept"]]["evidence"],
                "emb_sim": round(esc[c["concept"]], 4),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="gen_*.json", help="raw gen files to score")
    ap.add_argument("--out", default=str(SCORED))
    args = ap.parse_args()

    bench = load_bench()
    all_rows: list[dict] = []
    for path in sorted(RAW.glob(args.glob)):
        all_rows.extend(score_gen_file(path, bench))
    for path in sorted(RAW.glob("eca_*.json")):
        all_rows.extend(score_eca_file(path, bench))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"taus": TAUS, "rows": all_rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  ({len(all_rows)} concept-judgements)")


if __name__ == "__main__":
    main()
