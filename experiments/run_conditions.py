"""
EFA falsification — generation harness.

Runs the controlled conditions that the existing repo never ran:

  A — Naive:    bare problem statement, no system prompt.  (realistic default usage)
  B — Steelman: strongest realistic maximal-coverage system prompt.  (the Opus objection)
  C — ECA:      the existing 9-component pipeline.            (secondary)

Same problem text in all conditions. N stochastic samples per (problem x condition x model)
at temperature 1.0 for A and B. The decisive question is whether B closes the gap to the
ground-truth outside-frame concepts that A misses — if a strong prompt recovers them, EFA
is a prompting artifact, not a structural absence.

Usage:
  python experiments/run_conditions.py --dry           # 1 problem, N=2, both models, A+B
  python experiments/run_conditions.py --n 10           # full A+B sweep, both models
  python experiments/run_conditions.py --eca            # also run ECA (condition C) on sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.llm import LLMClient

BENCH = ROOT / "bench" / "efa_bench.json"
RAW = ROOT / "results" / "raw"

MODELS = {
    "sonnet": "anthropic/claude-sonnet-4-5",
    "gpt4o": "openai/gpt-4o",
    "opus": "anthropic/claude-opus-4-8",
}

# Condition B: the steelman. The strongest realistic coverage-eliciting prompt a
# thoughtful user or system would use. If EFA is merely "default vs thoughtful
# prompting," THIS prompt should surface the far concepts. It is designed to give the
# hypothesis every chance to be falsified.
STEELMAN_SYSTEM = (
    "You are an interdisciplinary panel of world-class experts spanning every field of "
    "science, engineering, mathematics, social science, the humanities, and professional "
    "practice. Your goal when answering is MAXIMAL conceptual coverage of everything "
    "genuinely relevant to the problem.\n\n"
    "Before and while answering:\n"
    "1. Identify the obvious disciplinary framing of the problem.\n"
    "2. Deliberately enumerate which OTHER disciplines — including ones far from the "
    "obvious framing — bear on this problem.\n"
    "3. Surface the specific named concepts, theories, models, laws, and frameworks from "
    "those other disciplines that a specialist in the obvious framing would likely "
    "overlook.\n\n"
    "Be exhaustive and concrete. Name specific concepts and the field each comes from. Do "
    "NOT restrict yourself to the vocabulary of the field the problem appears to belong to. "
    "Prioritize non-obvious, cross-disciplinary insight."
)

CONDITION_SYSTEM = {"A": None, "B": STEELMAN_SYSTEM}


def load_problems(limit: int | None = None) -> list[dict]:
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    probs = data["problems"]
    return probs[:limit] if limit else probs


def run_ab(model_tag: str, model_id: str, problems: list[dict], n: int,
           conditions: list[str]) -> dict:
    llm = LLMClient(model=model_id)
    generations: list[dict] = []

    for cond in conditions:
        system = CONDITION_SYSTEM[cond]
        # Build the full batch of (problem, sample) prompts for this condition.
        index: list[tuple[str, int]] = []
        prompts: list[str] = []
        for p in problems:
            for s in range(n):
                index.append((p["id"], s))
                prompts.append(p["problem_statement"])

        print(f"  [{model_tag}] condition {cond}: {len(prompts)} calls…", flush=True)
        t0 = time.time()
        texts = llm.complete_parallel(prompts, system=system, temperature=1.0, max_workers=8)
        dt = time.time() - t0
        n_err = sum(1 for t in texts if t.startswith("[ERROR"))
        print(f"      done in {dt:.0f}s  ({n_err} errors)", flush=True)

        for (pid, s), text in zip(index, texts):
            generations.append({
                "problem_id": pid, "condition": cond, "sample": s, "text": text,
            })

    return {"model": model_id, "model_tag": model_tag, "n_samples": n,
            "conditions": conditions, "generations": generations}


def run_eca(problems: list[dict], model_id: str, n: int = 1) -> dict:
    """Condition C — the existing ECA pipeline. Secondary, cost-bounded (default N=1)."""
    from efa import ECA
    from efa.config import DTG_PATH

    records: list[dict] = []
    for p in problems:
        for s in range(n):
            try:
                eca = ECA(dtg_path=DTG_PATH, model=model_id, k_gaps=3)
                res = eca.run(p["problem_statement"])
                records.append({
                    "problem_id": p["id"], "sample": s,
                    "response": res.response,
                    "surfaced_concepts": [dc.concept for dc in res.concepts],
                    "gap_domains": [n_.domain for n_ in res.gap_nodes],
                    "audit": res.coverage_audit,
                })
                print(f"  [ECA] {p['id']} s{s}: {len(res.concepts)} concepts", flush=True)
            except Exception as e:
                records.append({"problem_id": p["id"], "sample": s, "error": str(e)})
                print(f"  [ECA] {p['id']} s{s}: ERROR {e}", flush=True)
    return {"model": model_id, "n_samples": n, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per problem x condition x model")
    ap.add_argument("--models", nargs="+", default=["sonnet", "gpt4o"])
    ap.add_argument("--conditions", nargs="+", default=["A", "B"])
    ap.add_argument("--limit", type=int, default=None, help="limit number of problems")
    ap.add_argument("--dry", action="store_true", help="dry run: 1 problem, N=2")
    ap.add_argument("--eca", action="store_true", help="also run ECA condition C on sonnet")
    ap.add_argument("--eca-n", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="overwrite existing raw files")
    ap.add_argument("--tag", default="", help="filename suffix (e.g. 'dry')")
    args = ap.parse_args()

    if args.dry:
        args.n = 2
        args.limit = 1

    RAW.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.limit)
    print(f"Problems: {len(problems)} | N={args.n} | models={args.models} | "
          f"conditions={args.conditions}")

    suffix = f"_{args.tag}" if args.tag else ""

    for mt in args.models:
        out = RAW / f"gen_{mt}{suffix}.json"
        if out.exists() and not args.force:
            print(f"  skip {out.name} (exists; --force to overwrite)")
            continue
        result = run_ab(mt, MODELS[mt], problems, args.n, args.conditions)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {out}")

    if args.eca:
        out = RAW / f"eca_sonnet{suffix}.json"
        if out.exists() and not args.force:
            print(f"  skip {out.name} (exists)")
        else:
            result = run_eca(problems, MODELS["sonnet"], n=args.eca_n)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
