"""
PRELIMINARY embedding-only scoring (no LLM judge — runs fully local, free).

Used only when API credits are exhausted, to extract a preliminary read from the
responses already paid for. Sets `judge_present := (best_chunk_sim >= tau)`. The LLM
judge remains the primary detector for the FINAL verdict; results here are explicitly
provisional and must be confirmed by judge.py once credits are restored.

Efficiency: all sentence chunks from all responses are flattened and embedded in ONE
batched encode call (with a progress bar) instead of one encode per response — far
faster on a CPU-only laptop. Responses are clipped and chunk-capped to bound cost.

Skips error responses. Output is scored.json-compatible so analyze.py consumes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.embeddings import embed

BENCH = ROOT / "bench" / "efa_bench.json"
RAW = ROOT / "results" / "raw"

# Whole-response embedding (one vector per response). bge-large is too slow on this
# CPU-only laptop for sentence-level chunking of ~12k chunks (~10s/64-batch). We embed
# the clipped response as a single unit. bge-large truncates at ~512 tokens anyway, so
# CLIP_CHARS≈2200 ≈ what the model actually reads. Naive (short) responses are fully
# captured; very long steelman responses are truncated → B recall may be UNDERcounted
# in this free proxy. The final LLM-judge run (max_tokens-capped) removes this caveat.
CLIP_CHARS = 2200
MAX_CHUNKS = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.60)
    ap.add_argument("--glob", default="gen_*.json")
    ap.add_argument("--out", default=str(ROOT / "results" / "scored_embed.json"))
    args = ap.parse_args()

    bench = {p["id"]: p for p in json.loads(BENCH.read_text(encoding="utf-8"))["problems"]}

    # ---- gather all responses + flatten chunks --------------------------------------
    gens: list[dict] = []
    n_skipped = 0
    for path in sorted(RAW.glob(args.glob)):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        for g in data["generations"]:
            text = g.get("text") or ""
            if text.startswith("[ERROR"):
                n_skipped += 1
                continue
            gens.append({"model": model, **g})

    all_chunks: list[str] = []
    spans: list[tuple[int, int]] = []   # (start, end) into all_chunks per gen
    for g in gens:
        if MAX_CHUNKS <= 1:
            chunks = [g["text"][:CLIP_CHARS]]            # whole (clipped) response
        else:
            chunks = embed._sentences(g["text"][:CLIP_CHARS])[:MAX_CHUNKS]
        start = len(all_chunks)
        all_chunks.extend(chunks)
        spans.append((start, len(all_chunks)))
    print(f"{len(gens)} responses, {len(all_chunks)} chunks ({n_skipped} errors skipped)",
          flush=True)

    # ---- one big batched encode for chunks, one for concepts ------------------------
    embed._load()
    print("encoding chunks…", flush=True)
    chunk_vecs = embed._model.encode(all_chunks, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=True)
    chunk_vecs = np.asarray(chunk_vecs)

    # unique concepts
    uniq = []
    cindex: dict[str, int] = {}
    for p in bench.values():
        for c in p["ground_truth"]:
            if c["concept"] not in cindex:
                cindex[c["concept"]] = len(uniq)
                uniq.append(c["concept"])
    print(f"encoding {len(uniq)} unique concepts…", flush=True)
    concept_vecs = np.asarray(embed._model.encode(uniq, normalize_embeddings=True,
                                                  batch_size=64, show_progress_bar=False))

    # ---- score ----------------------------------------------------------------------
    rows: list[dict] = []
    for g, (s, e) in zip(gens, spans):
        prob = bench[g["problem_id"]]
        if e > s:
            cm = chunk_vecs[s:e]                       # (n_chunks, dim)
            for c in prob["ground_truth"]:
                cv = concept_vecs[cindex[c["concept"]]]
                sim = float(np.max(cm @ cv))
                rows.append(_row(g, c, sim, args.tau))
        else:
            for c in prob["ground_truth"]:
                rows.append(_row(g, c, 0.0, args.tau))

    Path(args.out).write_text(json.dumps(
        {"taus": [0.5, 0.6, 0.7], "tau_used_for_present": args.tau,
         "detector": "EMBEDDING_ONLY_PROXY", "clip_chars": CLIP_CHARS,
         "max_chunks": MAX_CHUNKS, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} rows, tau={args.tau}")


def _row(g: dict, c: dict, sim: float, tau: float) -> dict:
    return {
        "model": g["model"], "problem_id": g["problem_id"],
        "condition": g["condition"], "sample": g["sample"],
        "concept": c["concept"], "from_domain": c["from_domain"],
        "judge_present": bool(sim >= tau),  # EMBEDDING PROXY, not LLM judge
        "judge_evidence": "(embedding-only proxy)",
        "emb_sim": round(sim, 4),
    }


if __name__ == "__main__":
    main()
