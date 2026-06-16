"""
Recalibrated strong-EFA test (free, uses existing scored_embed.json).

bge-large compresses cosine distances, so the roadmap's absolute strata (far>0.50)
are unreachable (max observed ~0.44). We instead use PERCENTILE tiers (distance terciles)
so "far" = relatively-far within this embedder's geometry. The decisive question:

  Do the most-distant concepts (top tercile) stay absent under the steelman (B),
  or does the strong prompt recover them?

  - strong-EFA (structural absence)  => top-tercile recall stays LOW under B
  - weak-EFA (prompt-correctable)    => top-tercile recall RISES under B

Outputs a console report + results/recalibrated.json + a tiered suppression figure.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.embeddings import embed

BENCH = ROOT / "bench" / "efa_bench.json"
SCORED = ROOT / "results" / "scored_embed.json"
OUT = ROOT / "results" / "recalibrated.json"
FIGS = ROOT / "results" / "figures"


def main() -> None:
    bench = {p["id"]: p for p in json.loads(BENCH.read_text(encoding="utf-8"))["problems"]}
    rows = json.loads(SCORED.read_text(encoding="utf-8"))["rows"]

    # ---- distances (bge-large) ------------------------------------------------------
    dist: dict[tuple[str, str], float] = {}
    for p in bench.values():
        c = np.mean(embed.encode(p["frame_concepts"]), axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        for g in p["ground_truth"]:
            dist[(p["id"], g["concept"])] = float(1.0 - np.dot(embed.encode(g["concept"]), c))

    all_d = np.array(sorted(dist.values()))
    q33, q67 = np.percentile(all_d, [33.33, 66.67])
    def tier(d: float) -> str:
        return "T1_near" if d < q33 else ("T2_mid" if d < q67 else "T3_far")

    # ---- recall@N per (model, condition, concept) -----------------------------------
    bucket: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        bucket[(r["model"], r["condition"], r["problem_id"], r["concept"])].append(bool(r["judge_present"]))
    def recall(presents): return 1 if any(presents) else 0

    models = sorted({k[0] for k in bucket})
    report = {"tier_bounds": {"q33": round(float(q33), 4), "q67": round(float(q67), 4),
                              "max_distance": round(float(all_d.max()), 4)},
              "tiers": {}, "strong_efa_test": {}, "top_tier_concepts": [], "residual_B": {}}

    for m in models:
        for cond in ("A", "B", "C_ECA"):
            keys = [k for k in bucket if k[0] == m and k[1] == cond]
            if not keys:
                continue
            tiered = defaultdict(list)
            for k in keys:
                tiered[tier(dist[(k[2], k[3])])].append(recall(bucket[k]))
            report["tiers"][f"{m}|{cond}"] = {
                t: {"recall": round(float(np.mean(v)), 3), "n": len(v)}
                for t, v in sorted(tiered.items())
            }

    # ---- strong-EFA test: top-tier recall A vs B, paired on concepts with both -------
    for m in models:
        pairs = []
        for k in [k for k in bucket if k[0] == m and k[1] == "A"]:
            kb = (m, "B", k[2], k[3])
            if kb in bucket and tier(dist[(k[2], k[3])]) == "T3_far":
                pairs.append((recall(bucket[k]), recall(bucket[kb]),
                              k[2], k[3], dist[(k[2], k[3])]))
        if pairs:
            a = np.mean([p[0] for p in pairs]); b = np.mean([p[1] for p in pairs])
            report["strong_efa_test"][m] = {
                "n_top_tier_pairs": len(pairs),
                "recall_A_top_tier": round(float(a), 3),
                "recall_B_top_tier": round(float(b), 3),
                "interpretation": ("B recovers far concepts -> weak/correctable"
                                   if b - a > 0.1 and b >= 0.7 else
                                   "far concepts stay low under B -> strong-EFA support"
                                   if b < 0.5 else "mixed"),
            }

    # ---- concept-level table for top tier (most distant) ----------------------------
    top = sorted([(d, pid, c) for (pid, c), d in dist.items() if tier(d) == "T3_far"],
                 reverse=True)
    for d, pid, c in top:
        entry = {"problem_id": pid, "concept": c, "distance": round(d, 4)}
        for m in models:
            for cond in ("A", "B"):
                k = (m, cond, pid, c)
                entry[f"{m.split('/')[-1]}_{cond}"] = (recall(bucket[k]) if k in bucket else None)
        report["top_tier_concepts"].append(entry)

    # ---- residual absence under B with distances ------------------------------------
    for m in models:
        res = [{"problem_id": k[2], "concept": k[3], "distance": round(dist[(k[2], k[3])], 4),
                "tier": tier(dist[(k[2], k[3])])}
               for k in bucket if k[0] == m and k[1] == "B" and recall(bucket[k]) == 0]
        res.sort(key=lambda x: -x["distance"])
        report["residual_B"][m] = res

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figure: recall vs distance, tier-colored, A vs B (sonnet) ------------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    son = next((m for m in models if "sonnet" in m), models[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, color in [("A", "tab:red"), ("B", "tab:green")]:
        keys = [k for k in bucket if k[0] == son and k[1] == cond]
        if not keys:
            continue
        xs = [dist[(k[2], k[3])] for k in keys]
        ys = [np.mean(bucket[k]) for k in keys]
        ax.scatter(xs, ys, alpha=0.6, color=color, label=f"{cond} (n={len(xs)})")
    for q in (q33, q67):
        ax.axvline(q, ls=":", color="gray", lw=1)
    ax.set_xlabel("cosine distance from frame (bge-large)")
    ax.set_ylabel("recall-rate over samples (embedding proxy)")
    ax.set_title(f"Recalibrated suppression — {son.split('/')[-1]}\n(dotted = distance terciles)")
    ax.set_ylim(-0.05, 1.05); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "recalibrated_suppression.png", dpi=120)
    plt.close(fig)

    # ---- console --------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RECALIBRATED STRONG-EFA TEST (percentile tiers, embedding proxy)")
    print("=" * 72)
    print(f"distance terciles: T1<{q33:.3f}  T2<{q67:.3f}  T3>= {q67:.3f}  (max={all_d.max():.3f})")
    print("\nRecall@N by tier:")
    for k, v in report["tiers"].items():
        print(f"  {k:52s} " + "  ".join(f"{t}={d['recall']:.2f}(n{d['n']})" for t, d in v.items()))
    print("\nSTRONG-EFA TEST (top-tier far concepts, A vs B paired):")
    for m, v in report["strong_efa_test"].items():
        print(f"  {m.split('/')[-1]:20s} A={v['recall_A_top_tier']:.2f} B={v['recall_B_top_tier']:.2f} "
              f"(n={v['n_top_tier_pairs']})  => {v['interpretation']}")
    print("\nMost-distant concepts (recall: sonnet A/B, gpt4o A/B):")
    for e in report["top_tier_concepts"][:15]:
        sa = e.get("claude-sonnet-4-5_A"); sb = e.get("claude-sonnet-4-5_B")
        ga = e.get("gpt-4o_A"); gb = e.get("gpt-4o_B")
        print(f"  d={e['distance']:.3f} {e['concept'][:42]:42s} sonnet[A={sa},B={sb}] gpt4o[A={ga},B={gb}]")
    print("\nResidual absence under B (sonnet):")
    for e in report["residual_B"].get(son, []):
        print(f"  d={e['distance']:.3f} ({e['tier']})  {e['concept']}")
    print(f"\nwrote {OUT} and {FIGS / 'recalibrated_suppression.png'}")


if __name__ == "__main__":
    main()
