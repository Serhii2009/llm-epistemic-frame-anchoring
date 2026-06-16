"""
EFA falsification — analysis and verdict computation.

Consumes results/scored.json + bench/efa_bench.json and produces:
  - per-concept cosine distance to its problem frame (BGE-M3 centroid)
  - recall@N and recall-rate per (model, condition), overall and by distance stratum
  - the SUPPRESSION CURVE: recall-rate vs distance, separately under A and B, per model,
    with Spearman rho/p  (EFA's signature is a negative slope that PERSISTS under B)
  - steelman gap closure: paired A-vs-B recall@N (Wilcoxon)
  - residual absences: source-verified concepts with recall@N == 0 even under B
  - judge<->embedding agreement at tau in {0.5,0.6,0.7}
  - figures + results/summary.json
  - prints the pre-registered decision-rule evaluation

Run: python experiments/analyze.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.embeddings import embed

BENCH = ROOT / "bench" / "efa_bench.json"
SCORED = ROOT / "results" / "scored.json"
SUMMARY = ROOT / "results" / "summary.json"
FIGS = ROOT / "results" / "figures"

NEAR, FAR = 0.30, 0.50  # distance strata cut points (computed distances, post-hoc)


def frame_distances() -> dict[tuple[str, str], float]:
    """distance(concept, frame) = 1 - cos(embed(concept), centroid(embed(frame_concepts)))."""
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    dist: dict[tuple[str, str], float] = {}
    for p in data["problems"]:
        fc = p["frame_concepts"]
        centroid = np.mean(embed.encode(fc), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        for c in p["ground_truth"]:
            cvec = embed.encode(c["concept"])
            dist[(p["id"], c["concept"])] = float(1.0 - np.dot(cvec, centroid))
    return dist


def stratum(d: float) -> str:
    return "near" if d < NEAR else ("medium" if d < FAR else "far")


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=str(SCORED))
    ap.add_argument("--summary", default=str(SUMMARY))
    args = ap.parse_args()
    scored_path = Path(args.scored)
    summary_path = Path(args.summary)

    rows = load_rows(scored_path)
    dist = frame_distances()
    FIGS.mkdir(parents=True, exist_ok=True)

    # Index: per (model, condition, problem_id, concept) -> list of judge_present over samples
    bucket: dict[tuple, list[bool]] = defaultdict(list)
    emb_bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["condition"], r["problem_id"], r["concept"])
        bucket[key].append(bool(r["judge_present"]))
        emb_bucket[key].append(float(r["emb_sim"]))

    models = sorted({r["model"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})

    # ---- recall tables -------------------------------------------------------------
    summary: dict = {"models": models, "conditions": conditions,
                     "strata": {"near": f"<{NEAR}", "medium": f"{NEAR}-{FAR}", "far": f">{FAR}"},
                     "n_concepts": len({(p, c) for (_, _, p, c) in bucket}),
                     "per_condition": {}, "suppression": {}, "ab_closure": {},
                     "residual_absence": {}, "judge_embedding_agreement": {}, "eca": {}}

    def recall_at_n(presents: list[bool]) -> int:
        return 1 if any(presents) else 0

    # per (model, condition): mean recall@N overall + by stratum
    for m in models:
        for cond in conditions:
            keys = [k for k in bucket if k[0] == m and k[1] == cond]
            if not keys:
                continue
            r_at_n = [recall_at_n(bucket[k]) for k in keys]
            by_stratum = defaultdict(list)
            for k in keys:
                by_stratum[stratum(dist[(k[2], k[3])])].append(recall_at_n(bucket[k]))
            summary["per_condition"][f"{m}|{cond}"] = {
                "n_concepts": len(keys),
                "recall_at_N": round(float(np.mean(r_at_n)), 4),
                "by_stratum": {s: round(float(np.mean(v)), 4) for s, v in sorted(by_stratum.items())},
                "by_stratum_n": {s: len(v) for s, v in sorted(by_stratum.items())},
            }

    # ---- suppression curve: recall-rate vs distance, Spearman, per (model, condition) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for m in models:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cond in [c for c in conditions if c in ("A", "B")]:
            keys = [k for k in bucket if k[0] == m and k[1] == cond]
            if not keys:
                continue
            xs = np.array([dist[(k[2], k[3])] for k in keys])
            ys = np.array([np.mean(bucket[k]) for k in keys])  # recall-rate over samples
            if len(xs) >= 3 and np.std(ys) > 0:
                rho, pval = stats.spearmanr(xs, ys)
            else:
                rho, pval = float("nan"), float("nan")
            summary["suppression"][f"{m}|{cond}"] = {
                "spearman_rho": None if np.isnan(rho) else round(float(rho), 4),
                "p_value": None if np.isnan(pval) else round(float(pval), 6),
                "n": int(len(xs)),
            }
            ax.scatter(xs, ys, alpha=0.6, label=f"{cond} (ρ={rho:.2f}, p={pval:.3g})")
            # trend line
            if len(xs) >= 3:
                z = np.polyfit(xs, ys, 1)
                xr = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xr, np.polyval(z, xr), linewidth=1)
        ax.set_xlabel("cosine distance of concept from frame (BGE-M3)")
        ax.set_ylabel("recall-rate over N samples")
        ax.set_title(f"Suppression curve — {m}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        safe = m.replace("/", "_")
        fig.savefig(FIGS / f"suppression_{safe}.png", dpi=120)
        plt.close(fig)

    # ---- steelman gap closure: paired A vs B recall@N (Wilcoxon) --------------------
    for m in models:
        common = [(k[2], k[3]) for k in bucket if k[0] == m and k[1] == "A"]
        a = []; b = []
        for (pid, c) in common:
            ka = (m, "A", pid, c); kb = (m, "B", pid, c)
            if ka in bucket and kb in bucket:
                a.append(recall_at_n(bucket[ka])); b.append(recall_at_n(bucket[kb]))
        if a and any(ai != bi for ai, bi in zip(a, b)):
            try:
                stat, p = stats.wilcoxon(b, a)  # B - A
            except Exception:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        summary["ab_closure"][m] = {
            "n_pairs": len(a),
            "recall_A": round(float(np.mean(a)), 4) if a else None,
            "recall_B": round(float(np.mean(b)), 4) if b else None,
            "delta_B_minus_A": round(float(np.mean(b) - np.mean(a)), 4) if a else None,
            "wilcoxon_p": None if np.isnan(p) else round(float(p), 6),
        }

    # ---- residual absence: concepts with recall@N == 0 under B (per model) ----------
    for m in models:
        res = []
        for k in [k for k in bucket if k[0] == m and k[1] == "B"]:
            if recall_at_n(bucket[k]) == 0:
                d = dist[(k[2], k[3])]
                res.append({"problem_id": k[2], "concept": k[3],
                            "distance": round(d, 4), "stratum": stratum(d)})
        res.sort(key=lambda x: -x["distance"])
        summary["residual_absence"][m] = {
            "count": len(res),
            "by_stratum": {s: sum(1 for r in res if r["stratum"] == s)
                           for s in ("near", "medium", "far")},
            "items": res,
        }

    # ---- judge <-> embedding agreement at each tau ---------------------------------
    taus = json.loads(scored_path.read_text(encoding="utf-8")).get("taus", [0.5, 0.6, 0.7])
    for tau in taus:
        agree = 0; tot = 0; jp = 0; ep = 0
        for r in rows:
            j = bool(r["judge_present"]); e = float(r["emb_sim"]) >= tau
            tot += 1; agree += int(j == e); jp += int(j); ep += int(e)
        summary["judge_embedding_agreement"][str(tau)] = {
            "agreement": round(agree / tot, 4) if tot else None,
            "judge_present_rate": round(jp / tot, 4) if tot else None,
            "embed_present_rate": round(ep / tot, 4) if tot else None,
        }

    # ---- ECA (condition C) recall, if present --------------------------------------
    eca_keys = [k for k in bucket if k[1] == "C_ECA"]
    if eca_keys:
        r = [recall_at_n(bucket[k]) for k in eca_keys]
        summary["eca"]["recall_at_N"] = round(float(np.mean(r)), 4)
        summary["eca"]["n_concepts"] = len(eca_keys)

    # ---- decision-rule evaluation (pre-registered) ---------------------------------
    # EFA REAL requires, on >=2 models under B: (i) Spearman rho<0 p<0.01;
    # (ii) non-trivial far residual absences under B; (iii) Opus fails to recover (phase 2).
    verdict = {"models_with_neg_suppression_under_B": [], "models_with_far_residual_B": []}
    for m in models:
        s = summary["suppression"].get(f"{m}|B")
        if s and s["spearman_rho"] is not None and s["spearman_rho"] < 0 and \
           s["p_value"] is not None and s["p_value"] < 0.01:
            verdict["models_with_neg_suppression_under_B"].append(m)
        ra = summary["residual_absence"].get(m, {})
        if ra.get("by_stratum", {}).get("far", 0) >= 3:
            verdict["models_with_far_residual_B"].append(m)
    summary["decision_rule_phase1"] = verdict

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- console report ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EFA FALSIFICATION — PHASE 1 SUMMARY")
    print("=" * 72)
    print(f"concepts/condition ≈ {summary['n_concepts']}  | models: {models}")
    print("\nRecall@N (judge) by condition:")
    for k, v in summary["per_condition"].items():
        print(f"  {k:55s} overall={v['recall_at_N']:.3f}  by_stratum={v['by_stratum']}")
    print("\nSuppression (recall-rate vs distance, under B is decisive):")
    for k, v in summary["suppression"].items():
        print(f"  {k:55s} rho={v['spearman_rho']}  p={v['p_value']}  n={v['n']}")
    print("\nSteelman A→B closure:")
    for m, v in summary["ab_closure"].items():
        print(f"  {m:40s} A={v['recall_A']} B={v['recall_B']} ΔB-A={v['delta_B_minus_A']} (wilcoxon p={v['wilcoxon_p']})")
    print("\nResidual absence under B (recall@N==0):")
    for m, v in summary["residual_absence"].items():
        print(f"  {m:40s} total={v['count']} by_stratum={v['by_stratum']}")
    print("\nJudge↔embedding agreement:", summary["judge_embedding_agreement"])
    if summary["eca"]:
        print("ECA (cond C) recall@N:", summary["eca"])
    print("\nDecision-rule (phase 1):")
    print("  neg-suppression under B (rho<0,p<.01):", verdict["models_with_neg_suppression_under_B"])
    print("  >=3 FAR residual absences under B:    ", verdict["models_with_far_residual_B"])
    print(f"\nwrote {summary_path}  and figures to {FIGS}")


if __name__ == "__main__":
    main()
