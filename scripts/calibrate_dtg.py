"""
DTG calibration check — as specified in the design doc.

Validates edge weight quality against known domain-transfer pairs and
diagnoses miscalibrations (e.g., medical domains scoring high for
educational problems).

Usage:
    calibrate-dtg
    python scripts/calibrate_dtg.py
    python scripts/calibrate_dtg.py --verbose
    python scripts/calibrate_dtg.py --simulate-gaps "educational reform policy"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.config import DTG_PATH

# Known strong domain-transfer pairs: (node_id_a, node_id_b, description, min_expected_weight)
KNOWN_GOOD_PAIRS = [
    ("nat_bio",    "cross_bioinf",  "Biology ↔ Bioinformatics",         0.3),
    ("nat_cs",     "cross_bioinf",  "Computer Science ↔ Bioinformatics", 0.3),
    ("nat_phys",   "nat_math",      "Physics ↔ Mathematics",             0.3),
    ("nat_cs",     "nat_math",      "Computer Science ↔ Mathematics",    0.3),
    ("cross_datasi","nat_math",     "Data Science ↔ Mathematics",        0.3),
    ("soc_psych",  "cross_beheco",  "Psychology ↔ Behavioral Economics", 0.3),
    ("soc_econ",   "cross_beheco",  "Economics ↔ Behavioral Economics",  0.3),
    ("med_basic",  "nat_bio",       "Basic Medicine ↔ Biology",          0.3),
    ("nat_cs",     "cross_ai",      "Computer Science ↔ AI",             0.3),
    ("nat_cs",     "cross_cyber",   "Computer Science ↔ Cybersecurity",  0.3),
    ("eng_med",    "med_clinical",  "Medical Engineering ↔ Clinical Med",0.3),
]

# Medical division nodes that should have LOW weight to social-science domains
MEDICAL_DIVISION = ["med_basic", "med_clinical", "med_health", "med_biotech", "med_nursing"]
EDUCATIONAL_NODES = ["soc_edu", "soc_psych", "soc_soc"]

# Test skeleton for the gap-ranking simulation
TEST_SKELETON_EDUCATION = (
    "improve long-term outcomes for a population of learners "
    "through institutional intervention"
)
TEST_FRAME_EDUCATION = "educational reform and institutional policy"


def edge_weight(edges_by_src: dict, src: str, tgt: str) -> float:
    for e in edges_by_src.get(src, []):
        if e["target"] == tgt:
            return e["weight"]
    return 0.0


def load_dtg(path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    edges_by_src: dict[str, list] = {}
    for e in data["edges"]:
        edges_by_src.setdefault(e["source"], []).append(e)
    return data, nodes, edges_by_src


def print_weight_distribution(data: dict) -> None:
    weights = [e["weight"] for e in data["edges"]]
    buckets = [
        (0.0, 0.05),
        (0.05, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.5),
        (0.5, 0.9),
        (0.9, 1.01),
    ]
    total = len(weights)
    print(f"\n{'Edge weight distribution':}")
    print(f"  Total edges: {total}  |  Unique nodes: {len(data['nodes'])}")
    print(f"  {'Range':<12}  {'Count':>6}  {'Pct':>6}  Bar")
    for lo, hi in buckets:
        count = sum(1 for w in weights if lo <= w < hi)
        pct = 100 * count / total if total else 0
        bar = "#" * int(pct / 2)
        label = f"{lo:.2f}–{hi:.2f}" if hi < 1.01 else f"{lo:.2f}–1.00"
        print(f"  {label:<12}  {count:>6}  {pct:>5.1f}%  {bar}")


def check_known_good_pairs(nodes: dict, edges_by_src: dict) -> list[str]:
    failures = []
    print(f"\n{'Known domain-transfer pair check':}")
    print(f"  {'Pair':<45}  {'Weight':>7}  {'Status'}")
    for id_a, id_b, desc, min_w in KNOWN_GOOD_PAIRS:
        if id_a not in nodes or id_b not in nodes:
            print(f"  {desc:<45}  {'N/A':>7}  SKIP (node missing)")
            continue
        w_ab = edge_weight(edges_by_src, id_a, id_b)
        w_ba = edge_weight(edges_by_src, id_b, id_a)
        w = max(w_ab, w_ba)
        status = "OK" if w >= min_w else f"LOW (expected >= {min_w})"
        print(f"  {desc:<45}  {w:>7.4f}  {status}")
        if w < min_w:
            failures.append(f"{desc}: {w:.4f} < {min_w}")
    return failures


def check_medical_education_leakage(nodes: dict, edges_by_src: dict) -> list[str]:
    warnings = []
    threshold = 0.15
    print(f"\n{'Medical → Education edge check (false-connection diagnostic)':}")
    print(f"  Edges above {threshold} are likely false connections:")
    print(f"  {'Medical node':<35}  {'Edu node':<35}  {'Weight':>7}")
    found_any = False
    for med_id in MEDICAL_DIVISION:
        if med_id not in nodes:
            continue
        for edu_id in EDUCATIONAL_NODES:
            if edu_id not in nodes:
                continue
            w_med_edu = edge_weight(edges_by_src, med_id, edu_id)
            w_edu_med = edge_weight(edges_by_src, edu_id, med_id)
            w = max(w_med_edu, w_edu_med)
            if w > threshold:
                med_name = nodes[med_id]["domain"]
                edu_name = nodes[edu_id]["domain"]
                print(f"  {med_name:<35}  {edu_name:<35}  {w:>7.4f}  ** HIGH **")
                warnings.append(f"{med_name} ↔ {edu_name}: {w:.4f}")
                found_any = True
    if not found_any:
        print(f"  None found (all medical-education edges <= {threshold})")
    return warnings


def simulate_gap_ranking(frame_summary: str, skeleton: str, nodes: dict, edges_by_src: dict) -> None:
    """
    Simulate gap ranking WITHOUT loading the embedding model.
    Uses only structural edge weight + a string overlap heuristic
    so we can run this without torch/bge-large.
    """
    print(f"\n{'Gap ranking simulation (structural-only, no embeddings)':}")
    print(f"  Frame: {frame_summary!r}")
    print(f"  Skeleton: {skeleton!r}")

    _STOPWORDS = {"and", "or", "of", "the", "in", "a", "an", "for", "to",
                  "with", "by", "at", "on", "is", "are", "be", "reform",
                  "policy", "institutional", "applied", "management"}

    # Approximate activation: find nodes whose domain matches the frame summary words
    frame_words = {w for w in frame_summary.lower().split() if w not in _STOPWORDS and len(w) > 3}
    activated_ids = set()
    for nid, n in nodes.items():
        node_words = {w for w in (n["domain"].lower() + " " + n["division"].lower()).split()
                      if w not in _STOPWORDS and len(w) > 3}
        if node_words & frame_words:
            activated_ids.add(nid)

    if not activated_ids:
        print("  No nodes activated by frame (string matching fallback failed)")
        return

    print(f"  Approximate activated nodes: {[nodes[nid]['domain'] for nid in activated_ids]}")

    # Compute footprint at k_hops=1
    footprint = set(activated_ids)
    for src in list(activated_ids):
        for e in edges_by_src.get(src, []):
            footprint.add(e["target"])

    # Score gap nodes using simple word overlap with skeleton (no embeddings)
    skeleton_words = set(skeleton.lower().split())
    gap_scores = []
    for nid in nodes:
        if nid in footprint:
            continue
        node = nodes[nid]
        desc_words = set(node["description"].lower().split())
        overlap = len(desc_words & skeleton_words) / max(len(desc_words), 1)

        max_edge = 0.0
        for fp_nid in footprint:
            w = edge_weight(edges_by_src, fp_nid, nid)
            max_edge = max(max_edge, w)

        # Current formula (with 0.1 floor)
        score_current = max(max_edge, 0.1) * overlap

        # Fixed formula (0.02 base + cross-division bonus)
        activated_divisions = {nodes[nid2]["division"] for nid2 in activated_ids if nid2 in nodes}
        cross_div = 1.3 if node["division"] not in activated_divisions else 1.0
        score_fixed = max(max_edge, 0.02) * overlap * cross_div

        gap_scores.append((nid, node["domain"], node["division"], max_edge, overlap, score_current, score_fixed))

    gap_scores.sort(key=lambda x: x[5], reverse=True)
    top_current = gap_scores[:5]

    gap_scores.sort(key=lambda x: x[6], reverse=True)
    top_fixed = gap_scores[:5]

    print(f"\n  Top 5 gaps — CURRENT formula (0.1 floor):")
    for nid, domain, div, edge, ov, sc, sf in top_current:
        print(f"    [{div:<25}] {domain:<35}  score={sc:.4f}  edge={edge:.3f}  overlap={ov:.3f}")

    print(f"\n  Top 5 gaps — FIXED formula (0.02 base + cross-div bonus):")
    for nid, domain, div, edge, ov, sc, sf in top_fixed:
        print(f"    [{div:<25}] {domain:<35}  score={sf:.4f}  edge={edge:.3f}  overlap={ov:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibrate-dtg",
        description="Validate DTG edge weight quality and diagnose miscalibrations.",
    )
    parser.add_argument("--dtg", default=str(DTG_PATH), help="Path to dtg_300.json")
    parser.add_argument("--verbose", action="store_true", help="Print all edges for flagged pairs")
    parser.add_argument(
        "--simulate-gaps",
        metavar="FRAME",
        default=TEST_FRAME_EDUCATION,
        help="Frame summary to use for gap ranking simulation",
    )
    args = parser.parse_args()

    path = Path(args.dtg)
    if not path.exists():
        print(f"DTG not found at {path}. Run `build-dtg` first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading DTG from {path}...")
    data, nodes, edges_by_src = load_dtg(path)
    print(f"  {len(nodes)} nodes, {len(data['edges'])} edges")

    print_weight_distribution(data)
    failures = check_known_good_pairs(nodes, edges_by_src)
    warnings = check_medical_education_leakage(nodes, edges_by_src)
    simulate_gap_ranking(args.simulate_gaps, TEST_SKELETON_EDUCATION, nodes, edges_by_src)

    print(f"\n{'Summary':}")
    if failures:
        print(f"  Known-pair failures ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
    else:
        print(f"  Known-pair check: PASS (all pairs above minimum weight)")

    if warnings:
        print(f"  Medical-education false connections ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
        print(f"  Recommendation: lower the Jaccard fallback weight or add")
        print(f"  explicit exclusion for cross-division vocabulary overlap.")
    else:
        print(f"  Medical-education false-connection check: PASS")

    if failures or warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
