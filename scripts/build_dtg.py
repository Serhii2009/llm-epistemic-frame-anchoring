"""
Build the Domain Topology Graph (DTG).

Strategy:
  1. Load OECD FORD + cross-disciplinary nodes from efa.dtg.nodes
  2. For each Semantic Scholar field pair (A, B), query S2 to count papers that
     appear in BOTH fields (cross-domain co-occurrence).
     weight(A, B) = papers_in_both / papers_searched (capped at 1.0)
  3. Map S2-level weights back to OECD node pairs (max over field combinations).
  4. Save data/dtg_300.json with metadata, nodes, and edges.

Resumable: partial results saved to data/dtg_partial.json after each domain.
Resume by re-running; already-computed pairs are skipped.

Runtime: ~5-15 minutes (S2 rate limit ~1 req/s, ~190 unique field pairs).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from efa.dtg.nodes import FORD_NODES, S2_FIELD_MAP

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
PARTIAL_PATH = ROOT / "data" / "dtg_partial.json"
OUTPUT_PATH = ROOT / "data" / "dtg_300.json"
MIN_WEIGHT = 0.03   # ignore very sparse connections
PAPERS_PER_QUERY = 100
RATE_LIMIT_SLEEP = 1.1  # seconds between API calls


def s2_field_co_occurrence(field_a: str, field_b: str, n: int = PAPERS_PER_QUERY) -> float:
    """
    Query S2 for papers in field_a that also appear in field_b.
    Returns fraction of returned papers that contain BOTH fields.
    """
    params = {
        "query": f"{field_a} {field_b} interdisciplinary",
        "fields": "fieldsOfStudy",
        "limit": n,
    }
    try:
        resp = requests.get(S2_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  S2 error ({field_a}×{field_b}): {e}")
        return 0.0

    papers = data.get("data", [])
    if not papers:
        return 0.0

    both_count = sum(
        1
        for p in papers
        if p.get("fieldsOfStudy")
        and field_a in p["fieldsOfStudy"]
        and field_b in p["fieldsOfStudy"]
    )
    return both_count / len(papers)


def build_s2_matrix(existing: dict[str, float]) -> dict[str, float]:
    """Compute co-occurrence weights for all unique S2 field pairs."""
    all_s2_fields: set[str] = set()
    for fields in S2_FIELD_MAP.values():
        all_s2_fields.update(fields)

    pairs = list(combinations(sorted(all_s2_fields), 2))
    total = len(pairs)
    print(f"Computing weights for {total} S2 field pairs…")

    for i, (fa, fb) in enumerate(pairs):
        key = f"{fa}||{fb}"
        if key in existing:
            continue

        w = s2_field_co_occurrence(fa, fb)
        existing[key] = w
        print(f"  [{i+1}/{total}] {fa} × {fb} → {w:.3f}")

        # Save checkpoint after every pair
        PARTIAL_PATH.parent.mkdir(exist_ok=True)
        PARTIAL_PATH.write_text(json.dumps(existing, indent=2))

        time.sleep(RATE_LIMIT_SLEEP)

    return existing


def node_pair_weight(node_a: dict, node_b: dict, s2_matrix: dict[str, float]) -> float:
    """Compute DTG edge weight between two FORD nodes via their S2 field mappings."""
    fields_a = S2_FIELD_MAP.get(node_a["id"], [])
    fields_b = S2_FIELD_MAP.get(node_b["id"], [])
    if not fields_a or not fields_b:
        return 0.0

    best = 0.0
    for fa in fields_a:
        for fb in fields_b:
            if fa == fb:
                # Same S2 field → strong intra-domain connection
                best = max(best, 0.9)
                continue
            key = "||".join(sorted([fa, fb]))
            best = max(best, s2_matrix.get(key, 0.0))
    return best


def build_graph(s2_matrix: dict[str, float]) -> tuple[list[dict], list[dict]]:
    nodes = FORD_NODES
    edges = []

    pairs = list(combinations(nodes, 2))
    print(f"Building {len(pairs)} node-pair edges from S2 matrix…")

    for a, b in pairs:
        w = node_pair_weight(a, b, s2_matrix)
        if w >= MIN_WEIGHT:
            # Add both directions for a directed graph
            edges.append({
                "source": a["id"],
                "target": b["id"],
                "weight": round(w, 4),
                "relationship": "co-occurrence",
            })
            edges.append({
                "source": b["id"],
                "target": a["id"],
                "weight": round(w, 4),
                "relationship": "co-occurrence",
            })

    return nodes, edges


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Load existing partial results
    existing_s2: dict[str, float] = {}
    if PARTIAL_PATH.exists():
        existing_s2 = json.loads(PARTIAL_PATH.read_text())
        print(f"Resuming from {len(existing_s2)} cached field pairs.")

    # Compute S2 co-occurrence matrix
    s2_matrix = build_s2_matrix(existing_s2)

    # Build nodes and edges
    nodes, edges = build_graph(s2_matrix)

    dtg = {
        "metadata": {
            "version": "1.0",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "min_weight": MIN_WEIGHT,
            "source": "OECD FORD taxonomy + Semantic Scholar co-occurrence",
        },
        "nodes": nodes,
        "edges": edges,
    }

    OUTPUT_PATH.write_text(json.dumps(dtg, indent=2))
    print(f"\nDTG saved → {OUTPUT_PATH}")
    print(f"  Nodes: {len(nodes)}  |  Edges: {len(edges)}")


if __name__ == "__main__":
    main()
