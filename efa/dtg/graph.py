"""Domain Topology Graph — load, map frames, traverse, rank coverage gaps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from efa.config import (
    DTG_PATH,
    FRAME_MAP_THRESHOLD,
    FRAME_MAP_TOP_K,
    DTG_K_HOPS,
)
from efa.embeddings import embed


@dataclass
class Node:
    id: str
    domain: str
    division: str
    description: str
    embedding: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"Node({self.id}: {self.domain})"


class DTG:
    """Domain Topology Graph over OECD FORD + cross-disciplinary domains."""

    def __init__(self, path: Path | str = DTG_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"DTG not found at {path}. Run: python scripts/build_dtg.py"
            )
        data = json.loads(path.read_text())
        self._nodes: dict[str, Node] = {}
        self._graph = nx.DiGraph()

        for n in data["nodes"]:
            node = Node(
                id=n["id"],
                domain=n["domain"],
                division=n["division"],
                description=n["description"],
            )
            self._nodes[n["id"]] = node
            self._graph.add_node(n["id"])

        for e in data["edges"]:
            self._graph.add_edge(e["source"], e["target"], weight=e["weight"])

        # Pre-compute node embeddings lazily (on first map_frame call)
        self._embeddings: Optional[np.ndarray] = None
        self._node_ids: list[str] = list(self._nodes.keys())

    def _ensure_embeddings(self):
        if self._embeddings is None:
            descriptions = [self._nodes[nid].description for nid in self._node_ids]
            self._embeddings = embed.encode(descriptions)
            for i, nid in enumerate(self._node_ids):
                self._nodes[nid].embedding = self._embeddings[i]

    def map_frame(
        self,
        frame_text: str,
        top_k: int = FRAME_MAP_TOP_K,
        threshold: float = FRAME_MAP_THRESHOLD,
    ) -> list[Node]:
        """Embed frame_text and return top-k DTG nodes above similarity threshold."""
        self._ensure_embeddings()
        frame_vec = embed.encode(frame_text)
        sims = self._embeddings @ frame_vec
        ranked_idx = np.argsort(sims)[::-1]

        activated = []
        for idx in ranked_idx[:top_k * 3]:  # check wider before threshold filter
            if sims[idx] >= threshold:
                activated.append(self._nodes[self._node_ids[idx]])
            if len(activated) >= top_k:
                break
        return activated

    def get_activation_footprint(
        self,
        activated_nodes: list[Node],
        k_hops: int = DTG_K_HOPS,
    ) -> set[str]:
        """BFS from activated nodes up to k_hops — returns set of node IDs."""
        footprint: set[str] = set()
        frontier = {n.id for n in activated_nodes}
        footprint.update(frontier)

        for _ in range(k_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for neighbor in self._graph.successors(nid):
                    if neighbor not in footprint:
                        next_frontier.add(neighbor)
            footprint.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return footprint

    def rank_gaps(
        self,
        footprint: set[str],
        problem_skeleton_text: str,
        top_k: int = 3,
        frame_summary_text: str = "",
    ) -> list[Node]:
        """
        Score unactivated nodes by structural proximity × S(P) alignment
        × cross-division bonus × frame-repulsion penalty.

        Scoring formula:
          effective_edge   = max(dtg_edge_weight, 0.02)
          cross_div_bonus  = 1.3 if node.division not in activated divisions else 1.0
          frame_repulsion  = max(0.3, 1.0 - 0.5 × cosine_sim(frame_vec, node_vec))
          score = effective_edge × sp_alignment × cross_div_bonus × frame_repulsion

        frame_repulsion addresses the root cause of healthcare-domain false positives:
        Nursing/Health descriptions share institutional vocabulary ("patient care
        quality", "healthcare management") with neutral educational skeletons. A domain
        whose description is semantically similar to the ACTIVE FRAME gets penalised —
        it looks like more-of-the-same, not genuine cross-domain insight.
        frame_sim=0 → no penalty (1.0); frame_sim=0.5 → 25% reduction (0.75).
        """
        self._ensure_embeddings()
        sp_vec = embed.encode(problem_skeleton_text)
        frame_vec = embed.encode(frame_summary_text) if frame_summary_text else None
        unactivated = [nid for nid in self._node_ids if nid not in footprint]

        # Divisions already represented in the active frame footprint
        activated_divisions = {
            self._nodes[nid].division
            for nid in footprint
            if nid in self._nodes
        }

        scored: list[tuple[float, Node]] = []
        for nid in unactivated:
            node = self._nodes[nid]

            # Structural signal: max edge weight from any footprint node to this node
            edge_weight = 0.0
            for fp_nid in footprint:
                w = self._graph.get_edge_data(fp_nid, nid, {}).get("weight", 0.0)
                edge_weight = max(edge_weight, w)

            # Semantic alignment with problem skeleton
            sp_alignment = max(0.0, float(sp_vec @ node.embedding))

            # Cross-division novelty bonus
            cross_div_bonus = 1.3 if node.division not in activated_divisions else 1.0

            # Frame-repulsion: penalise gaps whose descriptions resemble the active frame.
            # A node semantically similar to "educational reform" is likely to produce
            # output similar to what an education expert would already know — not a
            # genuine outside-frame perspective.
            frame_repulsion = 1.0
            if frame_vec is not None and node.embedding is not None:
                frame_sim = max(0.0, float(frame_vec @ node.embedding))
                frame_repulsion = max(0.3, 1.0 - 0.5 * frame_sim)

            effective_edge = max(edge_weight, 0.02)
            score = effective_edge * sp_alignment * cross_div_bonus * frame_repulsion
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)
