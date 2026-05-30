"""Component 3: Coverage Estimator — pure graph query, no LLM call."""
from __future__ import annotations

from efa.config import FRAME_MAP_TOP_K, DTG_K_HOPS, K_GAPS
from efa.dtg.graph import DTG, Node
from efa.pipeline.mea import MEAResult


class CoverageEstimator:
    def __init__(self, dtg: DTG):
        self._dtg = dtg

    def estimate(
        self,
        mea_result: MEAResult,
        k_gaps: int = K_GAPS,
    ) -> tuple[list[Node], list[Node], set[str]]:
        """
        Returns:
            activated_nodes: DTG nodes activated by the frame
            gap_nodes:       top-k unactivated nodes ranked by relevance to S(P)
            footprint:       full activation footprint (node IDs)
        """
        # Use extracted frame summary if available; fall back to raw prompt.
        # The summary ("HR management") maps to correct domains;
        # the raw prompt ("engineering team") activates wrong technical domains.
        frame_for_mapping = (
            mea_result.frame_summary
            if mea_result.frame_summary
            else mea_result.frame_text
        )
        activated = self._dtg.map_frame(
            frame_for_mapping,
            top_k=FRAME_MAP_TOP_K,
        )

        footprint = self._dtg.get_activation_footprint(activated, k_hops=DTG_K_HOPS)

        gaps = self._dtg.rank_gaps(
            footprint,
            mea_result.skeleton_text,
            top_k=k_gaps,
        )

        return activated, gaps, footprint
