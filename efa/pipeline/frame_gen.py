"""Component 4: Contrastive Frame Generator."""
from __future__ import annotations

from efa.dtg.graph import Node
from efa.llm import LLMClient
from efa.pipeline.mea import MEAResult


class ContrastiveFrameGenerator:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    def generate(
        self,
        mea_result: MEAResult,
        gaps: list[Node],
    ) -> list[str]:
        """
        For each coverage gap domain, generate a domain-anchored reformulation
        of the problem. One LLM call per gap.
        """
        queries = []
        for gap in gaps:
            query = self._build_query(mea_result, gap)
            queries.append(query)

        # k_gaps LLM calls (non-parallel at this stage; parallelism is in Component 5)
        results = []
        for query in queries:
            resp = self._llm.complete(query, temperature=0.7)
            results.append(resp)

        return results

    def _build_query(self, mea_result: MEAResult, gap: Node) -> str:
        sp = mea_result.skeleton_text or str(mea_result.skeleton)
        frame_domain = mea_result.frame_summary or "the original domain"
        return (
            f"You are an expert in {gap.domain}.\n\n"
            f"A problem has been described in terms of {frame_domain}. "
            f"I want you to approach it through the lens of {gap.domain} — "
            f"{gap.description}.\n\n"
            f"Problem structure (domain-neutral):\n{sp}\n\n"
            f"Provide: (1) key concepts from {gap.domain} that are relevant here, "
            f"(2) how a {gap.domain} expert would frame this problem differently, "
            f"(3) what solutions or approaches from {gap.domain} might apply.\n\n"
            f"Focus strictly on {gap.domain} concepts. "
            f"Do not bridge back to {frame_domain} terminology."
        )
