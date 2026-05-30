"""Component 5: Parallel Epistemic Sampler with cross-contamination check."""
from __future__ import annotations

from efa.config import CONTAMINATION_THRESHOLD_RESP
from efa.embeddings import embed
from efa.llm import LLMClient
from efa.pipeline.mea import MEAResult


class ParallelEpistemicSampler:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    def sample(
        self,
        queries: list[str],
        mea_result: MEAResult,
    ) -> list[tuple[str, bool]]:
        """
        Run queries in parallel via asyncio.
        Returns list of (response_text, is_contaminated).
        is_contaminated = True if response is too similar to original frame.
        """
        system = (
            "Respond within the domain specified. "
            "Do not reference the original problem's terminology or bridge back to other frameworks."
        )

        responses = self._llm.complete_parallel(queries, system=system, temperature=0.8)

        frame_vec = embed.encode(mea_result.frame_text)
        results: list[tuple[str, bool]] = []

        for resp in responses:
            if not resp:
                results.append(("", True))
                continue
            resp_vec = embed.encode(resp[:2000])  # clip for efficiency
            sim = embed.cosine_sim(frame_vec, resp_vec)
            contaminated = sim > CONTAMINATION_THRESHOLD_RESP
            results.append((resp, contaminated))

        return results
