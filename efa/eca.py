"""ECA — Epistemic Coverage Architecture orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from efa.config import DEFAULT_MODEL, DTG_PATH, K_GAPS, CAUSAL_PROBE_TOP_N
from efa.dtg.graph import DTG, Node
from efa.llm import LLMClient
from efa.pipeline.mea import MetaEpistemicAnalyzer, MEAResult
from efa.pipeline.coverage import CoverageEstimator
from efa.pipeline.frame_gen import ContrastiveFrameGenerator
from efa.pipeline.sampler import ParallelEpistemicSampler
from efa.pipeline.delta import ConceptDeltaExtractor, DeltaConcept
from efa.pipeline.probe import CausalStructureProbe
from efa.pipeline.synthesis import CoverageAwareSynthesis


@dataclass
class ECAResult:
    response: str
    concepts: list[DeltaConcept]
    activated_nodes: list[Node]
    gap_nodes: list[Node]
    coverage_audit: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Activated frame domains: {[n.domain for n in self.activated_nodes]}",
            f"Explored gap domains: {[n.domain for n in self.gap_nodes]}",
            f"Outside-frame concepts surfaced: {len(self.concepts)}",
            f"Contamination warning: {self.coverage_audit.get('contamination_warning', False)}",
        ]
        if self.concepts:
            lines.append("Top concepts:")
            for dc in self.concepts[:5]:
                lines.append(f"  [{dc.source_domain}] {dc.concept} (score={dc.score:.3f})")
        return "\n".join(lines)


class ECA:
    """
    Epistemic Coverage Architecture.

    Orchestrates all 9 components to produce a coverage-aware response
    that surfaces outside-frame concepts the user didn't know to ask about.
    """

    def __init__(
        self,
        dtg_path: Path | str = DTG_PATH,
        model: str = DEFAULT_MODEL,
        k_gaps: int = K_GAPS,
    ):
        self._dtg = DTG(dtg_path)
        self._llm = LLMClient(model=model)
        self._k_gaps = k_gaps

        self._mea = MetaEpistemicAnalyzer(self._llm)
        self._coverage = CoverageEstimator(self._dtg)
        self._frame_gen = ContrastiveFrameGenerator(self._llm)
        self._sampler = ParallelEpistemicSampler(self._llm)
        self._delta = ConceptDeltaExtractor(self._llm)
        self._probe = CausalStructureProbe(self._llm)
        self._synthesis = CoverageAwareSynthesis(self._llm)

    def run(self, prompt: str) -> ECAResult:
        """Full ECA pipeline: prompt → coverage-aware response."""

        # 1. Extract frame + skeleton
        mea = self._mea.analyze(prompt)

        # 3. Coverage estimation (pure graph, no LLM)
        activated, gaps, footprint = self._coverage.estimate(mea, k_gaps=self._k_gaps)

        if not gaps:
            # No coverage gaps found — return vanilla response with audit note
            response = self._llm.complete(prompt)
            return ECAResult(
                response=response,
                concepts=[],
                activated_nodes=activated,
                gap_nodes=[],
                coverage_audit={
                    "contamination_warning": mea.contamination_warning,
                    "note": "No coverage gaps identified; full domain coverage within DTG.",
                },
            )

        # 4. Generate domain-anchored queries
        gap_queries = self._frame_gen.generate(mea, gaps)

        # 5. Parallel sampling across gap domains
        sampler_results = self._sampler.sample(gap_queries, mea)

        # 6. Extract delta concepts
        delta = self._delta.extract(prompt, sampler_results, gaps, mea)

        # 7. Causal probe filter
        relevant = self._probe.filter(delta, mea, top_n=CAUSAL_PROBE_TOP_N)

        # Remaining gaps (explored but not in top-k activated)
        remaining_gaps = [
            n for n in self._dtg.rank_gaps(footprint, mea.skeleton_text, top_k=10)
            if n not in gaps
        ][:5]

        # 8. Coverage-aware synthesis
        response = self._synthesis.synthesize(
            original_prompt=prompt,
            mea_result=mea,
            relevant_concepts=relevant,
            remaining_gaps=remaining_gaps,
            contamination_warning=mea.contamination_warning,
        )

        contamination_flags = [contaminated for _, contaminated in sampler_results]

        return ECAResult(
            response=response,
            concepts=relevant,
            activated_nodes=activated,
            gap_nodes=gaps,
            coverage_audit={
                "contamination_warning": mea.contamination_warning,
                "contamination_sim": mea.contamination_sim,
                "frame_summary": mea.frame_summary,
                "frame_concepts": mea.frame_concepts,
                "activated_domains": [n.domain for n in activated],
                "explored_gap_domains": [n.domain for n in gaps],
                "remaining_unexplored": [n.domain for n in remaining_gaps],
                "sampler_contamination_flags": contamination_flags,
                "delta_concepts_before_probe": len(delta),
                "concepts_after_probe": len(relevant),
            },
        )
