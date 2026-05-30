"""Component 7: Causal Structure Probe.

Filters delta concepts by causal relevance to S(P).
Uses S(P) as anchor — NOT the original frame vocabulary.
Decision rule: concept passes if ANY ONE of 3 causal criteria is met.
"""
from __future__ import annotations

import json

from efa.config import CAUSAL_PROBE_TOP_N
from efa.llm import LLMClient
from efa.pipeline.delta import DeltaConcept
from efa.pipeline.mea import MEAResult

_SYSTEM = """\
You are a causal reasoning assistant. You will be given a problem structure
(in domain-neutral terms) and a concept. Your job is to assess whether the concept
is causally relevant to the problem.

Answer in JSON: {"reasoning": "...", "passes": true/false}

A concept PASSES if it satisfies ANY ONE of:
(a) It causally influences at least one stated goal or constraint
(b) Knowing it would change the recommended approach to achieving the goals
(c) It is a necessary precondition or systematic consequence of the problem

Use ONLY the problem structure provided. Do not rely on the original problem's
domain vocabulary. Be strict: a concept that is merely thematically related
should NOT pass. Only pass if there is a genuine causal or structural link.
"""


class CausalStructureProbe:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    def filter(
        self,
        delta_concepts: list[DeltaConcept],
        mea_result: MEAResult,
        top_n: int = CAUSAL_PROBE_TOP_N,
    ) -> list[DeltaConcept]:
        """Return concepts that pass the causal relevance test (any one criterion)."""
        candidates = delta_concepts[:top_n]
        sp_text = mea_result.skeleton_text or str(mea_result.skeleton)

        passed = []
        for dc in candidates:
            prompt = (
                f"Problem structure:\n{sp_text}\n\n"
                f"Concept to evaluate: '{dc.concept}' (from {dc.source_domain})\n\n"
                "Does this concept satisfy any one of the causal criteria?"
            )
            try:
                result = self._llm.complete_json(prompt, system=_SYSTEM)
                if result.get("passes", False):
                    passed.append(dc)
            except (json.JSONDecodeError, Exception):
                # On parse error, be conservative and skip
                continue

        return passed
