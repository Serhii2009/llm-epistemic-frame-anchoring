"""Component 8: Coverage-Aware Synthesis.

Generates final response integrating in-frame knowledge + outside-frame concepts,
with explicit epistemic provenance and known-gaps section.
"""
from __future__ import annotations

from efa.dtg.graph import Node
from efa.llm import LLMClient
from efa.pipeline.delta import DeltaConcept
from efa.pipeline.mea import MEAResult

_SYSTEM = """\
You are a knowledgeable assistant generating a comprehensive response.
Structure your response in three clearly labeled sections:

**In-Frame Answer**
Answer the question using standard knowledge within its disciplinary context.

**Outside-Frame Considerations**
For each outside-frame concept listed below, explain in 2-3 sentences how it applies
to the problem. Label each with its source domain in brackets: [Domain Name]

**Known Coverage Gaps**
List the domains that were NOT explored due to exploration limits. Be brief.

If there was a contamination warning, prepend: [WARNING] Frame extraction was imperfect;
some outside-frame concepts may retain original-domain vocabulary.
"""


class CoverageAwareSynthesis:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    def synthesize(
        self,
        original_prompt: str,
        mea_result: MEAResult,
        relevant_concepts: list[DeltaConcept],
        remaining_gaps: list[Node],
        contamination_warning: bool = False,
    ) -> str:
        concepts_block = self._format_concepts(relevant_concepts)
        gaps_block = ", ".join(n.domain for n in remaining_gaps) if remaining_gaps else "none identified"

        prompt = (
            f"Original question: {original_prompt}\n\n"
            f"Outside-frame concepts to integrate:\n{concepts_block}\n\n"
            f"Unexplored domains (known gaps): {gaps_block}\n\n"
            f"Contamination warning: {'yes' if contamination_warning else 'no'}\n\n"
            "Generate the three-section response now."
        )

        return self._llm.complete(prompt, system=_SYSTEM, temperature=0.4)

    @staticmethod
    def _format_concepts(concepts: list[DeltaConcept]) -> str:
        if not concepts:
            return "(none identified)"
        lines = []
        for dc in concepts:
            lines.append(f"- [{dc.source_domain}] {dc.concept} (relevance score: {dc.score:.3f})")
        return "\n".join(lines)
