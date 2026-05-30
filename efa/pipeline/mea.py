"""Component 1: Meta-Epistemic Analyzer.

Extracts (frame_text, problem_skeleton) from the user's prompt.
Frame = the ontological commitments encoded in the prompt.
Skeleton = frame-stripped structural problem representation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from efa.config import CONTAMINATION_THRESHOLD_SP
from efa.embeddings import embed
from efa.llm import LLMClient

_SYSTEM = """\
You are an analytical assistant. Your task is to separate a problem statement into two components:

1. FRAME: the disciplinary vocabulary, ontological assumptions, and domain commitments encoded in the prompt
2. SKELETON: a frame-independent structural description of the actual problem — stated in neutral terms without the original domain vocabulary

Return ONLY valid JSON in this exact schema:
{
  "frame_summary": "brief description of the frame's disciplinary commitments",
  "frame_concepts": ["key concept 1", "key concept 2", "..."],
  "skeleton": {
    "goals": ["what needs to be achieved"],
    "constraints": ["what cannot be violated"],
    "entities": ["core objects, agents, or systems involved"],
    "success_criteria": ["how you would know the problem is solved"]
  }
}

Rules for the skeleton:
- Do NOT use vocabulary from the frame's domain
- Describe goals/constraints in the most general terms possible
- If an entity is domain-specific (e.g., "API endpoint"), re-describe it functionally (e.g., "communication interface")
"""


@dataclass
class MEAResult:
    frame_summary: str
    frame_concepts: list[str]
    skeleton: dict
    skeleton_text: str
    frame_text: str
    contamination_warning: bool = False
    contamination_sim: float = 0.0


class MetaEpistemicAnalyzer:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    def analyze(self, prompt: str, max_retries: int = 3) -> MEAResult:
        frame_text = prompt  # frame_text is the original prompt

        best_result: MEAResult | None = None
        best_sim = 1.0

        for attempt in range(max_retries):
            stronger_instruction = (
                "\n\nIMPORTANT: The skeleton MUST NOT use any domain-specific vocabulary "
                "from the original prompt. Replace every domain term with a neutral functional description."
                if attempt > 0 else ""
            )
            try:
                raw = self._llm.complete(
                    prompt + stronger_instruction,
                    system=_SYSTEM,
                    json_mode=True,
                    temperature=0.2,
                )
                parsed = json.loads(raw)
            except (json.JSONDecodeError, Exception):
                continue

            skeleton = parsed.get("skeleton", {})
            skeleton_text = self._skeleton_to_text(skeleton)

            # Contamination check
            frame_vec = embed.encode(frame_text)
            skeleton_vec = embed.encode(skeleton_text)
            sim = embed.cosine_sim(frame_vec, skeleton_vec)

            result = MEAResult(
                frame_summary=parsed.get("frame_summary", ""),
                frame_concepts=parsed.get("frame_concepts", []),
                skeleton=skeleton,
                skeleton_text=skeleton_text,
                frame_text=frame_text,
                contamination_warning=sim >= CONTAMINATION_THRESHOLD_SP,
                contamination_sim=sim,
            )

            if sim < CONTAMINATION_THRESHOLD_SP:
                return result  # Clean extraction

            if best_result is None or sim < best_sim:
                best_result = result
                best_sim = sim

        # All retries exhausted — return best available with warning
        if best_result is None:
            best_result = MEAResult(
                frame_summary="",
                frame_concepts=[],
                skeleton={},
                skeleton_text=prompt[:500],
                frame_text=frame_text,
                contamination_warning=True,
                contamination_sim=1.0,
            )
        best_result.contamination_warning = True
        return best_result

    @staticmethod
    def _skeleton_to_text(skeleton: dict) -> str:
        parts = []
        for key, values in skeleton.items():
            if values:
                parts.append(f"{key}: {'; '.join(str(v) for v in values)}")
        return ". ".join(parts)
