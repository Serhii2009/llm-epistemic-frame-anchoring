"""Component 1: Meta-Epistemic Analyzer.

Extracts (frame_text, problem_skeleton) from the user's prompt.
Frame = the ontological commitments encoded in the prompt.
Skeleton = frame-stripped structural problem representation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from efa.config import CONTAMINATION_THRESHOLD_SP
from efa.dtg.nodes import FORD_NODES
from efa.embeddings import embed
from efa.llm import LLMClient

# All valid DTG node IDs — injected into the prompt at runtime so the LLM
# can pick one exactly rather than free-form text that may not map to a node.
_ALL_NODE_IDS = [n["id"] for n in FORD_NODES]
_NODE_ID_LIST = ", ".join(f'"{nid}"' for nid in _ALL_NODE_IDS)

_SYSTEM_TEMPLATE = """\
You are an analytical assistant. Your task is to separate a problem statement into two components:

1. FRAME: the DISCIPLINE being applied to analyze or solve the problem — the conceptual lens, methodology, or professional field from which the problem is being approached. This is NOT the subject matter; it is the analytic tradition.
   Example: "How do I reduce employee turnover in my engineering team?" — the FRAME is "management/HR" (the discipline applied), NOT "software engineering" (the subject of the team).
   Example: "How do I fix a memory leak in my Python code?" — the FRAME is "software engineering".

2. SKELETON: a frame-independent structural description of the actual problem — stated in neutral terms without the original domain vocabulary.

Return ONLY valid JSON in this exact schema:
{{
  "frame_summary": "one-sentence description of the discipline/field being applied (e.g., 'software engineering practices', 'HR management')",
  "frame_concepts": ["key concept 1", "key concept 2"],
  "primary_domain_id": "the single most specific node ID from the list below that describes the applied discipline. Must be EXACTLY one of the listed IDs.",
  "skeleton": {{
    "goals": ["what outcome needs to be achieved, in neutral language"],
    "constraints": ["what cannot be violated or changed"],
    "entities": ["the core agents, systems, or resources involved — described functionally, not by domain label"],
    "success_criteria": ["how you would know the problem is solved"]
  }}
}}

Valid primary_domain_id values (pick the closest match):
{node_id_list}

Examples for primary_domain_id:
  "Why do school reform initiatives fail?" → "soc_edu"
  "My ML model overfits" → "cross_ai"
  "How do I speed up my Postgres queries?" → "nat_cs"
  "How do I reduce employee turnover?" → "cross_orgbeh"
  "How do I price my SaaS product?" → "soc_econ"
  "Why does my city have traffic jams?" → "cross_urban"

Rules for the skeleton:
- Do NOT use the discipline-specific vocabulary from the frame
- Describe goals/constraints in the most general functional terms
- "engineering team" → "skilled technical contributors"; "API" → "interface"; "database" → "persistent data store"
- The skeleton should read as a problem description a generalist could understand
"""

_SYSTEM = _SYSTEM_TEMPLATE.format(node_id_list=_NODE_ID_LIST)


@dataclass
class MEAResult:
    frame_summary: str
    frame_concepts: list[str]
    skeleton: dict
    skeleton_text: str
    frame_text: str
    primary_domain_id: str = ""          # DTG node ID of the primary domain
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
                parsed = self._llm.complete_json(
                    prompt + stronger_instruction,
                    system=_SYSTEM,
                )
            except (json.JSONDecodeError, Exception):
                continue

            skeleton = parsed.get("skeleton", {})
            skeleton_text = self._skeleton_to_text(skeleton)

            # Contamination check
            frame_vec = embed.encode(frame_text)
            skeleton_vec = embed.encode(skeleton_text)
            sim = embed.cosine_sim(frame_vec, skeleton_vec)

            # Validate primary_domain_id — must be one of the known node IDs
            raw_pid = parsed.get("primary_domain_id", "")
            primary_domain_id = raw_pid if raw_pid in _ALL_NODE_IDS else ""

            result = MEAResult(
                frame_summary=parsed.get("frame_summary", ""),
                frame_concepts=parsed.get("frame_concepts", []),
                skeleton=skeleton,
                skeleton_text=skeleton_text,
                frame_text=frame_text,
                primary_domain_id=primary_domain_id,
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
                primary_domain_id="",
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
