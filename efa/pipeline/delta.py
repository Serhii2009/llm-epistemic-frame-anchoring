"""Component 6: Concept Delta Extractor.

Extracts concepts present in outside-frame responses but absent from the
vanilla in-frame response (C₀). Uses spaCy NP chunking + cosine filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from efa.config import DELTA_EXCLUSION_THRESHOLD
from efa.dtg.graph import Node
from efa.embeddings import embed
from efa.llm import LLMClient
from efa.pipeline.mea import MEAResult


@dataclass
class DeltaConcept:
    concept: str
    source_domain: str          # gap domain that produced this concept
    score: float = 0.0
    distance_from_c0: float = 0.0
    freq_across_frames: int = 0
    sp_alignment: float = 0.0
    contaminated: bool = False


class ConceptDeltaExtractor:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()
        self._nlp = None  # lazy load spaCy

    def _load_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown formatting so spaCy sees clean prose, not syntax tokens."""
        text = re.sub(r"```[\s\S]*?```", " ", text)            # fenced code blocks
        text = re.sub(r"`[^`]+`", " ", text)                   # inline code
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)               # italic
        text = re.sub(r"__(.+?)__", r"\1", text)               # bold alt
        text = re.sub(r"_(.+?)_", r"\1", text)                 # italic alt
        text = re.sub(r"#+\s*", "", text)                      # headings
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [link](url) → text
        text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)   # bullets
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)   # numbered lists
        text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)      # blockquotes
        text = re.sub(r"\n+", " ", text)                       # collapse all remaining newlines to space
        return text.strip()

    def _extract_noun_phrases(self, text: str) -> list[str]:
        self._load_nlp()
        doc = self._nlp(self._strip_markdown(text)[:5000])
        phrases = []
        for chunk in doc.noun_chunks:
            cleaned = chunk.text.strip().lower()
            if 2 <= len(cleaned.split()) <= 5 and len(cleaned) > 4:
                phrases.append(cleaned)
        return list(dict.fromkeys(phrases))  # deduplicate preserving order

    def _in_c0(self, concept: str, c0_vecs: list[np.ndarray]) -> bool:
        if not c0_vecs:
            return False
        concept_vec = embed.encode(concept)
        sims = [embed.cosine_sim(concept_vec, v) for v in c0_vecs]
        return max(sims) >= DELTA_EXCLUSION_THRESHOLD

    def extract(
        self,
        original_prompt: str,
        responses: list[tuple[str, bool]],
        gap_nodes: list[Node],
        mea_result: MEAResult,
    ) -> list[DeltaConcept]:
        """
        Extract delta concepts across all sampler responses.

        C₀: greedy vanilla LLM response to original prompt.
        DELTA = concepts in gap responses NOT in C₀.
        """
        # Get C₀ (greedy, T=0) — strip markdown before phrase extraction
        c0_text = self._strip_markdown(self._llm.complete(original_prompt, temperature=0.0))
        c0_phrases = self._extract_noun_phrases(c0_text)
        c0_vecs = [embed.encode(p) for p in c0_phrases] if c0_phrases else []

        sp_vec = embed.encode(mea_result.skeleton_text) if mea_result.skeleton_text else None

        # Collect all delta concepts with metadata
        concept_tracker: dict[str, DeltaConcept] = {}

        for (resp_text, contaminated), gap_node in zip(responses, gap_nodes):
            if not resp_text:
                continue
            phrases = self._extract_noun_phrases(resp_text)
            for phrase in phrases:
                if self._in_c0(phrase, c0_vecs):
                    continue  # already in original frame

                phrase_vec = embed.encode(phrase)
                c0_centroid = np.mean(c0_vecs, axis=0) if c0_vecs else None
                dist = embed.cosine_dist(phrase_vec, c0_centroid) if c0_centroid is not None else 0.5
                sp_align = float(sp_vec @ phrase_vec) if sp_vec is not None else 0.0

                key = phrase
                if key not in concept_tracker:
                    concept_tracker[key] = DeltaConcept(
                        concept=phrase,
                        source_domain=gap_node.domain,
                        distance_from_c0=dist,
                        sp_alignment=max(0.0, sp_align),
                        contaminated=contaminated,
                    )
                concept_tracker[key].freq_across_frames += 1
                if contaminated:
                    concept_tracker[key].contaminated = True

        # Score: freq × dist × sp_alignment × contamination_penalty
        for dc in concept_tracker.values():
            penalty = 0.5 if dc.contaminated else 1.0
            dc.score = dc.freq_across_frames * dc.distance_from_c0 * dc.sp_alignment * penalty

        ranked = sorted(concept_tracker.values(), key=lambda x: x.score, reverse=True)
        return ranked
