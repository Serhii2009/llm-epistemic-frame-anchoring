"""Local embedding model — BAAI/bge-large-en-v1.5, no API key required.

NOTE: the mission asked for BGE-M3, but BGE-M3 (~560M params) is too slow on the
researcher's CPU-only laptop for chunk-level scoring of hundreds of long responses.
We use bge-large-en-v1.5 (~335M params, 1024-dim) instead — the model the original
repo shipped with — purely for hardware tractability. This is documented in the
final verdict. All distances and similarities in the study are computed with this
single model, so internal comparisons remain consistent.
"""
from __future__ import annotations

import re
import numpy as np
from typing import Union


class EmbeddingModel:
    """Lazy singleton wrapper around BAAI/bge-large-en-v1.5."""

    _instance: "EmbeddingModel | None" = None
    _model = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading BAAI/bge-large-en-v1.5…")
            self._model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            print("Embedding model ready.")

    def encode(self, text: Union[str, list[str]]) -> np.ndarray:
        self._load()
        single = isinstance(text, str)
        inputs = [text] if single else text
        vecs = self._model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        return vecs[0] if single else vecs

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for normalized vectors (= dot product)."""
        return float(np.dot(a, b))

    def cosine_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - self.cosine_sim(a, b)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Cheap sentence/line splitter — no extra deps."""
        # Split on sentence terminators and newlines/bullets; keep chunks of useful length.
        parts = re.split(r"(?<=[.!?])\s+|\n+|[;•]\s+", text)
        return [p.strip() for p in parts if len(p.strip()) >= 12]

    def best_chunk_sim(self, concept: str, response_text: str) -> float:
        """
        Max cosine similarity between a concept and any sentence-level chunk of a
        response. Chunk-level (not whole-document) matching avoids dilution: a single
        sentence invoking the concept should score high even in a long answer.
        Returns 0.0 if the response has no usable chunks.
        """
        chunks = self._sentences(response_text)
        if not chunks:
            return 0.0
        cvec = self.encode(concept)
        chunk_vecs = self.encode(chunks)
        sims = chunk_vecs @ cvec
        return float(np.max(sims))

    def chunk_matrix(self, response_text: str) -> np.ndarray | None:
        """Embed a response's sentence chunks ONCE; returns (n_chunks, dim) or None.

        Use with `concept_sims` to score many concepts against one response without
        re-embedding the response per concept (the slow path in best_chunk_sim).
        """
        chunks = self._sentences(response_text)
        if not chunks:
            return None
        return self.encode(chunks)

    def concept_sims(self, concepts: list[str], chunk_vecs: np.ndarray | None) -> list[float]:
        """Max-over-chunks cosine for each concept against a precomputed chunk matrix."""
        if chunk_vecs is None or len(chunk_vecs) == 0:
            return [0.0] * len(concepts)
        cvecs = self.encode(concepts)            # (n_concepts, dim)
        sims = cvecs @ chunk_vecs.T              # (n_concepts, n_chunks)
        return [float(np.max(row)) for row in sims]


# Module-level singleton — import this everywhere
embed = EmbeddingModel()
