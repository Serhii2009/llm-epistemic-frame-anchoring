"""Local embedding model — BAAI/bge-large-en-v1.5, no API key required."""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Union


class EmbeddingModel:
    """Lazy singleton wrapper around bge-large-en-v1.5."""

    _instance: "EmbeddingModel | None" = None
    _model = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading bge-large-en-v1.5 (first run downloads ~1.3 GB)…")
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


# Module-level singleton — import this everywhere
embed = EmbeddingModel()
