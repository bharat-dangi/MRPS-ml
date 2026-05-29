import os

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = os.getenv("SBERT_MODEL", "all-mpnet-base-v2")


class ResumeEmbedder:
    """Singleton wrapper around sentence-transformers for 768-dim SBERT embeddings.

    normalize_embeddings=True means dot product == cosine similarity,
    which lets pgvector's <=> operator work correctly.
    """

    _instance: "ResumeEmbedder | None" = None
    _model: SentenceTransformer

    def __new__(cls) -> "ResumeEmbedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(_MODEL_NAME)
        return cls._instance

    def encode(self, text: str) -> np.ndarray:
        """Return a 768-dim normalised float32 vector."""
        return self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Batch encode for efficiency during ranking."""
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
