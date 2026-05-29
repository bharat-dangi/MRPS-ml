import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Scorer:
    """Thin wrapper around BM25Okapi for candidate corpus scoring."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None

    def fit(self, corpus: list[str]) -> None:
        tokenized = [_tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def score(self, query: str) -> np.ndarray:
        """Return per-document BM25 scores normalised to [0, 1]."""
        if self._bm25 is None:
            raise RuntimeError("BM25Scorer.fit() must be called before score()")
        raw = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        max_val = raw.max()
        if max_val > 0:
            return raw / max_val
        return raw
