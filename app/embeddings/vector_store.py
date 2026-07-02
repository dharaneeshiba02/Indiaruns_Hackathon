"""Vector search abstraction backed by FAISS when available."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store for cosine similarity search."""

    def __init__(self, vectors: np.ndarray, ids: list[str]):
        if len(vectors) != len(ids):
            raise ValueError("Vector count must match id count.")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = ids
        self._index = None
        self._load_faiss()

    def search(self, query: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return ids and similarity scores ordered from best to worst."""

        if self.vectors.size == 0:
            return []
        k = min(top_k or len(self.ids), len(self.ids))
        query_2d = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            scores, indices = self._index.search(query_2d, k)
            return [
                (self.ids[int(index)], float(score))
                for index, score in zip(indices[0], scores[0], strict=False)
                if int(index) >= 0
            ]

        scores = (self.vectors @ query_2d.T).reshape(-1)
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in order]

    def _load_faiss(self) -> None:
        try:
            import faiss

            dimension = self.vectors.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(self.vectors)
        except Exception as exc:
            logger.warning("FAISS unavailable; using NumPy vector search: %s", exc)
            self._index = None
