"""Embedding generation with Sentence Transformers and a local fallback."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Generate normalized embeddings for semantic comparison."""

    def __init__(self, model_name: str, backend: str = "auto"):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        self._fallback = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        if backend in {"auto", "sentence-transformers"}:
            self._load_sentence_transformer()

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        """Encode one or more texts into L2-normalized vectors."""

        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)
        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        if self._model is not None:
            vectors = self._model.encode(
                text_list,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vectors = self._fallback.transform(text_list).astype(np.float32).toarray()
            vectors = normalize(vectors, norm="l2")

        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors[0] if is_single else vectors

    def _load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            if self.backend == "sentence-transformers":
                raise
            logger.warning(
                "Sentence Transformer unavailable; using hashing embeddings: %s", exc
            )
            self._model = None
