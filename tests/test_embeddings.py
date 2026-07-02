"""Tests for embedding generation."""

from app.embeddings.encoder import EmbeddingEncoder


def test_hash_embedding_generation_shape():
    encoder = EmbeddingEncoder("unused", backend="hash")

    vectors = encoder.encode(["python fastapi", "excel reporting"])

    assert vectors.shape == (2, 384)
    assert float(vectors[0].sum()) > 0
