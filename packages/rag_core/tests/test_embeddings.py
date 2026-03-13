from __future__ import annotations

from rag_core.impl.embeddings_e5 import HashingEmbedder


def test_hashing_embedder_empty_batch_shape() -> None:
    embedder = HashingEmbedder(dimensions=32)
    vectors = embedder.embed_documents([])
    assert vectors.shape == (0, 32)
