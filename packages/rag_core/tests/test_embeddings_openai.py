from __future__ import annotations

from rag_core.impl.embeddings_openai import OpenAIEmbedder


class _FakeEmbeddingRow:
    def __init__(self, index: int | None, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, *, with_none_index: bool = False) -> None:
        first_index = None if with_none_index else 1
        self.data = [
            _FakeEmbeddingRow(index=first_index, embedding=[0.0, 1.0]),
            _FakeEmbeddingRow(index=0, embedding=[1.0, 0.0]),
        ]


class _FakeEmbeddings:
    def __init__(self, *, with_none_index: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.with_none_index = with_none_index

    def create(self, **payload):
        self.calls.append(payload)
        if "dimensions" in payload:
            raise Exception("dimensions is unsupported for this model")
        return _FakeEmbeddingResponse(with_none_index=self.with_none_index)


class _FakeClient:
    def __init__(self, *, with_none_index: bool = False) -> None:
        self.embeddings = _FakeEmbeddings(with_none_index=with_none_index)


def test_openai_embedder_retries_without_dimensions() -> None:
    embedder = OpenAIEmbedder(model_name="test-embedding-model", dimensions=256)
    embedder._client = _FakeClient()

    vectors = embedder.embed_documents(["a", "b"])

    calls = embedder._client.embeddings.calls  # type: ignore[union-attr]
    assert vectors.shape == (2, 2)
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert len(calls) == 2
    assert "dimensions" in calls[0]
    assert "dimensions" not in calls[1]


def test_openai_embedder_handles_none_index_rows() -> None:
    embedder = OpenAIEmbedder(model_name="test-embedding-model")
    embedder._client = _FakeClient(with_none_index=True)

    vectors = embedder.embed_documents(["a", "b"])

    assert vectors.shape == (2, 2)
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
