from __future__ import annotations

import numpy as np
import pytest

from rag_core.impl.faiss_streaming import StreamingFaissArtifactWriter
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.types import Document


def test_streaming_faiss_writer_persists_index_and_metadata(tmp_path) -> None:
    pytest.importorskip("faiss")
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.jsonl"
    documents = [
        Document(id="a", text="alpha", metadata={"subject": "math"}),
        Document(id="b", text="beta", metadata={"subject": "science"}),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with StreamingFaissArtifactWriter(index_path=index_path, metadata_path=metadata_path) as writer:
        writer.add_batch(embeddings=embeddings, documents=documents)
        writer.save_index()
        assert writer.dimension == 2
        assert writer.vectors == 2

    store = FaissVectorStore.load(index_path=index_path, metadata_path=metadata_path)
    assert store.size == 2
    hits = store.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)
    assert hits and hits[0].document.id == "a"

