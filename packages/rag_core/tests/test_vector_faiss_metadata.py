from __future__ import annotations

import json

from rag_core.impl.vector_faiss import FaissVectorStore


def test_load_documents_from_jsonl(tmp_path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "alpha", "metadata": {"subject": "math"}}),
                json.dumps({"id": "b", "text": "beta", "metadata": {"subject": "science"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    documents = FaissVectorStore._load_documents(path)
    assert [document.id for document in documents] == ["a", "b"]


def test_load_documents_from_legacy_json(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {"id": "a", "text": "alpha", "metadata": {"subject": "math"}},
                    {"id": "b", "text": "beta", "metadata": {"subject": "science"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = FaissVectorStore._load_documents(path)
    assert [document.id for document in documents] == ["a", "b"]
