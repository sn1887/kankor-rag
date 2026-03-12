from __future__ import annotations

import json

from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource


def test_local_jsonl_corpus_source_streams_documents(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                json.dumps({"id": "doc-1", "text": "hello", "metadata": {"language": "en"}}),
                json.dumps({"id": "doc-2", "text": "world", "metadata": {"language": "fa"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    source = HFDatasetCorpusSource(local_path=str(path))
    documents = list(source.load_documents())
    assert [item.id for item in documents] == ["doc-1", "doc-2"]
