from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import numpy as np

from rag_core.types import Document


class StreamingFaissArtifactWriter:
    def __init__(self, *, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._index = None
        self._metadata_handle: TextIO | None = None
        self._json_started = False
        self._json_first_row = True
        self._vectors = 0
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def vectors(self) -> int:
        return self._vectors

    def __enter__(self) -> "StreamingFaissArtifactWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_handle = self.metadata_path.open("w", encoding="utf-8")
        if self.metadata_path.suffix.lower() == ".json":
            self._metadata_handle.write('{"documents":[\n')
            self._json_started = True

    def close(self) -> None:
        handle = self._metadata_handle
        if handle is None:
            return
        if self._json_started:
            handle.write("\n]}\n")
        handle.close()
        self._metadata_handle = None

    @staticmethod
    def _document_to_row(document: Document) -> dict[str, object]:
        return {
            "id": document.id,
            "text": document.text,
            "metadata": document.metadata,
        }

    def _write_row(self, row: dict[str, object]) -> None:
        if self._metadata_handle is None:
            raise RuntimeError("Metadata writer is not open.")
        if self.metadata_path.suffix.lower() == ".jsonl":
            self._metadata_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        if not self._json_first_row:
            self._metadata_handle.write(",\n")
        self._metadata_handle.write("  " + json.dumps(row, ensure_ascii=False))
        self._json_first_row = False

    def add_batch(self, *, embeddings: np.ndarray, documents: list[Document]) -> None:
        if not documents:
            return
        if embeddings.ndim != 2:
            raise ValueError("Embeddings matrix must be 2-dimensional.")
        if embeddings.shape[0] != len(documents):
            raise ValueError(
                f"Embeddings/document mismatch: {embeddings.shape[0]} vectors for {len(documents)} documents."
            )
        if embeddings.shape[1] <= 0:
            raise ValueError("Embedding dimension must be > 0.")

        import faiss

        matrix = np.asarray(embeddings, dtype=np.float32)
        if self._index is None:
            self._dimension = int(matrix.shape[1])
            self._index = faiss.IndexFlatIP(self._dimension)
        elif self._dimension != int(matrix.shape[1]):
            raise ValueError(
                "Embedding dimension changed across batches: "
                f"expected {self._dimension}, got {matrix.shape[1]}."
            )

        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._vectors += int(matrix.shape[0])

        for document in documents:
            self._write_row(self._document_to_row(document))

    def save_index(self) -> None:
        if self._index is None:
            raise ValueError("Cannot save FAISS index with zero vectors.")
        import faiss

        faiss.write_index(self._index, str(self.index_path))

