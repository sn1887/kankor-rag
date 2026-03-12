from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

import numpy as np

from rag_core.contracts.vector_store import VectorStore
from rag_core.types import Document, Hit


class FaissVectorStore(VectorStore):
    def __init__(self, index, documents: list[Document]) -> None:
        self.index = index
        self.documents = documents

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector: np.ndarray, *, top_k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        if self.size == 0 or top_k <= 0:
            return []
        import faiss

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(max(top_k * 4, top_k), self.size))
        hits: list[Hit] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            document = self.documents[int(index)]
            if filters and any(str(document.metadata.get(k)) != str(v) for k, v in filters.items()):
                continue
            hits.append(Hit(document=document, score=float(score)))
            if len(hits) >= top_k:
                break
        return hits

    @classmethod
    def build(cls, embeddings: np.ndarray, documents: list[Document]) -> 'FaissVectorStore':
        if not documents:
            raise ValueError("Cannot build FAISS index with zero documents.")
        import faiss

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings matrix must be 2-dimensional.")
        if matrix.shape[0] != len(documents):
            raise ValueError(
                f"Embeddings/document mismatch: {matrix.shape[0]} vectors for {len(documents)} documents."
            )
        if matrix.shape[1] <= 0:
            raise ValueError("Embedding dimension must be > 0.")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, documents=documents)

    @staticmethod
    def _document_to_dict(document: Document) -> dict[str, object]:
        return {
            "id": document.id,
            "text": document.text,
            "metadata": document.metadata,
        }

    @classmethod
    def _load_documents_legacy_json(cls, metadata_path: Path) -> list[Document] | None:
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None
        records = raw.get("documents")
        if not isinstance(records, list):
            return None
        return [Document(**item) for item in records]

    @classmethod
    def _load_documents_jsonl(cls, metadata_path: Path) -> list[Document]:
        documents: list[Document] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                documents.append(Document(**row))
        return documents

    @classmethod
    def _load_documents(cls, metadata_path: Path) -> list[Document]:
        if metadata_path.suffix.lower() == ".jsonl":
            return cls._load_documents_jsonl(metadata_path)

        first_nonempty = ""
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                first_nonempty = line.strip()
                if first_nonempty:
                    break

        if first_nonempty and not first_nonempty.startswith("{"):
            return cls._load_documents_jsonl(metadata_path)

        legacy = cls._load_documents_legacy_json(metadata_path)
        if legacy is not None:
            return legacy
        return cls._load_documents_jsonl(metadata_path)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> 'FaissVectorStore':
        import faiss

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index = faiss.read_index(str(index_path))
        documents = cls._load_documents(metadata_path)
        return cls(index=index, documents=documents)

    def _save_jsonl_metadata(self, metadata_path: Path, rows: Iterable[dict[str, object]]) -> None:
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_json_metadata(self, metadata_path: Path, rows: list[dict[str, object]]) -> None:
        metadata_path.write_text(
            json.dumps({"documents": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        import faiss

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        rows = [self._document_to_dict(document) for document in self.documents]

        if metadata_path.suffix.lower() == ".jsonl":
            self._save_jsonl_metadata(metadata_path, rows)
            return

        self._save_json_metadata(metadata_path, rows)
