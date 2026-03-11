from __future__ import annotations
import json
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
        if self.size == 0:
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
        import faiss
        matrix = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, documents=documents)
    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> 'FaissVectorStore':
        import faiss
        index = faiss.read_index(str(index_path))
        raw = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
        documents = [Document(**item) for item in raw['documents']]
        return cls(index=index, documents=documents)
    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        import faiss
        index_path = Path(index_path); metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True); metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps({'documents': [{'id': d.id, 'text': d.text, 'metadata': d.metadata} for d in self.documents]}, ensure_ascii=False, indent=2), encoding='utf-8')
