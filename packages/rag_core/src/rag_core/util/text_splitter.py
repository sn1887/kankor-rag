from __future__ import annotations
from collections.abc import Iterable
from rag_core.types import Document
from rag_core.util.hashing import stable_hash

def split_text(text: str, chunk_size: int = 750, chunk_overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = ' '.join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks

def iter_chunked_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 140,
    chunk_overlap: int = 24,
) -> Iterable[Document]:
    for document in documents:
        parts = split_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(parts) == 1:
            yield document
            continue
        for index, part in enumerate(parts):
            metadata = dict(document.metadata)
            metadata['chunk_index'] = index
            metadata['chunk_total'] = len(parts)
            yield Document(
                id=stable_hash({'parent_id': document.id, 'chunk_index': index, 'text': part}),
                text=part,
                metadata={**metadata, 'parent_id': document.id},
            )


def chunk_documents(documents: Iterable[Document], *, chunk_size: int = 140, chunk_overlap: int = 24) -> list[Document]:
    return list(
        iter_chunked_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
