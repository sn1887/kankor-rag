from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rag_core.contracts.structure_lookup import StructureLookup
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource
from rag_core.types import Document, Hit
from rag_core.util.query_normalization import (
    extract_grade_hint,
    extract_structural_reference,
    extract_subject_hint,
    normalize_query_text,
)


_LOCATOR_STOPWORDS = frozenset(
    {
        "where",
        "which",
        "chapter",
        "page",
        "lesson",
        "unit",
        "topic",
        "book",
        "textbook",
        "grade",
        "is",
        "the",
        "in",
        "of",
        "kankor",
        "در",
        "کدام",
        "کجا",
        "فصل",
        "صفحه",
        "بخش",
        "درس",
        "موضوع",
        "کتاب",
        "صنف",
        "را",
        "است",
        "می",
        "شود",
        "تدریس",
        "شده",
        "آمده",
        "په",
        "کوم",
        "کې",
        "صفحه",
        "فصل",
        "درس",
    }
)


def _normalized_tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_query_text(text)
    if not normalized:
        return ()
    return tuple(
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in _LOCATOR_STOPWORDS
    )


class JsonlStructureLookup(StructureLookup):
    def __init__(self, *, documents: Iterable[Document]) -> None:
        self._documents = list(documents)

    @classmethod
    def load(
        cls,
        *,
        chapter_index_path: str | Path | None = None,
        topic_index_path: str | Path | None = None,
    ) -> "JsonlStructureLookup":
        documents: list[Document] = []
        for path in (chapter_index_path, topic_index_path):
            if path is None:
                continue
            candidate = Path(path)
            if not candidate.exists():
                continue
            source = HFDatasetCorpusSource(local_path=str(candidate))
            documents.extend(source.load_documents())
        return cls(documents=documents)

    def search(self, *, question: str, top_k: int = 5) -> list[Hit]:
        normalized_query = normalize_query_text(question)
        if not normalized_query:
            return []

        query_tokens = set(_normalized_tokens(question))
        subject_hint = extract_subject_hint(question)
        grade_hint = extract_grade_hint(question)
        structural_reference = extract_structural_reference(question)

        hits: list[Hit] = []
        for document in self._documents:
            metadata = document.metadata
            lookup_text = str(metadata.get("normalized_lookup_text") or normalize_query_text(document.text)).strip()
            if not lookup_text:
                continue

            title_tokens = set(lookup_text.split())
            overlap = len(query_tokens & title_tokens)
            score = 0.0
            if overlap > 0:
                score += overlap / max(1, len(query_tokens))
            if lookup_text and lookup_text in normalized_query:
                score += 0.45
            elif normalized_query and normalized_query in lookup_text:
                score += 0.25

            if subject_hint:
                if subject_hint == str(metadata.get("subject", "")).strip():
                    score += 0.12
                else:
                    score -= 0.05
            if grade_hint:
                if grade_hint == str(metadata.get("grade_band", "")).strip():
                    score += 0.08
                else:
                    score -= 0.03

            if structural_reference is not None:
                entry_kind = str(metadata.get("lookup_kind", "")).strip()
                entry_number = str(metadata.get("chapter_number") or "").strip()
                if entry_kind == structural_reference.kind and entry_number == structural_reference.number:
                    score += 0.35

            if score <= 0.0:
                continue
            hits.append(Hit(document=document, score=min(score, 1.0)))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]
