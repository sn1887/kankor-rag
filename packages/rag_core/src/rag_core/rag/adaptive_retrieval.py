from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from rag_core.contracts.vector_store import VectorStore
from rag_core.types import Document, Hit


@dataclass(frozen=True, slots=True)
class RetrievalAssessment:
    weak: bool
    reason: str
    top_score: float | None
    hit_count: int


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def merge_retrieval_hits(
    *,
    hit_groups: Sequence[Sequence[Hit]],
    top_k: int,
    min_score: float,
) -> list[Hit]:
    score_by_id: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}
    for group in hit_groups:
        for hit in group:
            if hit.score < min_score:
                continue
            document_id = hit.document.id
            current = score_by_id.get(document_id)
            if current is None or hit.score > current:
                score_by_id[document_id] = float(hit.score)
                doc_by_id[document_id] = hit.document

    ranked = sorted(
        (
            Hit(document=doc_by_id[document_id], score=score)
            for document_id, score in score_by_id.items()
        ),
        key=lambda hit: hit.score,
        reverse=True,
    )
    return ranked[: max(1, top_k)]


def _iter_vector_store_documents(vector_store: VectorStore) -> Sequence[Document]:
    docs = getattr(vector_store, "documents", None)
    if isinstance(docs, Sequence):
        return docs
    return ()


_CHAPTER_NUMBER_TOKEN_MAP = {
    "1": "1",
    "اول": "1",
    "نخست": "1",
    "یکم": "1",
    "۱": "1",
    "2": "2",
    "دوم": "2",
    "دوهم": "2",
    "ثانی": "2",
    "۲": "2",
    "3": "3",
    "سوم": "3",
    "درېم": "3",
    "۳": "3",
    "4": "4",
    "چهارم": "4",
    "څلورم": "4",
    "۴": "4",
    "5": "5",
    "پنجم": "5",
    "۵": "5",
    "6": "6",
    "ششم": "6",
    "۶": "6",
    "7": "7",
    "هفتم": "7",
    "۷": "7",
    "8": "8",
    "هشتم": "8",
    "۸": "8",
    "9": "9",
    "نهم": "9",
    "۹": "9",
    "10": "10",
    "دهم": "10",
    "لسم": "10",
    "۱۰": "10",
    "11": "11",
    "یازدهم": "11",
    "۱۱": "11",
    "12": "12",
    "دوازدهم": "12",
    "۱۲": "12",
}

_SUBJECT_HINTS = {
    "physics": {"physics", "physic", "فزیک", "فیزیک", "فزیکي"},
    "mathematics": {"math", "mathematics", "ریاضی", "رياضی", "الجبر"},
    "chemistry": {"chemistry", "کیمیا", "شیمی"},
    "biology": {"biology", "بیولوژی", "حیات"},
    "geography": {"geography", "جغرافیه", "جغرافيا"},
    "history": {"history", "تاریخ"},
    "dari": {"dari", "دری", "فارسی"},
    "english": {"english", "انگلیسی", "انگليسي"},
}

_GRADE_HINT_TOKENS = {
    "10": {"10", "۱۰", "دهم", "صنف دهم", "grade 10"},
    "11": {"11", "۱۱", "یازدهم", "صنف یازدهم", "grade 11"},
    "12": {"12", "۱۲", "دوازدهم", "صنف دوازدهم", "grade 12"},
}


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def _extract_chapter_number(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None
    explicit = re.search(
        r"(?:chapter|chap|فصل|باب)\s+([0-9۰-۹]{1,2}|[a-z\u0600-\u06FF]+)",
        normalized,
        flags=re.IGNORECASE | re.UNICODE,
    )
    if explicit:
        token = explicit.group(1).strip().lower()
        mapped = _CHAPTER_NUMBER_TOKEN_MAP.get(token)
        if mapped:
            return mapped
    for token in normalized.split():
        mapped = _CHAPTER_NUMBER_TOKEN_MAP.get(token.strip().lower())
        if mapped is not None:
            return mapped
    return None


def _extract_subject_hint(question: str) -> str | None:
    normalized = _normalize_text(question)
    for subject, terms in _SUBJECT_HINTS.items():
        if any(term in normalized for term in terms):
            return subject
    return None


def _extract_grade_hint(question: str) -> str | None:
    normalized = _normalize_text(question)
    for grade, terms in _GRADE_HINT_TOKENS.items():
        if any(term in normalized for term in terms):
            return grade
    return None


def _chapter_query_variants(chapter_number: str) -> tuple[str, ...]:
    return (
        f"chapter {chapter_number}",
        f"chap {chapter_number}",
        f"فصل {chapter_number}",
        f"باب {chapter_number}",
    )


def find_topic_locator_chapter_hits(
    *,
    question: str,
    vector_store: VectorStore,
    top_k: int,
) -> list[Hit]:
    chapter_number = _extract_chapter_number(question)
    if chapter_number is None:
        return []

    subject_hint = _extract_subject_hint(question)
    grade_hint = _extract_grade_hint(question)
    variants = _chapter_query_variants(chapter_number)

    candidates: list[Hit] = []
    for document in _iter_vector_store_documents(vector_store):
        normalized_text = _normalize_text(document.text)
        if not normalized_text:
            continue

        match_count = 0
        for phrase in variants:
            if phrase in normalized_text:
                match_count += 1
        if match_count == 0:
            continue

        metadata = document.metadata
        score = 0.42 + (0.08 * float(match_count))

        subject = str(metadata.get("subject", "")).strip().lower()
        if subject_hint:
            if subject == subject_hint:
                score += 0.12
            else:
                score -= 0.08

        grade = str(metadata.get("grade_band", "")).strip()
        if grade_hint:
            if grade == grade_hint:
                score += 0.08
            else:
                score -= 0.04

        page = _coerce_int(metadata.get("page"))
        if page is not None and page <= 10:
            score += 0.03

        candidates.append(Hit(document=document, score=max(0.0, score)))

    ranked = sorted(candidates, key=lambda hit: hit.score, reverse=True)
    return ranked[: max(1, int(top_k))]


def filter_topic_locator_hits(
    *,
    hits: Sequence[Hit],
    question: str,
    top_k: int,
) -> list[Hit]:
    subject_hint = _extract_subject_hint(question)
    grade_hint = _extract_grade_hint(question)
    if not subject_hint and not grade_hint:
        return list(hits)[: max(1, int(top_k))]

    filtered: list[Hit] = []
    for hit in hits:
        metadata = hit.document.metadata
        subject = str(metadata.get("subject", "")).strip().lower()
        grade = str(metadata.get("grade_band", "")).strip()
        if subject_hint and subject and subject != subject_hint:
            continue
        if grade_hint and grade and grade != grade_hint:
            continue
        filtered.append(hit)

    if not filtered:
        return list(hits)[: max(1, int(top_k))]
    ranked = sorted(filtered, key=lambda item: item.score, reverse=True)
    return ranked[: max(1, int(top_k))]


def _neighbor_candidates(
    *,
    source_id: str,
    anchor_chunk_index: int,
    vector_store: VectorStore,
    neighbors_per_side: int,
) -> list[tuple[Document, int]]:
    if neighbors_per_side <= 0:
        return []
    if not source_id:
        return []

    chunk_lookup: dict[int, Document] = {}
    for document in _iter_vector_store_documents(vector_store):
        metadata: Mapping[str, object] = document.metadata
        if str(metadata.get("source_id", "")).strip() != source_id:
            continue
        chunk_index = _coerce_int(metadata.get("chunk_index"))
        if chunk_index is None:
            continue
        chunk_lookup[chunk_index] = document

    candidates: list[tuple[Document, int]] = []
    for distance in range(1, neighbors_per_side + 1):
        for offset in (-distance, distance):
            target_index = anchor_chunk_index + offset
            doc = chunk_lookup.get(target_index)
            if doc is not None:
                candidates.append((doc, distance))
    return candidates


def expand_local_window_hits(
    *,
    hits: Sequence[Hit],
    vector_store: VectorStore,
    neighbors_per_side: int,
    max_hits: int | None = None,
) -> list[Hit]:
    if not hits or neighbors_per_side <= 0:
        return list(hits)

    anchor = hits[0]
    metadata = anchor.document.metadata
    source_id = str(metadata.get("source_id", "")).strip()
    anchor_chunk_index = _coerce_int(metadata.get("chunk_index"))
    if not source_id or anchor_chunk_index is None:
        return list(hits)

    expanded_by_id: dict[str, Hit] = {hit.document.id: hit for hit in hits}
    for document, distance in _neighbor_candidates(
        source_id=source_id,
        anchor_chunk_index=anchor_chunk_index,
        vector_store=vector_store,
        neighbors_per_side=neighbors_per_side,
    ):
        decayed_score = max(0.0, float(anchor.score) - (0.005 * distance))
        existing = expanded_by_id.get(document.id)
        candidate = Hit(document=document, score=decayed_score)
        if existing is None or candidate.score > existing.score:
            expanded_by_id[document.id] = candidate

    ranked = sorted(expanded_by_id.values(), key=lambda hit: hit.score, reverse=True)
    if max_hits is None:
        return ranked
    return ranked[: max(1, int(max_hits))]


def assess_retrieval_confidence(
    *,
    hits: Sequence[Hit],
    min_top_score: float,
    min_hits: int,
) -> RetrievalAssessment:
    count = len(hits)
    if count == 0:
        return RetrievalAssessment(weak=True, reason="no_hits", top_score=None, hit_count=0)

    top_score = float(hits[0].score)
    if count < max(1, int(min_hits)):
        return RetrievalAssessment(
            weak=True,
            reason="insufficient_hits",
            top_score=top_score,
            hit_count=count,
        )

    if top_score < float(min_top_score):
        return RetrievalAssessment(
            weak=True,
            reason="low_top_score",
            top_score=top_score,
            hit_count=count,
        )

    return RetrievalAssessment(
        weak=False,
        reason="strong",
        top_score=top_score,
        hit_count=count,
    )
