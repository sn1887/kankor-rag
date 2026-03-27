from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from rag_core.contracts.vector_store import VectorStore
from rag_core.types import Document, Hit
from rag_core.util.query_normalization import (
    canonicalize_subject,
    extract_grade_hint,
    extract_structural_reference,
    extract_subject_hint,
    normalize_query_text,
)


@dataclass(frozen=True, slots=True)
class RetrievalAssessment:
    weak: bool
    reason: str
    top_score: float | None
    hit_count: int


_FRONT_MATTER_SOURCE_TYPES = frozenset(
    {
        "front_matter",
        "frontmatter",
        "cover",
        "title_page",
        "preface",
        "table_of_contents",
        "toc",
        "contents",
        "toc_manifest_frontmatter",
    }
)

_FRONT_MATTER_TEXT_MARKERS = frozenset(
    {
        "فهرست مطالب",
        "فهرست",
        "سرود ملی",
        "سال چاپ",
        "table of contents",
        "contents",
        "copyright",
        "isbn",
        "edition",
        "dedication",
        "publisher",
    }
)

_CHAPTER_HEADING_PATTERN = re.compile(
    r"(?:^|\s)(?:chapter|chap|lesson|unit|فصل|باب|بخش|درس)\s*[0-9۰-۹]{1,2}",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(slots=True)
class TopicLocatorFrontMatterPolicy:
    enabled: bool = True
    max_front_matter_page: int = 6
    suppress_when_page_unknown: bool = False
    keep_toc_manifest_hits: bool = True
    allow_front_matter_when_empty: bool = False
    source_type_markers: frozenset[str] = field(default_factory=lambda: _FRONT_MATTER_SOURCE_TYPES)
    text_markers: frozenset[str] = field(default_factory=lambda: _FRONT_MATTER_TEXT_MARKERS)

    def _contains_front_matter_marker(self, text: str) -> bool:
        normalized = _normalize_text(text)
        if not normalized:
            return False
        return any(marker in normalized for marker in self.text_markers)

    def _has_chapter_signal(self, hit: Hit) -> bool:
        metadata = hit.document.metadata
        if str(metadata.get("chapter_number", "")).strip():
            return True
        if str(metadata.get("frontmatter_chapter_title", "")).strip():
            return True
        chapter_title = str(metadata.get("chapter_title", "")).strip()
        if chapter_title and not self._contains_front_matter_marker(chapter_title):
            return True
        content = " ".join(
            [
                hit.document.text,
                chapter_title,
                str(metadata.get("line_text", "")),
            ]
        ).strip()
        return bool(_CHAPTER_HEADING_PATTERN.search(content))

    def is_front_matter(self, hit: Hit) -> bool:
        if not self.enabled:
            return False
        metadata = hit.document.metadata

        source_type = str(metadata.get("source_type", "")).strip().lower()
        if self.keep_toc_manifest_hits and source_type == "toc_manifest":
            return False
        if source_type and source_type in self.source_type_markers and not self._has_chapter_signal(hit):
            return True

        page = _coerce_int(metadata.get("page"))
        start_page = _coerce_int(metadata.get("start_page"))
        effective_page = page if page is not None else start_page
        if effective_page is None and self.suppress_when_page_unknown:
            return True

        text_fields: list[str] = [hit.document.text]
        chapter_title = str(metadata.get("chapter_title", "")).strip()
        if chapter_title:
            text_fields.append(chapter_title)
        frontmatter_chapter_title = str(metadata.get("frontmatter_chapter_title", "")).strip()
        if frontmatter_chapter_title:
            text_fields.append(frontmatter_chapter_title)
        combined_text = " ".join(part for part in text_fields if part).strip()
        contains_front_marker = self._contains_front_matter_marker(combined_text)
        has_chapter_signal = self._has_chapter_signal(hit)

        if effective_page is not None and effective_page <= max(1, int(self.max_front_matter_page)):
            if contains_front_marker:
                return True
            if not has_chapter_signal:
                return True
        if contains_front_marker and not has_chapter_signal:
            return True
        return False


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


def _normalize_text(text: str) -> str:
    return normalize_query_text(text)


def _extract_chapter_number(question: str) -> str | None:
    reference = extract_structural_reference(question)
    if reference is None:
        return None
    return reference.number


def _extract_subject_hint(question: str) -> str | None:
    return extract_subject_hint(question)


def _extract_grade_hint(question: str) -> str | None:
    return extract_grade_hint(question)


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

        subject = canonicalize_subject(str(metadata.get("subject", "")).strip())
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
    front_matter_policy: TopicLocatorFrontMatterPolicy | None = None,
) -> list[Hit]:
    resolved_top_k = max(1, int(top_k))
    policy = front_matter_policy or TopicLocatorFrontMatterPolicy()
    subject_hint = _extract_subject_hint(question)
    grade_hint = _extract_grade_hint(question)
    filtered: list[Hit] = []
    for hit in hits:
        metadata = hit.document.metadata
        subject = canonicalize_subject(str(metadata.get("subject", "")).strip())
        grade = str(metadata.get("grade_band", "")).strip()
        if subject_hint and subject and subject != subject_hint:
            continue
        if grade_hint and grade and grade != grade_hint:
            continue
        filtered.append(hit)
    candidates = filtered if filtered else list(hits)
    if not candidates:
        return []

    ranked_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    if not policy.enabled:
        return ranked_candidates[:resolved_top_k]

    non_front_matter_hits = [hit for hit in ranked_candidates if not policy.is_front_matter(hit)]
    if non_front_matter_hits:
        return non_front_matter_hits[:resolved_top_k]
    if policy.allow_front_matter_when_empty:
        return ranked_candidates[:resolved_top_k]
    return []


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
