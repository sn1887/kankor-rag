from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.adaptive_retrieval import merge_retrieval_hits
from rag_core.types import Document, Hit
from rag_core.util.query_normalization import normalize_query_text

_MEANING_QUERY_PHRASES = tuple(
    normalize_query_text(value)
    for value in (
        "meaning",
        "meanings",
        "definition",
        "define",
        "word meaning",
        "what does",
        "what is the meaning",
        "vocabulary",
        "معنی",
        "معنا",
        "تعریف",
        "مفهوم",
        "لغت",
        "واژه",
        "کلمه",
        "مانا",
        "اصطلاح",
    )
)


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _iter_documents(vector_store: VectorStore) -> Sequence[Document]:
    documents = getattr(vector_store, "documents", None)
    if isinstance(documents, Sequence):
        return documents
    return ()


def _hit_overlaps_page_span(hit: Hit, *, span_start: int, span_end: int) -> bool:
    metadata = hit.document.metadata
    start_page = _coerce_positive_int(metadata.get("start_page", metadata.get("page")))
    end_page = _coerce_positive_int(metadata.get("end_page", metadata.get("page")))
    if start_page is None or end_page is None:
        return False
    return end_page >= span_start and start_page <= span_end


def _is_meaning_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    return any(phrase and phrase in normalized_query for phrase in _MEANING_QUERY_PHRASES)


@dataclass(slots=True)
class RetrievalDiagnostics:
    query_count: int = 0
    dense_ms: int = 0
    lexical_ms: int = 0
    fusion_ms: int = 0
    total_ms: int = 0
    dense_search_calls: int = 0


@dataclass(frozen=True, slots=True)
class _CachedLexicalDocument:
    document: Document
    normalized_text: str
    token_set: frozenset[str]
    start_page: int | None
    end_page: int | None
    is_glossary: bool


class DenseRetriever:
    def __init__(self, *, embedder: Embedder, vector_store: VectorStore, min_score: float) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._min_score = float(min_score)

    def _filter_candidates(
        self,
        *,
        candidates: Sequence[Hit],
        page_span: tuple[int, int] | None,
    ) -> list[Hit]:
        filtered: list[Hit] = []
        for hit in candidates:
            if float(hit.score) < self._min_score:
                continue
            if page_span is not None and not _hit_overlaps_page_span(
                hit,
                span_start=page_span[0],
                span_end=page_span[1],
            ):
                continue
            filtered.append(hit)
        return filtered

    def search_many(
        self,
        *,
        questions: Sequence[str],
        top_k: int,
        filters: Mapping[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
    ) -> tuple[list[Hit], int]:
        cleaned_questions: list[str] = []
        seen: set[str] = set()
        for question in questions:
            cleaned = " ".join(str(question or "").split()).strip()
            if not cleaned:
                continue
            normalized = cleaned.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned_questions.append(cleaned)
        if not cleaned_questions:
            return [], 0

        query_vectors = np.asarray(self._embedder.embed_queries(cleaned_questions), dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        hit_groups: list[list[Hit]] = []
        search_calls = 0
        for query_vector in query_vectors:
            search_calls += 1
            candidates = self._vector_store.search(query_vector, top_k=max(1, int(top_k)), filters=filters)
            hit_groups.append(self._filter_candidates(candidates=candidates, page_span=page_span))

        return merge_retrieval_hits(hit_groups=hit_groups, top_k=top_k, min_score=self._min_score), search_calls

    def search_per_query(
        self,
        *,
        questions: Sequence[str],
        top_k: int,
        filters: Mapping[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
        batch_size: int = 64,
    ) -> tuple[list[list[Hit]], int]:
        if not questions:
            return [], 0
        normalized_batch_size = max(1, int(batch_size))
        results: list[list[Hit]] = [[] for _ in questions]
        search_calls = 0

        for start in range(0, len(questions), normalized_batch_size):
            batch_questions = list(questions[start : start + normalized_batch_size])
            indexed_texts: list[tuple[int, str]] = []
            for offset, question in enumerate(batch_questions):
                cleaned = " ".join(str(question or "").split()).strip()
                if not cleaned:
                    continue
                indexed_texts.append((offset, cleaned))
            if not indexed_texts:
                continue

            query_vectors = np.asarray(
                self._embedder.embed_queries([text for _, text in indexed_texts]),
                dtype=np.float32,
            )
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)

            for (offset, _), query_vector in zip(indexed_texts, query_vectors, strict=False):
                search_calls += 1
                candidates = self._vector_store.search(query_vector, top_k=max(1, int(top_k)), filters=filters)
                results[start + offset] = self._filter_candidates(
                    candidates=candidates,
                    page_span=page_span,
                )

        return results, search_calls


class LexicalRetriever:
    def __init__(self, *, vector_store: VectorStore, min_score: float = 0.01) -> None:
        self._vector_store = vector_store
        self._min_score = float(min_score)
        self._cached_documents = self._build_document_cache(_iter_documents(vector_store))

    @staticmethod
    def _build_document_cache(documents: Sequence[Document]) -> tuple[_CachedLexicalDocument, ...]:
        cached_documents: list[_CachedLexicalDocument] = []
        for document in documents:
            normalized_text = normalize_query_text(document.text)
            token_set = frozenset(normalized_text.split()) if normalized_text else frozenset()
            metadata = document.metadata
            cached_documents.append(
                _CachedLexicalDocument(
                    document=document,
                    normalized_text=normalized_text,
                    token_set=token_set,
                    start_page=_coerce_positive_int(metadata.get("start_page", metadata.get("page"))),
                    end_page=_coerce_positive_int(metadata.get("end_page", metadata.get("page"))),
                    is_glossary=bool(metadata.get("is_glossary")),
                )
            )
        return tuple(cached_documents)

    @staticmethod
    def _cached_document_overlaps_page_span(
        cached_document: _CachedLexicalDocument,
        *,
        span_start: int,
        span_end: int,
    ) -> bool:
        start_page = cached_document.start_page
        end_page = cached_document.end_page
        if start_page is None or end_page is None:
            return False
        return end_page >= span_start and start_page <= span_end

    def search_many(
        self,
        *,
        questions: Sequence[str],
        top_k: int,
        filters: Mapping[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
    ) -> list[Hit]:
        if not self._cached_documents:
            return []

        score_by_id: dict[str, float] = {}
        doc_by_id: dict[str, Document] = {}
        for question in questions:
            normalized_query = normalize_query_text(question)
            if not normalized_query:
                continue
            meaning_query = _is_meaning_query(normalized_query)
            query_tokens = {token for token in normalized_query.split() if len(token) >= 2}
            if not query_tokens:
                continue
            joined_query = " ".join(query_tokens)
            for cached_document in self._cached_documents:
                document = cached_document.document
                metadata = document.metadata
                if filters and any(str(metadata.get(key)) != str(value) for key, value in filters.items()):
                    continue
                if page_span is not None and not self._cached_document_overlaps_page_span(
                    cached_document,
                    span_start=page_span[0],
                    span_end=page_span[1],
                ):
                    continue
                normalized_text = cached_document.normalized_text
                if not normalized_text:
                    continue
                overlap = query_tokens & cached_document.token_set
                if not overlap:
                    continue
                overlap_ratio = len(overlap) / max(1, len(query_tokens))
                contiguous = 1.0 if joined_query and joined_query in normalized_text else 0.0
                score = (0.75 * overlap_ratio) + (0.25 * contiguous)
                if meaning_query and cached_document.is_glossary:
                    score += 0.08
                if score < self._min_score:
                    continue
                existing = score_by_id.get(document.id)
                if existing is None or score > existing:
                    score_by_id[document.id] = float(score)
                    doc_by_id[document.id] = document

        ranked = sorted(
            (Hit(document=doc_by_id[doc_id], score=score) for doc_id, score in score_by_id.items()),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[: max(1, int(top_k))]


class RRFFusionPolicy:
    def __init__(self, *, rrf_k: int = 20) -> None:
        self._rrf_k = max(1, int(rrf_k))

    def fuse(self, *, dense_hits: Sequence[Hit], lexical_hits: Sequence[Hit], top_k: int) -> list[Hit]:
        scored_rrf: dict[str, float] = {}
        best_hits: dict[str, Hit] = {}

        def _accumulate(hits: Sequence[Hit]) -> None:
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit.document.id
                scored_rrf[doc_id] = scored_rrf.get(doc_id, 0.0) + (1.0 / (self._rrf_k + rank))
                existing = best_hits.get(doc_id)
                if existing is None or float(hit.score) > float(existing.score):
                    best_hits[doc_id] = hit

        _accumulate(dense_hits)
        _accumulate(lexical_hits)

        fused = list(best_hits.values())
        fused.sort(
            key=lambda item: (scored_rrf.get(item.document.id, 0.0), float(item.score)),
            reverse=True,
        )
        return fused[: max(1, int(top_k))]


class PageRetrievalEngine:
    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        fusion_policy: RRFFusionPolicy,
    ) -> None:
        self._dense_retriever = dense_retriever
        self._lexical_retriever = lexical_retriever
        self._fusion_policy = fusion_policy

    def search(
        self,
        *,
        questions: Sequence[str],
        top_k: int,
        filters: Mapping[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
    ) -> tuple[list[Hit], RetrievalDiagnostics]:
        started = time.perf_counter()
        diagnostics = RetrievalDiagnostics(query_count=len([question for question in questions if str(question).strip()]))

        dense_started = time.perf_counter()
        dense_hits, dense_search_calls = self._dense_retriever.search_many(
            questions=questions,
            top_k=top_k,
            filters=filters,
            page_span=page_span,
        )
        diagnostics.dense_ms = int((time.perf_counter() - dense_started) * 1000)
        diagnostics.dense_search_calls = dense_search_calls

        lexical_started = time.perf_counter()
        lexical_hits = self._lexical_retriever.search_many(
            questions=questions,
            top_k=top_k,
            filters=filters,
            page_span=page_span,
        )
        diagnostics.lexical_ms = int((time.perf_counter() - lexical_started) * 1000)

        fusion_started = time.perf_counter()
        fused_hits = self._fusion_policy.fuse(
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            top_k=top_k,
        )
        diagnostics.fusion_ms = int((time.perf_counter() - fusion_started) * 1000)
        diagnostics.total_ms = int((time.perf_counter() - started) * 1000)
        return fused_hits, diagnostics
