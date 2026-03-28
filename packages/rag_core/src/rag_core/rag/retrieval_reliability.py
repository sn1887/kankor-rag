from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
import logging
import re
from typing import Any, Literal, Mapping

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.toc_locator import TOCIndex
from rag_core.types import Document, Hit
from rag_core.util.query_normalization import (
    canonicalize_subject,
    extract_grade_hint,
    extract_subject_hint,
    normalize_query_text,
    subject_hint_from_text,
)

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_RRF_K = 20
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RETRIEVAL_LEG_TIMEOUT_MS = 200
DEFAULT_RETRIEVAL_DEADLINE_MS = 600
DEFAULT_RERANK_TOP_N = 8
DEFAULT_RERANK_TIMEOUT_MS = 400
DEFAULT_PAGE_LOCALIZATION_MIN_CONFIDENCE = 0.4
DEFAULT_AMBIGUITY_TOP_N = 5
DEFAULT_AMBIGUITY_MIN_SUBJECTS = 3
DEFAULT_AMBIGUITY_DOMINANCE_THRESHOLD = 0.50

_NON_TEXT_PATTERN = re.compile(r"[^\w\u0600-\u06FF\s]", flags=re.UNICODE)
_PERSON_PHRASES = (
    "چه کسی",
    "کدام شخص",
)
_PERSON_TOKENS = frozenset({"کی", "who"})
_PERSON_CONTEXT_TOKENS = frozenset({"توسط"})

_DATE_TOKENS = frozenset(
    {
        "سال",
        "میلادی",
        "هجری",
        "year",
        "date",
    }
)

_LOCATION_PRIMARY_TOKENS = frozenset({"موقعیت", "موقعيت"})
_LOCATION_TYPE_TOKENS = frozenset({"کشور", "ولایت", "ولايت", "شهر", "ولسوالی", "قریه", "قريه", "منطقه"})
_MCQ_OPTION_PATTERN = re.compile(
    r"^\s*(?:[A-Da-d]|[ابجد]|الف|ب|ج|د|گزینه\s*[A-Da-dابجد])\s*[)\].:：\-]\s*(.+?)\s*$",
    flags=re.UNICODE,
)
_NUMERIC_DATE_PATTERN = re.compile(r"\b(?:[0-9]{3,4})\b")

def normalize_retrieval_text(text: str) -> str:
    normalized = normalize_query_text(text)
    return _NON_TEXT_PATTERN.sub(" ", normalized).strip()


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _subject_hint_from_text(normalized_text: str) -> str | None:
    return subject_hint_from_text(normalized_text)


def _grade_hint_from_text(normalized_text: str) -> str | None:
    return extract_grade_hint(normalized_text)


def _extract_mcq_options(raw_query: str) -> tuple[str, ...]:
    options: list[str] = []
    for line in str(raw_query or "").splitlines():
        match = _MCQ_OPTION_PATTERN.match(line)
        if not match:
            continue
        normalized = normalize_retrieval_text(match.group(1))
        if not normalized:
            continue
        tokens = [token for token in normalized.split() if len(token) >= 3]
        if not tokens:
            continue
        options.append(" ".join(tokens))
    return tuple(options)

def _looks_like_person_question(*, normalized_query: str, tokens: tuple[str, ...]) -> bool:
    if any(phrase in normalized_query for phrase in _PERSON_PHRASES):
        return True
    if _PERSON_CONTEXT_TOKENS.intersection(tokens) and _PERSON_TOKENS.intersection(tokens):
        return True

    # "کی" is ambiguous (who/when) in Dari; only treat it as a person intent when
    # there are no strong date signals.
    if "کی" in tokens and not _looks_like_date_question(normalized_query=normalized_query, tokens=tokens):
        return True
    return False


def _looks_like_date_question(*, normalized_query: str, tokens: tuple[str, ...]) -> bool:
    if _DATE_TOKENS.intersection(tokens):
        return True
    return bool(_NUMERIC_DATE_PATTERN.search(normalized_query))


def _looks_like_place_question(*, normalized_query: str, tokens: tuple[str, ...]) -> bool:
    # Use token boundary checks to avoid false positives from generic phrasing like "در کدام ...".
    if any(token.startswith("کجا") for token in tokens):
        return True
    if _LOCATION_PRIMARY_TOKENS.intersection(tokens):
        return True
    if "کدام" in tokens and _LOCATION_TYPE_TOKENS.intersection(tokens):
        return True
    if "where" in tokens:
        return True
    return False


def _detect_intents(*, raw_query: str, normalized_query: str, mcq_options: Sequence[str]) -> frozenset[str]:
    intents: set[str] = set()
    tokens = tuple(normalized_query.split())
    if _looks_like_person_question(normalized_query=normalized_query, tokens=tokens):
        intents.add("person")
    if _looks_like_date_question(normalized_query=normalized_query, tokens=tokens):
        intents.add("date")
    if _looks_like_place_question(normalized_query=normalized_query, tokens=tokens):
        intents.add("place")
    if mcq_options:
        intents.add("mcq")
    if raw_query.strip():
        intents.add("query")
    return frozenset(intents)


@dataclass(frozen=True)
class TOCCandidate:
    chapter_id: str
    title: str
    score: float
    page_start: int | None


@dataclass(frozen=True)
class QueryContext:
    raw_query: str
    normalized_query: str
    canonical_tokens: tuple[str, ...]
    detected_intents: frozenset[str]
    subject_hint: str | None
    grade_hint: str | None
    mcq_options: tuple[str, ...]
    toc_candidates: tuple[TOCCandidate, ...]


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    chunk_id: str
    source_id: str
    subject: str | None
    page_start: int | None
    page_end: int | None
    score: float
    leg: Literal["dense", "lexical"]
    matched_terms: tuple[str, ...]
    metadata: Mapping[str, Any]
    rrf_score: float = 0.0


@dataclass(frozen=True)
class RetrievalDebugTrace:
    dense_timeout: bool
    lexical_timeout: bool
    reranker_timeout: bool
    reranker_degraded: bool
    reason_flags: tuple[str, ...]
    toc_trace: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LocalizationDecision:
    page_start: int | None
    page_end: int | None
    chapter_id: str | None
    confidence: float
    abstained: bool
    clarification_prompt: str | None
    top_candidates: tuple[Candidate, ...]
    reranker_degraded: bool
    trace: RetrievalDebugTrace | None


class RetrieverPlugin(ABC):
    @abstractmethod
    async def search(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> list[Candidate]:
        raise NotImplementedError


class FusionPolicy(ABC):
    @abstractmethod
    def fuse(
        self,
        dense: list[Candidate],
        lexical: list[Candidate],
        *,
        top_k: int,
    ) -> list[Candidate]:
        raise NotImplementedError


class RerankerPlugin(ABC):
    @abstractmethod
    async def rerank(
        self,
        ctx: QueryContext,
        candidates: list[Candidate],
        *,
        top_n: int,
        timeout_ms: int,
    ) -> list[Candidate]:
        raise NotImplementedError


class LocalizationDecisionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        ctx: QueryContext,
        candidates: list[Candidate],
    ) -> LocalizationDecision:
        raise NotImplementedError


class QueryContextBuilder(ABC):
    @abstractmethod
    def build(self, raw_query: str) -> QueryContext:
        raise NotImplementedError


class DefaultQueryContextBuilder(QueryContextBuilder):
    def __init__(self, *, toc_index: TOCIndex | None = None, toc_top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> None:
        self._toc_index = toc_index
        self._toc_top_k = max(1, int(toc_top_k))

    def _build_toc_candidates(self, raw_query: str) -> tuple[TOCCandidate, ...]:
        if self._toc_index is None:
            return ()
        if hasattr(self._toc_index, "search_with_trace"):
            hits, _ = self._toc_index.search_with_trace(question=raw_query, top_k=self._toc_top_k)
        else:
            hits = self._toc_index.search(question=raw_query, top_k=self._toc_top_k)
        candidates: list[TOCCandidate] = []
        for hit in hits:
            metadata = hit.document.metadata
            chapter_id = (
                str(metadata.get("chapter_id", "")).strip()
                or str(metadata.get("chapter_number", "")).strip()
                or str(hit.document.id).strip()
            )
            candidates.append(
                TOCCandidate(
                    chapter_id=chapter_id,
                    title=str(hit.document.text or metadata.get("title", "")).strip(),
                    score=float(hit.score),
                    page_start=_coerce_optional_int(metadata.get("start_page", metadata.get("page"))),
                )
            )
        return tuple(candidates)

    def build(self, raw_query: str) -> QueryContext:
        raw = str(raw_query or "").strip()
        normalized = normalize_retrieval_text(raw)
        canonical_tokens = tuple(normalized.split())
        mcq_options = _extract_mcq_options(raw)
        detected_intents = _detect_intents(
            raw_query=raw,
            normalized_query=normalized,
            mcq_options=mcq_options,
        )
        return QueryContext(
            raw_query=raw,
            normalized_query=normalized,
            canonical_tokens=canonical_tokens,
            detected_intents=detected_intents,
            subject_hint=extract_subject_hint(normalized),
            grade_hint=extract_grade_hint(normalized),
            mcq_options=mcq_options,
            toc_candidates=self._build_toc_candidates(raw),
        )


def _document_to_candidate(
    document: Document,
    *,
    score: float,
    leg: Literal["dense", "lexical"],
    matched_terms: tuple[str, ...] = (),
) -> Candidate:
    metadata = dict(document.metadata or {})
    source_id = str(metadata.get("source_id", "")).strip() or document.id
    chunk_id = str(metadata.get("chunk_id", "")).strip() or document.id
    subject = canonicalize_subject(str(metadata.get("subject", "")).strip()) or None
    page_start = _coerce_optional_int(metadata.get("start_page", metadata.get("page")))
    page_end = _coerce_optional_int(metadata.get("end_page", metadata.get("page")))
    if page_end is None:
        page_end = page_start
    safe_metadata: dict[str, Any] = dict(metadata)
    safe_metadata.setdefault("document_text", document.text)
    return Candidate(
        doc_id=document.id,
        chunk_id=chunk_id,
        source_id=source_id,
        subject=subject,
        page_start=page_start,
        page_end=page_end,
        score=float(score),
        leg=leg,
        matched_terms=matched_terms,
        metadata=safe_metadata,
    )


class DenseVectorRetrieverPlugin(RetrieverPlugin):
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._min_score = float(min_score)

    async def search(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> list[Candidate]:
        if top_k <= 0 or deadline_ms <= 0:
            return []

        query = ctx.normalized_query or ctx.raw_query
        if not query:
            return []

        vector = self._embedder.embed_query(query)
        hits = self._vector_store.search(vector, top_k=max(1, int(top_k)))
        candidates: list[Candidate] = []
        for hit in hits:
            if float(hit.score) < self._min_score:
                continue
            candidates.append(_document_to_candidate(hit.document, score=float(hit.score), leg="dense"))
        return candidates


class LexicalRetrieverPlugin(RetrieverPlugin):
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        min_score: float = 0.01,
    ) -> None:
        self._vector_store = vector_store
        self._min_score = float(min_score)

    def _iter_documents(self) -> Sequence[Document]:
        documents = getattr(self._vector_store, "documents", None)
        if isinstance(documents, Sequence):
            return documents
        return ()

    async def search(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> list[Candidate]:
        if top_k <= 0 or deadline_ms <= 0:
            return []
        if not ctx.canonical_tokens:
            return []

        query_tokens = tuple(token for token in ctx.canonical_tokens if len(token) >= 2)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        joined_query = " ".join(query_tokens)

        scored: list[Candidate] = []
        for document in self._iter_documents():
            normalized_text = normalize_retrieval_text(document.text)
            if not normalized_text:
                continue
            token_set = set(normalized_text.split())
            overlap = query_token_set & token_set
            if not overlap:
                continue
            overlap_ratio = len(overlap) / max(1, len(query_token_set))
            contiguous = 1.0 if joined_query and joined_query in normalized_text else 0.0
            score = (0.75 * overlap_ratio) + (0.25 * contiguous)
            if score < self._min_score:
                continue
            matched_terms = tuple(sorted(overlap))
            scored.append(
                _document_to_candidate(
                    document,
                    score=float(score),
                    leg="lexical",
                    matched_terms=matched_terms,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, int(top_k))]


class NoopRetrieverPlugin(RetrieverPlugin):
    async def search(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> list[Candidate]:
        _ = ctx, top_k, deadline_ms
        return []


class TOCLexicalRetrieverPlugin(RetrieverPlugin):
    def __init__(
        self,
        *,
        toc_index: TOCIndex,
        min_score: float = 0.5,
        guard_entity_questions: bool = True,
    ) -> None:
        self._toc_index = toc_index
        self._min_score = float(min_score)
        self._guard_entity_questions = bool(guard_entity_questions)

    def _allow_for_question(self, ctx: QueryContext, *, hit_score: float, match_kind: str) -> bool:
        if not self._guard_entity_questions:
            return True
        if not ({"person", "date", "place"} & set(ctx.detected_intents)):
            return True
        # Avoid TOC-driven routing for entity/date/place queries unless the TOC match is very strong
        # (chapter-number match, or a near-certain title match).
        if match_kind in {"chapter_number", "structural_number"}:
            return True
        return hit_score >= 0.92

    async def search_with_trace(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> tuple[list[Candidate], Mapping[str, Any] | None]:
        _ = deadline_ms
        if top_k <= 0:
            return [], None
        if hasattr(self._toc_index, "search_with_trace"):
            hits, toc_trace = self._toc_index.search_with_trace(question=ctx.raw_query, top_k=max(1, int(top_k)))
        else:
            hits = self._toc_index.search(question=ctx.raw_query, top_k=max(1, int(top_k)))
            toc_trace = None
        candidates: list[Candidate] = []
        for hit in hits:
            score = float(hit.score)
            if score < self._min_score:
                continue
            match_kind = str(hit.document.metadata.get("toc_match_kind", "")).strip().lower()
            if not self._allow_for_question(ctx, hit_score=score, match_kind=match_kind):
                continue
            candidates.append(_document_to_candidate(hit.document, score=score, leg="lexical"))
        return candidates, toc_trace

    async def search(
        self,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
    ) -> list[Candidate]:
        candidates, _ = await self.search_with_trace(ctx, top_k=top_k, deadline_ms=deadline_ms)
        return candidates


class RRFFusionPolicy(FusionPolicy):
    def __init__(self, *, rrf_k: int = DEFAULT_RETRIEVAL_RRF_K) -> None:
        self._rrf_k = max(1, int(rrf_k))

    def fuse(
        self,
        dense: list[Candidate],
        lexical: list[Candidate],
        *,
        top_k: int,
    ) -> list[Candidate]:
        scored_rrf: dict[tuple[str, str], float] = {}
        canonical: dict[tuple[str, str], Candidate] = {}

        def _accumulate(candidates: Sequence[Candidate]) -> None:
            for rank, candidate in enumerate(candidates, start=1):
                key = (candidate.doc_id, candidate.chunk_id)
                scored_rrf[key] = scored_rrf.get(key, 0.0) + (1.0 / (self._rrf_k + rank))
                current = canonical.get(key)
                if current is None or candidate.score > current.score:
                    canonical[key] = candidate

        _accumulate(dense)
        _accumulate(lexical)

        fused: list[Candidate] = [
            replace(candidate, rrf_score=float(scored_rrf[key]))
            for key, candidate in canonical.items()
        ]
        # Primary: fused rank (rrf_score). Secondary: original retriever score as a stable tie-breaker.
        fused.sort(key=lambda item: (item.rrf_score, item.score), reverse=True)
        return fused[: max(1, int(top_k))]


class StableIdentityRerankerPlugin(RerankerPlugin):
    def __init__(self, *, warmup_on_start: bool = True, artificial_delay_ms: int = 0) -> None:
        self._warmed = False
        self._artificial_delay_ms = max(0, int(artificial_delay_ms))
        if warmup_on_start:
            self._warmed = True

    async def _warmup(self) -> None:
        if self._warmed:
            return
        await asyncio.sleep(0)
        self._warmed = True

    async def rerank(
        self,
        ctx: QueryContext,
        candidates: list[Candidate],
        *,
        top_n: int,
        timeout_ms: int,
    ) -> list[Candidate]:
        _ = ctx, timeout_ms
        await self._warmup()
        if self._artificial_delay_ms > 0:
            await asyncio.sleep(self._artificial_delay_ms / 1000.0)
        if not candidates:
            return []
        return list(candidates[: max(1, int(top_n))])


class DefaultLocalizationDecisionPolicy(LocalizationDecisionPolicy):
    def __init__(
        self,
        *,
        page_localization_min_confidence: float = DEFAULT_PAGE_LOCALIZATION_MIN_CONFIDENCE,
        ambiguity_top_n: int = DEFAULT_AMBIGUITY_TOP_N,
        ambiguity_min_subjects: int = DEFAULT_AMBIGUITY_MIN_SUBJECTS,
        ambiguity_dominance_threshold: float = DEFAULT_AMBIGUITY_DOMINANCE_THRESHOLD,
        clarification_prompt: str = (
            "برای تعیین صفحه دقیق، لطفاً مضمون یا صنف سوال را واضح‌تر مشخص کنید."
        ),
    ) -> None:
        self._min_confidence = float(page_localization_min_confidence)
        self._ambiguity_top_n = max(1, int(ambiguity_top_n))
        self._ambiguity_min_subjects = max(2, int(ambiguity_min_subjects))
        self._ambiguity_dominance_threshold = float(ambiguity_dominance_threshold)
        self._clarification_prompt = clarification_prompt

    def _resolve_confidence(self, candidate: Candidate | None) -> float:
        if candidate is None:
            return 0.0
        baseline = max(float(candidate.score), float(candidate.rrf_score))
        return max(0.0, min(1.0, baseline))

    def _has_multi_subject_ambiguity(self, candidates: Sequence[Candidate]) -> bool:
        top_candidates = list(candidates[: self._ambiguity_top_n])
        subjects = [candidate.subject for candidate in top_candidates if candidate.subject]
        if len(set(subjects)) < self._ambiguity_min_subjects:
            return False
        counts = Counter(subjects)
        dominant_ratio = max(counts.values()) / max(1, len(subjects))
        return dominant_ratio <= self._ambiguity_dominance_threshold

    def decide(
        self,
        ctx: QueryContext,
        candidates: list[Candidate],
    ) -> LocalizationDecision:
        _ = ctx
        ranked = list(candidates)
        ranked.sort(key=lambda item: (item.rrf_score, item.score), reverse=True)
        top = ranked[0] if ranked else None
        confidence = self._resolve_confidence(top)
        chapter_id = None
        page_start = None
        page_end = None
        if top is not None:
            chapter_id = (
                str(top.metadata.get("chapter_id", "")).strip()
                or str(top.metadata.get("chapter_number", "")).strip()
                or None
            )
            page_start = top.page_start
            page_end = top.page_end

        abstained = False
        reason_prompt: str | None = None
        if top is None:
            abstained = True
            reason_prompt = self._clarification_prompt
        elif confidence < self._min_confidence:
            abstained = True
            reason_prompt = self._clarification_prompt
        elif self._has_multi_subject_ambiguity(ranked):
            abstained = True
            reason_prompt = self._clarification_prompt

        return LocalizationDecision(
            page_start=page_start,
            page_end=page_end,
            chapter_id=chapter_id,
            confidence=confidence,
            abstained=abstained,
            clarification_prompt=reason_prompt,
            top_candidates=tuple(ranked[: self._ambiguity_top_n]),
            reranker_degraded=False,
            trace=None,
        )


class RetrievalPipeline:
    def __init__(
        self,
        dense: RetrieverPlugin,
        lexical: RetrieverPlugin,
        fusion: FusionPolicy,
        reranker: RerankerPlugin,
        decision: LocalizationDecisionPolicy,
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        retrieval_deadline_ms: int = DEFAULT_RETRIEVAL_DEADLINE_MS,
        retrieval_leg_timeout_ms: int = DEFAULT_RETRIEVAL_LEG_TIMEOUT_MS,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        reranker_timeout_ms: int = DEFAULT_RERANK_TIMEOUT_MS,
        trace_enabled: bool = False,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._fusion = fusion
        self._reranker = reranker
        self._decision = decision
        self._top_k = max(1, int(top_k))
        self._retrieval_deadline_ms = max(1, int(retrieval_deadline_ms))
        self._retrieval_leg_timeout_ms = max(1, int(retrieval_leg_timeout_ms))
        self._rerank_top_n = max(1, int(rerank_top_n))
        self._reranker_timeout_ms = max(1, int(reranker_timeout_ms))
        self._trace_enabled = bool(trace_enabled)

    async def _run_leg(
        self,
        plugin: RetrieverPlugin,
        ctx: QueryContext,
        *,
        top_k: int,
        deadline_ms: int,
        leg_name: Literal["dense", "lexical"],
    ) -> tuple[list[Candidate], bool, str | None, Mapping[str, Any] | None]:
        try:
            timeout_s = max(0.001, float(deadline_ms) / 1000.0)
            if hasattr(plugin, "search_with_trace"):
                candidates, trace = await asyncio.wait_for(
                    plugin.search_with_trace(ctx, top_k=top_k, deadline_ms=deadline_ms),
                    timeout=timeout_s,
                )
            else:
                candidates = await asyncio.wait_for(
                    plugin.search(ctx, top_k=top_k, deadline_ms=deadline_ms),
                    timeout=timeout_s,
                )
                trace = None
            return list(candidates), False, None, trace
        except TimeoutError:
            logger.debug("retrieval_leg_timeout leg=%s deadline_ms=%s", leg_name, deadline_ms)
            return [], True, f"{leg_name}_timeout", None
        except Exception:  # pragma: no cover - defensive path
            logger.exception("retrieval_leg_error leg=%s", leg_name)
            return [], False, f"{leg_name}_error", None

    async def run(self, ctx: QueryContext) -> LocalizationDecision:
        shared_deadline_ms = self._retrieval_deadline_ms
        leg_deadline_ms = min(self._retrieval_leg_timeout_ms, shared_deadline_ms)
        reason_flags: list[str] = []

        dense_task = asyncio.create_task(
            self._run_leg(
                self._dense,
                ctx,
                top_k=self._top_k,
                deadline_ms=leg_deadline_ms,
                leg_name="dense",
            )
        )
        lexical_task = asyncio.create_task(
            self._run_leg(
                self._lexical,
                ctx,
                top_k=self._top_k,
                deadline_ms=leg_deadline_ms,
                leg_name="lexical",
            )
        )
        dense_candidates, dense_timeout, dense_flag, _ = await dense_task
        lexical_candidates, lexical_timeout, lexical_flag, toc_trace = await lexical_task
        if dense_flag:
            reason_flags.append(dense_flag)
        if lexical_flag:
            reason_flags.append(lexical_flag)

        fused_candidates = self._fusion.fuse(
            dense_candidates,
            lexical_candidates,
            top_k=self._top_k,
        )
        if not fused_candidates and (dense_candidates or lexical_candidates):
            fallback = list(dense_candidates) + list(lexical_candidates)
            deduped: dict[tuple[str, str], Candidate] = {}
            for candidate in fallback:
                key = (candidate.doc_id, candidate.chunk_id)
                current = deduped.get(key)
                if current is None or candidate.score > current.score:
                    deduped[key] = candidate
            fused_candidates = sorted(deduped.values(), key=lambda item: item.score, reverse=True)[: self._top_k]

        reranker_timeout = False
        reranker_degraded = False
        ranked_candidates = list(fused_candidates)
        if ranked_candidates:
            try:
                reranked = await asyncio.wait_for(
                    self._reranker.rerank(
                        ctx,
                        ranked_candidates,
                        top_n=self._rerank_top_n,
                        timeout_ms=self._reranker_timeout_ms,
                    ),
                    timeout=max(0.001, self._reranker_timeout_ms / 1000.0),
                )
                if reranked:
                    ranked_candidates = list(reranked)
                else:
                    reranker_degraded = True
                    reason_flags.append("reranker_empty_result")
            except TimeoutError:
                reranker_timeout = True
                reranker_degraded = True
                reason_flags.append("reranker_timeout")
            except Exception:  # pragma: no cover - defensive path
                reranker_degraded = True
                reason_flags.append("reranker_error")
                logger.exception("reranker_failure")

        decision = self._decision.decide(ctx, ranked_candidates)
        if reranker_degraded:
            decision = replace(decision, reranker_degraded=True)

        trace = RetrievalDebugTrace(
            dense_timeout=dense_timeout,
            lexical_timeout=lexical_timeout,
            reranker_timeout=reranker_timeout,
            reranker_degraded=reranker_degraded,
            reason_flags=tuple(reason_flags),
            toc_trace=toc_trace,
        )
        if self._trace_enabled or decision.abstained:
            decision = replace(decision, trace=trace)
        return decision


class RetrievalRequestHandler:
    def __init__(
        self,
        *,
        query_context_builder: QueryContextBuilder,
        retrieval_pipeline: RetrievalPipeline,
    ) -> None:
        self._query_context_builder = query_context_builder
        self._retrieval_pipeline = retrieval_pipeline

    async def handle(self, raw_query: str) -> LocalizationDecision:
        ctx = self._query_context_builder.build(raw_query)
        return await self._retrieval_pipeline.run(ctx)


def build_default_retrieval_stack(
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    toc_index: TOCIndex | None = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rrf_k: int = DEFAULT_RETRIEVAL_RRF_K,
    trace_enabled: bool = False,
    page_localization_min_confidence: float = DEFAULT_PAGE_LOCALIZATION_MIN_CONFIDENCE,
) -> tuple[QueryContextBuilder, RetrievalPipeline]:
    query_context_builder = DefaultQueryContextBuilder(toc_index=toc_index, toc_top_k=top_k)
    lexical_retriever: RetrieverPlugin
    if toc_index is not None:
        lexical_retriever = TOCLexicalRetrieverPlugin(toc_index=toc_index)
    else:
        # Avoid full-corpus scans by default. Swap in a real BM25/lexical plugin when available.
        lexical_retriever = NoopRetrieverPlugin()
    retrieval_pipeline = RetrievalPipeline(
        dense=DenseVectorRetrieverPlugin(embedder=embedder, vector_store=vector_store),
        lexical=lexical_retriever,
        fusion=RRFFusionPolicy(rrf_k=rrf_k),
        reranker=StableIdentityRerankerPlugin(warmup_on_start=True),
        decision=DefaultLocalizationDecisionPolicy(
            page_localization_min_confidence=page_localization_min_confidence
        ),
        top_k=top_k,
        trace_enabled=trace_enabled,
    )
    return query_context_builder, retrieval_pipeline


def localization_decision_to_hits(decision: LocalizationDecision) -> list[Hit]:
    hits: list[Hit] = []
    for candidate in decision.top_candidates:
        metadata = dict(candidate.metadata)
        metadata.setdefault("source_id", candidate.source_id)
        if candidate.page_start is not None:
            metadata.setdefault("start_page", candidate.page_start)
            metadata.setdefault("page", candidate.page_start)
        if candidate.page_end is not None:
            metadata.setdefault("end_page", candidate.page_end)
        document_text = str(metadata.get("document_text", "")).strip()
        document = Document(
            id=candidate.doc_id,
            text=document_text,
            metadata=metadata,
        )
        # Keep original retriever scores on the legacy Hit surface so existing thresholds
        # (min_score, OOS gate, adaptive attachment calibration) remain compatible.
        hits.append(Hit(document=document, score=float(candidate.score)))
    return hits