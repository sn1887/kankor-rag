from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import logging
import queue
import re
import threading
import time

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.progress import ProgressSink
from rag_core.contracts.reranker import NoopReranker, Reranker
from rag_core.contracts.structure_lookup import StructureLookup
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.adaptive_retrieval import (
    RetrievalAssessment,
    assess_retrieval_confidence,
    expand_local_window_hits,
)
from rag_core.rag.citations import (
    CitationPolicy,
    DEFAULT_SOURCE_PDF_URL_TEMPLATE,
    answer_includes_references_heading,
)
from rag_core.rag.context_plugins import (
    AdaptiveAttachmentRetrievalSignals,
    GroundingContextPlugin,
    TextGroundingContextPlugin,
)
from rag_core.rag.intent_router import IntentRouter, RAGIntent, RETRIEVAL_INTENTS
from rag_core.rag.prompts import build_system_prompt, build_task_directive, detect_answer_language, is_supportive_grounding_query
from rag_core.rag.progress import PipelineProgressReporter
from rag_core.rag.retrieval import (
    DenseRetriever,
    LexicalRetriever,
    PageRetrievalEngine,
    RRFFusionPolicy,
)
from rag_core.rag.topic_locator_response import render_topic_locator_answer
from rag_core.types import ChatTurn, Hit

logger = logging.getLogger(__name__)


_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "salam",
    "salaam",
    "سلام",
    "سلام علیکم",
    "سلام عليكم",
    "السلام علیکم",
    "السلام عليكم",
    "سلامونه",
    "درود",
    "مرحبا",
    "اهلا",
}

_GREETING_TOKENS = {
    "hi",
    "hello",
    "hey",
    "salam",
    "salaam",
    "سلام",
    "سلامونه",
    "درود",
    "مرحبا",
    "اهلا",
    "السلام",
    "علیکم",
    "عليكم",
}

_LOCATOR_PATTERNS = (
    "where is",
    "where can i find",
    "which chapter",
    "which lesson",
    "which page",
    "which section",
    "in which chapter",
    "where taught",
    "where covered",
    "در کدام فصل",
    "در کدام صفحه",
    "کدام فصل",
    "کدام صفحه",
    "کجاست",
    "په کوم فصل",
    "په کومه صفحه",
    "في أي فصل",
    "في أي صفحة",
)


class RAGPipeline:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: Embedder,
        vector_store: VectorStore,
        corpus_version: str,
        top_k: int = 5,
        min_score: float = 0.15,
        max_new_tokens: int = 256,
        max_new_tokens_limit: int | None = None,
        temperature: float = 0.2,
        temperature_min: float = 0.0,
        temperature_max: float = 2.0,
        default_language: str = "auto",
        source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
        intent_router: IntentRouter | None = None,
        grounding_context_plugin: GroundingContextPlugin | None = None,
        reranker: Reranker | None = None,
        structure_lookup: StructureLookup | None = None,
        reranker_candidate_pool_size: int = 20,
        reranker_timeout_ms: int = 1500,
        local_expansion_neighbors: int = 1,
        retrieval_confidence_top_score: float = 0.27,
        retrieval_confidence_min_hits: int = 1,
        retrieval_oos_top_score_threshold: float | None = None,
        references_max_sources: int = 3,
        citation_policy: CitationPolicy | None = None,
        lexical_min_score: float = 0.01,
        rrf_k: int = 20,
        retrieval_engine: PageRetrievalEngine | None = None,
        toc_index=None,
        topic_locator_front_matter_policy=None,
        topic_locator_response_mode: str = "deterministic",
        use_v6_retrieval: bool = False,
        v6_query_context_builder=None,
        v6_retrieval_pipeline=None,
        toc_trace_sample_rate: float = 0.0,
    ) -> None:
        _ = (
            toc_index,
            topic_locator_front_matter_policy,
            topic_locator_response_mode,
            use_v6_retrieval,
            v6_query_context_builder,
            v6_retrieval_pipeline,
            toc_trace_sample_rate,
        )
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.corpus_version = corpus_version
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.max_new_tokens = max(1, int(max_new_tokens))
        limit = self.max_new_tokens if max_new_tokens_limit is None else int(max_new_tokens_limit)
        self.max_new_tokens_limit = max(1, limit)
        self.temperature = float(temperature)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        if self.temperature_min > self.temperature_max:
            raise ValueError("temperature_min must be <= temperature_max")
        self.default_language = default_language
        self.source_pdf_url_template = source_pdf_url_template
        self.intent_router = intent_router or IntentRouter()
        self.grounding_context_plugin = grounding_context_plugin or TextGroundingContextPlugin()
        self.reranker = reranker or NoopReranker()
        self.reranker_enabled = not isinstance(self.reranker, NoopReranker)
        self.structure_lookup = structure_lookup
        self.reranker_candidate_pool_size = max(self.top_k, int(reranker_candidate_pool_size))
        self.reranker_timeout_ms = max(1, int(reranker_timeout_ms))
        self.local_expansion_neighbors = max(0, int(local_expansion_neighbors))
        self.retrieval_confidence_top_score = float(retrieval_confidence_top_score)
        self.retrieval_confidence_min_hits = max(1, int(retrieval_confidence_min_hits))
        self.retrieval_oos_top_score_threshold = (
            float(retrieval_oos_top_score_threshold)
            if retrieval_oos_top_score_threshold is not None
            else None
        )
        self.references_max_sources = max(1, int(references_max_sources))
        self.citation_policy = citation_policy or CitationPolicy()
        self.retrieval_engine = retrieval_engine or PageRetrievalEngine(
            dense_retriever=DenseRetriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                min_score=self.min_score,
            ),
            lexical_retriever=LexicalRetriever(
                vector_store=self.vector_store,
                min_score=lexical_min_score,
            ),
            fusion_policy=RRFFusionPolicy(rrf_k=rrf_k),
        )

    @staticmethod
    def _normalize_user_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()

    @classmethod
    def _is_greeting_only(cls, question: str) -> bool:
        normalized = cls._normalize_user_text(question)
        if not normalized or len(normalized) > 80:
            return False
        if normalized in _GREETING_PHRASES:
            return True
        tokens = normalized.split()
        if len(tokens) > 4:
            return False
        return all(token in _GREETING_TOKENS for token in tokens)

    @classmethod
    def _smalltalk_response(cls, question: str) -> str | None:
        if not cls._is_greeting_only(question):
            return None
        if detect_answer_language(question, "match-user") == "پښتو":
            return "سلام! څنګه مرسته درسره وکړم؟ د کانکور د مضمون، فصل یا مفهوم نوم راکړه."
        return "سلام! خوش آمدید. بگویید روی کدام مضمون، فصل یا مفهوم کانکور کار کنیم."

    @staticmethod
    def _diag_set(diagnostics: dict[str, object] | None, key: str, value: object) -> None:
        if diagnostics is None:
            return
        diagnostics[key] = value

    @staticmethod
    def _resolve_bool_override(
        overrides: Mapping[str, object] | None,
        *,
        key: str,
    ) -> bool | None:
        if overrides is None or key not in overrides:
            return None
        raw = overrides.get(key)
        if isinstance(raw, bool):
            return raw
        normalized = str(raw or "").strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    def _resolve_effective_reranker_enabled(
        self,
        *,
        overrides: Mapping[str, object] | None,
    ) -> bool:
        override_value = self._resolve_bool_override(overrides, key="reranker_enabled")
        if override_value is None:
            return self.reranker_enabled
        return bool(override_value) and self.reranker_enabled

    @staticmethod
    def _run_callable_with_timeout(
        func,
        *,
        timeout_ms: int,
    ) -> tuple[bool, object]:
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result = func()
            except BaseException as exc:  # pragma: no cover
                result_queue.put(("error", exc))
                return
            result_queue.put(("ok", result))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(max(0.001, float(timeout_ms) / 1000.0))
        if thread.is_alive():
            return True, TimeoutError(f"reranker timed out after {timeout_ms}ms")
        status, payload = result_queue.get_nowait()
        if status == "error":
            raise payload
        return False, payload

    def retrieve(self, question: str) -> list[Hit]:
        return self.retrieve_many([question])

    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        filters: dict[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
        overfetch_multiplier: int = 8,
        result_top_k: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> list[Hit]:
        top_k = self.top_k if result_top_k is None else max(1, int(result_top_k))
        search_top_k = top_k * max(1, int(overfetch_multiplier))
        hits, retrieval_diagnostics = self.retrieval_engine.search(
            questions=questions,
            top_k=search_top_k,
            filters=filters,
            page_span=page_span,
        )
        self._diag_set(diagnostics, "retrieval_query_count", retrieval_diagnostics.query_count)
        self._diag_set(diagnostics, "retrieval_dense_ms", retrieval_diagnostics.dense_ms)
        self._diag_set(diagnostics, "retrieval_lexical_ms", retrieval_diagnostics.lexical_ms)
        self._diag_set(diagnostics, "retrieval_fusion_ms", retrieval_diagnostics.fusion_ms)
        self._diag_set(diagnostics, "retrieval_vector_search_ms", retrieval_diagnostics.dense_ms)
        self._diag_set(diagnostics, "retrieval_search_calls", retrieval_diagnostics.dense_search_calls)
        self._diag_set(diagnostics, "retrieval_total_ms", retrieval_diagnostics.total_ms)
        return hits[:top_k]

    @staticmethod
    def _should_rerank_intent(intent: RAGIntent) -> bool:
        return intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}

    def _resolved_retrieval_result_top_k(self, *, intent: RAGIntent, reranker_enabled: bool | None = None) -> int:
        effective_reranker_enabled = self.reranker_enabled if reranker_enabled is None else bool(reranker_enabled)
        if effective_reranker_enabled and self._should_rerank_intent(intent):
            return self.reranker_candidate_pool_size
        return self.top_k

    def _rerank_hits(
        self,
        *,
        question: str,
        hits: Sequence[Hit],
        reranker_enabled: bool | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> list[Hit]:
        effective_reranker_enabled = self.reranker_enabled if reranker_enabled is None else bool(reranker_enabled)
        if not effective_reranker_enabled or not hits:
            return list(hits[: self.top_k])

        rerank_started = time.perf_counter()
        self._diag_set(diagnostics, "reranker_timeout", False)
        self._diag_set(diagnostics, "reranker_degraded", False)
        try:
            timed_out, ranked_or_timeout = self._run_callable_with_timeout(
                lambda: self.reranker.rerank(
                    query=question,
                    hits=hits,
                    top_k=self.top_k,
                ),
                timeout_ms=self.reranker_timeout_ms,
            )
            if timed_out:
                self._diag_set(diagnostics, "rerank_total_ms", int((time.perf_counter() - rerank_started) * 1000))
                self._diag_set(diagnostics, "reranker_timeout", True)
                self._diag_set(diagnostics, "reranker_degraded", True)
                self._diag_set(diagnostics, "reranker_failed", False)
                return list(hits[: self.top_k])
            ranked_hits = ranked_or_timeout
        except Exception:  # pragma: no cover
            logger.exception("reranker_failure")
            self._diag_set(diagnostics, "rerank_total_ms", int((time.perf_counter() - rerank_started) * 1000))
            self._diag_set(diagnostics, "reranker_timeout", False)
            self._diag_set(diagnostics, "reranker_degraded", True)
            self._diag_set(diagnostics, "reranker_failed", True)
            return list(hits[: self.top_k])

        self._diag_set(diagnostics, "rerank_total_ms", int((time.perf_counter() - rerank_started) * 1000))
        self._diag_set(diagnostics, "reranker_timeout", False)
        self._diag_set(diagnostics, "reranker_degraded", False)
        self._diag_set(diagnostics, "reranker_failed", False)
        self._diag_set(diagnostics, "rerank_load_ms", int(getattr(self.reranker, "last_call_load_ms", 0) or 0))
        self._diag_set(diagnostics, "rerank_inference_ms", int(getattr(self.reranker, "last_call_inference_ms", 0) or 0))
        if not ranked_hits:
            return list(hits[: self.top_k])
        return list(ranked_hits[: self.top_k])

    def _confidence_fallback_response(self, *, question: str) -> str:
        if detect_answer_language(question, self.default_language) == "پښتو":
            return (
                "زه عمومي مرسته درکولی شم، خو دا ځواب مې د درسي کتابونو له متن څخه په کافي توګه تایید نه کړ."
            )
        return (
            "می‌توانم راهنمایی عمومی بدهم، اما نتوانستم این پاسخ را با اطمینان کافی از متن کتاب‌های درسی تأیید کنم."
        )

    @staticmethod
    def _resolve_top_score(hits: Sequence[Hit]) -> float | None:
        if not hits:
            return None
        try:
            return float(hits[0].score)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _resolve_score_gap(cls, hits: Sequence[Hit]) -> float | None:
        if len(hits) < 2:
            return None
        top_score = cls._resolve_top_score(hits[:1])
        second_score = cls._resolve_top_score(hits[1:2])
        if top_score is None or second_score is None:
            return None
        return top_score - second_score

    @staticmethod
    def _should_retrieve(intent: RAGIntent) -> bool:
        return intent in RETRIEVAL_INTENTS

    @staticmethod
    def _should_expand_neighbors(intent: RAGIntent) -> bool:
        return intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}

    @classmethod
    def _is_locator_style_query(cls, question: str) -> bool:
        normalized = cls._normalize_user_text(question)
        if not normalized:
            return False
        if any(pattern in normalized for pattern in _LOCATOR_PATTERNS):
            return True
        if ("chapter" in normalized or "فصل" in normalized or "باب" in normalized) and any(
            marker in normalized for marker in ("چی است", "چه است", "title", "name of")
        ):
            return True
        return False

    @classmethod
    def _should_consult_structure_lookup(cls, *, question: str, route_is_broad: bool) -> bool:
        if cls._is_locator_style_query(question):
            return True
        normalized = cls._normalize_user_text(question)
        if "chapter" in normalized or "فصل" in normalized or "باب" in normalized:
            return True
        return route_is_broad

    @classmethod
    def _select_unambiguous_structure_route(cls, hits: Sequence[Hit]) -> Hit | None:
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        top_score = float(hits[0].score)
        second_score = float(hits[1].score)
        if top_score >= 0.75 and top_score >= second_score + 0.12:
            return hits[0]
        source_ids = {
            str(hit.document.metadata.get("source_id", "")).strip()
            for hit in hits[:3]
            if str(hit.document.metadata.get("source_id", "")).strip()
        }
        if len(source_ids) == 1 and top_score >= 0.65:
            return hits[0]
        return None

    @staticmethod
    def _filters_from_structure_hit(hit: Hit) -> tuple[dict[str, str] | None, tuple[int, int] | None]:
        metadata = hit.document.metadata
        source_id = str(metadata.get("source_id", "")).strip()
        start_page = metadata.get("start_page", metadata.get("page"))
        end_page = metadata.get("end_page", metadata.get("page"))
        try:
            page_span = (int(start_page), int(end_page))
        except (TypeError, ValueError):
            page_span = None
        filters = {"source_id": source_id} if source_id else None
        return filters, page_span

    def resolve_generation_params(
        self,
        *,
        max_new_tokens: int | None,
        temperature: float | None,
    ) -> tuple[int, float]:
        try:
            requested_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_new_tokens must be an integer") from exc
        resolved_tokens = max(1, min(requested_tokens, self.max_new_tokens_limit))

        try:
            requested_temperature = self.temperature if temperature is None else float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be numeric") from exc
        resolved_temperature = min(self.temperature_max, max(self.temperature_min, requested_temperature))
        return resolved_tokens, resolved_temperature

    def stream_answer(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        diagnostics: dict[str, object] | None = None,
        diagnostic_overrides: Mapping[str, object] | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> Iterator[dict]:
        request_started = time.perf_counter()
        progress = PipelineProgressReporter(
            sink=progress_sink,
            question=question,
            default_language=self.default_language,
        )
        progress.thinking()
        try:
            route_started = time.perf_counter()
            route = self.intent_router.route(question=question, history=history)
            self._diag_set(diagnostics, "intent_routing_ms", int((time.perf_counter() - route_started) * 1000))
            intent = route.intent
            effective_reranker_enabled = self._resolve_effective_reranker_enabled(overrides=diagnostic_overrides)
            self._diag_set(diagnostics, "intent", intent.value)
            self._diag_set(diagnostics, "reranker_enabled_effective", effective_reranker_enabled)
            if diagnostic_overrides:
                self._diag_set(diagnostics, "applied_overrides", dict(diagnostic_overrides))

            if intent == RAGIntent.SMALLTALK:
                smalltalk = self._smalltalk_response(question) or "سلام! چگونه می‌توانم برای آمادگی کانکور کمک کنم؟"
                self._diag_set(diagnostics, "no_token_emitted", False)
                self._diag_set(diagnostics, "first_delta_offset_ms", 0)
                self._diag_set(diagnostics, "llm_ttft_ms", 0)
                self._diag_set(diagnostics, "llm_completion_ms", 0)
                self._diag_set(diagnostics, "pipeline_total_ms", int((time.perf_counter() - request_started) * 1000))
                yield {"type": "sources", "data": []}
                progress.writing(metadata={"intent": intent.value, "source_count": 0, "response_mode": "smalltalk"})
                yield {"type": "delta", "data": {"text": smalltalk}}
                progress.done(metadata={"intent": intent.value})
                return

            hits: list[Hit] = []
            locator_hits: list[Hit] = []
            retrieval_assessment: RetrievalAssessment | None = None
            adaptive_retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None
            filters: dict[str, str] | None = None
            page_span: tuple[int, int] | None = None
            supportive_grounding = False
            locator_style = self._is_locator_style_query(question)
            if self._should_retrieve(intent):
                retrieval_result_top_k = self._resolved_retrieval_result_top_k(
                    intent=intent,
                    reranker_enabled=effective_reranker_enabled,
                )
                self._diag_set(diagnostics, "retrieval_result_top_k", retrieval_result_top_k)
                progress.retrieving(
                    metadata={
                        "intent": intent.value,
                        "locator_style": locator_style,
                        "route_is_broad": bool(route.is_broad),
                    }
                )

                if self.structure_lookup is not None and self._should_consult_structure_lookup(
                    question=question,
                    route_is_broad=route.is_broad,
                ):
                    locator_hits = self.structure_lookup.search(question=question, top_k=self.top_k)
                    self._diag_set(diagnostics, "structure_lookup_hit_count", len(locator_hits))
                    structure_route = self._select_unambiguous_structure_route(locator_hits)
                    if structure_route is not None:
                        filters, page_span = self._filters_from_structure_hit(structure_route)
                        self._diag_set(diagnostics, "structure_lookup_selected", structure_route.document.id)
                        self._diag_set(diagnostics, "structure_lookup_selected_score", float(structure_route.score))
                        if locator_style and float(structure_route.score) >= 0.75:
                            source_payload = self.citation_policy.build_source_payload(
                                hits=locator_hits[: self.top_k],
                                corpus_version=self.corpus_version,
                                source_pdf_url_template=self.source_pdf_url_template,
                            )
                            progress.reading(
                                metadata={"intent": intent.value, "source_count": len(source_payload), "mode": "locator"}
                            )
                            yield {"type": "sources", "data": source_payload}
                            raw_answer = render_topic_locator_answer(hits=locator_hits, max_candidates=min(3, self.top_k))
                            if raw_answer:
                                self._diag_set(diagnostics, "no_token_emitted", False)
                                self._diag_set(diagnostics, "first_delta_offset_ms", int((time.perf_counter() - request_started) * 1000))
                                self._diag_set(diagnostics, "llm_ttft_ms", None)
                                self._diag_set(diagnostics, "llm_completion_ms", 0)
                                progress.writing(
                                    metadata={"intent": intent.value, "source_count": len(source_payload), "mode": "locator"}
                                )
                                yield {"type": "delta", "data": {"text": raw_answer}}
                            if source_payload:
                                selection = self.citation_policy.select_reference_sources_for_answer(
                                    raw_answer=raw_answer,
                                    cleaned_answer=raw_answer,
                                    sources=source_payload,
                                    max_sources=self.references_max_sources,
                                )
                                yield {
                                    "type": "references",
                                    "data": {
                                        "sources": selection.sources,
                                        "strategy": selection.strategy,
                                        "answer_has_references_heading": answer_includes_references_heading(raw_answer),
                                    },
                                }
                            self._diag_set(diagnostics, "pipeline_total_ms", int((time.perf_counter() - request_started) * 1000))
                            progress.done(metadata={"intent": intent.value, "mode": "locator"})
                            return

                queries = route.retrieval_queries or [question]
                hits = self.retrieve_many(
                    queries,
                    filters=filters,
                    page_span=page_span,
                    result_top_k=retrieval_result_top_k,
                    diagnostics=diagnostics,
                )
                if not hits and filters is not None:
                    hits = self.retrieve_many(
                        queries,
                        filters=filters,
                        result_top_k=retrieval_result_top_k,
                        diagnostics=diagnostics,
                    )
                if not hits and (filters is not None or page_span is not None):
                    hits = self.retrieve_many(
                        queries,
                        result_top_k=retrieval_result_top_k,
                        diagnostics=diagnostics,
                    )

                if effective_reranker_enabled and self._should_rerank_intent(intent):
                    hits = self._rerank_hits(
                        question=question,
                        hits=hits,
                        reranker_enabled=effective_reranker_enabled,
                        diagnostics=diagnostics,
                    )

                if self._should_expand_neighbors(intent):
                    pre_expansion_top_score = self._resolve_top_score(hits)
                    pre_expansion_score_gap = self._resolve_score_gap(hits)
                    ranked_hits = list(hits)
                    neighbor_started = time.perf_counter()
                    expanded_hits = expand_local_window_hits(
                        hits=hits,
                        vector_store=self.vector_store,
                        neighbors_per_side=self.local_expansion_neighbors,
                        max_hits=self.top_k + (self.local_expansion_neighbors * 2),
                    )
                    self._diag_set(diagnostics, "neighbor_expansion_ms", int((time.perf_counter() - neighbor_started) * 1000))
                    adaptive_retrieval_signals = AdaptiveAttachmentRetrievalSignals(
                        top_score_pre_expansion=pre_expansion_top_score,
                        score_gap_pre_expansion=pre_expansion_score_gap,
                        top_score_post_expansion=self._resolve_top_score(expanded_hits),
                        score_gap_post_expansion=self._resolve_score_gap(expanded_hits),
                    )
                    seen_ids: set[str] = set()
                    merged_hits: list[Hit] = []
                    for hit in ranked_hits + expanded_hits:
                        if hit.document.id in seen_ids:
                            continue
                        seen_ids.add(hit.document.id)
                        merged_hits.append(hit)
                    hits = merged_hits[: max(1, self.top_k + (self.local_expansion_neighbors * 2))]

                retrieval_assessment = assess_retrieval_confidence(
                    hits=hits,
                    min_top_score=self.retrieval_confidence_top_score,
                    min_hits=self.retrieval_confidence_min_hits,
                )
                supportive_grounding = is_supportive_grounding_query(
                    question=question,
                    hits=hits,
                    intent=intent.value,
                )
                self._diag_set(
                    diagnostics,
                    "grounding_mode",
                    "supportive_stem" if supportive_grounding else "strict_grounded",
                )
                self._diag_set(diagnostics, "retrieved_hit_count", len(hits))

            if (
                self._should_retrieve(intent)
                and self.retrieval_oos_top_score_threshold is not None
                and intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}
            ):
                top_score = float(hits[0].score) if hits else None
                if top_score is None or top_score < self.retrieval_oos_top_score_threshold:
                    self._diag_set(diagnostics, "no_token_emitted", False)
                    self._diag_set(diagnostics, "first_delta_offset_ms", int((time.perf_counter() - request_started) * 1000))
                    self._diag_set(diagnostics, "llm_ttft_ms", None)
                    self._diag_set(diagnostics, "llm_completion_ms", 0)
                    self._diag_set(diagnostics, "pipeline_total_ms", int((time.perf_counter() - request_started) * 1000))
                    progress.reading(metadata={"intent": intent.value, "source_count": 0, "response_mode": "out_of_scope"})
                    yield {"type": "sources", "data": []}
                    progress.writing(metadata={"intent": intent.value, "source_count": 0, "response_mode": "out_of_scope"})
                    yield {
                        "type": "delta",
                        "data": {
                            "text": (
                                "این سوال احتمالاً در محدودهٔ کتاب‌های درسی موجود در سیستم نیست. "
                                "اگر سوال شما از کتاب‌های کانکور است، نام مضمون/صنف و چند کلیدواژهٔ دقیق‌تر را اضافه کنید."
                            )
                        },
                    }
                    progress.done(metadata={"intent": intent.value, "response_mode": "out_of_scope"})
                    return

            source_payload = (
                self.citation_policy.build_source_payload(
                    hits=hits,
                    corpus_version=self.corpus_version,
                    source_pdf_url_template=self.source_pdf_url_template,
                )
                if self._should_retrieve(intent)
                else []
            )
            progress.reading(metadata={"intent": intent.value, "source_count": len(source_payload)})
            yield {"type": "sources", "data": source_payload}

            if self._should_retrieve(intent) and retrieval_assessment is not None and retrieval_assessment.weak:
                if not supportive_grounding:
                    raw_answer = self._confidence_fallback_response(question=question)
                    first_fallback_delta = True
                    progress.writing(metadata={"intent": intent.value, "source_count": len(source_payload), "response_mode": "fallback"})
                    for token in raw_answer.split(" "):
                        if first_fallback_delta:
                            self._diag_set(diagnostics, "first_delta_offset_ms", int((time.perf_counter() - request_started) * 1000))
                            self._diag_set(diagnostics, "no_token_emitted", False)
                            self._diag_set(diagnostics, "llm_ttft_ms", None)
                            self._diag_set(diagnostics, "llm_completion_ms", 0)
                            first_fallback_delta = False
                        yield {"type": "delta", "data": {"text": token + " "}}
                    if source_payload:
                        selection = self.citation_policy.select_reference_sources_for_answer(
                            raw_answer=raw_answer,
                            cleaned_answer=raw_answer,
                            sources=source_payload,
                            max_sources=self.references_max_sources,
                        )
                        yield {
                            "type": "references",
                            "data": {
                                "sources": selection.sources,
                                "strategy": selection.strategy,
                                "answer_has_references_heading": answer_includes_references_heading(raw_answer),
                            },
                        }
                    self._diag_set(diagnostics, "pipeline_total_ms", int((time.perf_counter() - request_started) * 1000))
                    progress.done(metadata={"intent": intent.value, "response_mode": "fallback"})
                    return

            resolved_max_new_tokens, resolved_temperature = self.resolve_generation_params(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            task_directive = build_task_directive(
                question=question,
                hits=hits,
                intent=intent.value,
            )
            system_prompt = build_system_prompt(
                question=question,
                hits=hits,
                corpus_version=self.corpus_version,
                default_language=self.default_language,
                intent=intent.value,
                supportive_grounding=supportive_grounding,
            )
            request = self.grounding_context_plugin.build(
                question=question,
                history=history,
                hits=hits,
                grounded=self._should_retrieve(intent),
                intent=intent.value,
                task_directive=task_directive,
                corpus_version=self.corpus_version,
                retrieval_signals=adaptive_retrieval_signals,
                supportive_grounding=supportive_grounding,
            )
            for key, value in request.diagnostics.items():
                self._diag_set(diagnostics, key, value)
            if request.attachments and not self.llm.supports_attachments:
                request = TextGroundingContextPlugin().build(
                    question=question,
                    history=history,
                    hits=hits,
                    grounded=self._should_retrieve(intent),
                    intent=intent.value,
                    task_directive=task_directive,
                    corpus_version=self.corpus_version,
                    retrieval_signals=adaptive_retrieval_signals,
                    supportive_grounding=supportive_grounding,
                )
                self._diag_set(diagnostics, "attachment_fallback_to_text", True)

            self._diag_set(diagnostics, "llm_start_offset_ms", int((time.perf_counter() - request_started) * 1000))
            raw_chunks: list[str] = []
            visible_chunks: list[str] = []
            citation_processor = self.citation_policy.create_answer_stream_processor()
            llm_started = time.perf_counter()
            first_delta_recorded = False
            progress.writing(metadata={"intent": intent.value, "source_count": len(source_payload), "response_mode": "llm"})
            try:
                for token in self.llm.stream_chat(
                    messages=request.messages,
                    system_prompt=system_prompt,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                    attachments=request.attachments,
                ):
                    raw_chunks.append(token)
                    visible_piece = citation_processor.feed(token)
                    if visible_piece:
                        if not first_delta_recorded:
                            first_delta_recorded = True
                            self._diag_set(diagnostics, "llm_ttft_ms", int((time.perf_counter() - llm_started) * 1000))
                            self._diag_set(diagnostics, "first_delta_offset_ms", int((time.perf_counter() - request_started) * 1000))
                            self._diag_set(diagnostics, "no_token_emitted", False)
                        visible_chunks.append(visible_piece)
                        yield {"type": "delta", "data": {"text": visible_piece}}

                tail = citation_processor.flush()
                if tail:
                    if not first_delta_recorded:
                        first_delta_recorded = True
                        self._diag_set(diagnostics, "llm_ttft_ms", int((time.perf_counter() - llm_started) * 1000))
                        self._diag_set(diagnostics, "first_delta_offset_ms", int((time.perf_counter() - request_started) * 1000))
                        self._diag_set(diagnostics, "no_token_emitted", False)
                    visible_chunks.append(tail)
                    yield {"type": "delta", "data": {"text": tail}}
            finally:
                if not first_delta_recorded:
                    self._diag_set(diagnostics, "no_token_emitted", True)
                    self._diag_set(diagnostics, "llm_ttft_ms", None)
                    self._diag_set(diagnostics, "first_delta_offset_ms", None)
                self._diag_set(diagnostics, "llm_completion_ms", int((time.perf_counter() - llm_started) * 1000))
                self._diag_set(diagnostics, "pipeline_total_ms", int((time.perf_counter() - request_started) * 1000))

            if self._should_retrieve(intent) and source_payload:
                raw_answer = "".join(raw_chunks)
                visible_answer = "".join(visible_chunks)
                selection = self.citation_policy.select_reference_sources_for_answer(
                    raw_answer=raw_answer,
                    cleaned_answer=visible_answer,
                    sources=source_payload,
                    max_sources=self.references_max_sources,
                )
                yield {
                    "type": "references",
                    "data": {
                        "sources": selection.sources,
                        "strategy": selection.strategy,
                        "answer_has_references_heading": answer_includes_references_heading(raw_answer),
                    },
                }
            progress.done(metadata={"intent": intent.value, "source_count": len(source_payload)})
        except Exception as exc:
            progress.error(metadata={"message": str(exc)})
            raise
        finally:
            progress.close()
