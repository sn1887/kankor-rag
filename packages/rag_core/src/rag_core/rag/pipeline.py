from __future__ import annotations

from collections.abc import Iterator, Sequence
import re

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.adaptive_retrieval import (
    RetrievalAssessment,
    TopicLocatorFrontMatterPolicy,
    assess_retrieval_confidence,
    expand_local_window_hits,
    filter_topic_locator_hits,
    find_topic_locator_chapter_hits,
    merge_retrieval_hits,
)
from rag_core.rag.citations import (
    DEFAULT_SOURCE_PDF_URL_TEMPLATE,
    InlineCitationStripper,
    answer_includes_references_heading,
    hits_to_source_payload,
    select_reference_sources_for_answer,
    strip_inline_citation_markers,
)
from rag_core.rag.context_plugins import (
    GroundingContextPlugin,
    TextGroundingContextPlugin,
)
from rag_core.rag.intent_router import IntentRouter, RAGIntent, RETRIEVAL_INTENTS
from rag_core.rag.prompts import (
    build_system_prompt,
    build_task_directive,
)
from rag_core.rag.toc_locator import TOCIndex
from rag_core.rag.topic_locator_response import render_topic_locator_answer
from rag_core.types import ChatTurn, Hit


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
        default_language: str = 'auto',
        source_pdf_url_template: str | None = DEFAULT_SOURCE_PDF_URL_TEMPLATE,
        intent_router: IntentRouter | None = None,
        grounding_context_plugin: GroundingContextPlugin | None = None,
        local_expansion_neighbors: int = 1,
        retrieval_confidence_top_score: float = 0.27,
        retrieval_confidence_min_hits: int = 1,
        retrieval_oos_top_score_threshold: float | None = None,
        toc_index: TOCIndex | None = None,
        topic_locator_front_matter_policy: TopicLocatorFrontMatterPolicy | None = None,
        topic_locator_response_mode: str = "hybrid",
        references_max_sources: int = 3,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.corpus_version = corpus_version
        self.top_k = top_k
        self.min_score = min_score
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
        self.local_expansion_neighbors = max(0, int(local_expansion_neighbors))
        self.retrieval_confidence_top_score = float(retrieval_confidence_top_score)
        self.retrieval_confidence_min_hits = max(1, int(retrieval_confidence_min_hits))
        self.retrieval_oos_top_score_threshold = (
            float(retrieval_oos_top_score_threshold)
            if retrieval_oos_top_score_threshold is not None
            else None
        )
        self.toc_index = toc_index
        self.topic_locator_front_matter_policy = topic_locator_front_matter_policy or TopicLocatorFrontMatterPolicy()
        self.topic_locator_response_mode = (topic_locator_response_mode or "hybrid").strip().lower()
        self.references_max_sources = max(1, int(references_max_sources))

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
        normalized = cls._normalize_user_text(question)
        if re.search(r"[ټځڅډړږښګڼۍې]", question):
            return "سلام! څنګه مرسته درسره وکړم؟ د کانکور د مضمون، فصل یا مفهوم نوم راکړه."
        if any(term in normalized for term in ("مرحبا", "اهلا", "السلام")):
            return "مرحباً! كيف يمكنني مساعدتك في التحضير للكانكور؟ اكتب المادة أو الفصل أو المفهوم."
        if re.search(r"[a-z]", question.lower()):
            return "Hi! How can I help with your Kankor prep today? Tell me the subject, chapter, or concept."
        return "سلام! خوش آمدید. بگویید روی کدام مضمون، فصل یا مفهوم کانکور کار کنیم."

    def _search_query(self, question: str) -> list[Hit]:
        return self._search_query_with_options(question)

    def _search_query_with_options(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[Hit]:
        query_vector = self.embedder.embed_query(question)
        requested_k = self.top_k if top_k is None else max(1, int(top_k))
        return self.vector_store.search(query_vector, top_k=requested_k, filters=filters)

    @staticmethod
    def _coerce_positive_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @classmethod
    def _hit_overlaps_page_span(cls, hit: Hit, *, span_start: int, span_end: int) -> bool:
        if span_start > span_end:
            return False
        meta = hit.document.metadata
        start_page = cls._coerce_positive_int(meta.get("start_page", meta.get("page")))
        end_page = cls._coerce_positive_int(meta.get("end_page", meta.get("page")))
        if start_page is None or end_page is None:
            return False
        return end_page >= span_start and start_page <= span_end

    def retrieve(self, question: str) -> list[Hit]:
        return self.retrieve_many([question])

    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        filters: dict[str, str] | None = None,
        page_span: tuple[int, int] | None = None,
        overfetch_multiplier: int = 8,
    ) -> list[Hit]:
        resolved_overfetch = max(1, int(overfetch_multiplier))

        # Per-request dedupe to avoid repeated embedding calls for identical variants.
        cleaned_questions: list[str] = []
        seen_keys: set[str] = set()
        for question in questions:
            cleaned = str(question).strip()
            if not cleaned:
                continue
            normalized_key = re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).casefold()
            if normalized_key in seen_keys:
                continue
            seen_keys.add(normalized_key)
            cleaned_questions.append(cleaned)

        if not cleaned_questions:
            return []

        query_vectors = np.asarray(self.embedder.embed_queries(cleaned_questions), dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if query_vectors.shape[0] != len(cleaned_questions):
            raise ValueError(
                "embed_queries() must return an array of shape (len(texts), embedding_dim). "
                f"Got shape={getattr(query_vectors, 'shape', None)} for texts={len(cleaned_questions)}."
            )

        requested_k = self.top_k * resolved_overfetch
        hit_groups: list[list[Hit]] = []
        for query_vector in query_vectors:
            candidates = self.vector_store.search(query_vector, top_k=requested_k, filters=filters)
            if page_span is not None:
                span_start, span_end = page_span
                candidates = [
                    hit
                    for hit in candidates
                    if self._hit_overlaps_page_span(hit, span_start=span_start, span_end=span_end)
                ]
            hit_groups.append(candidates)
        if not hit_groups:
            return []
        return merge_retrieval_hits(
            hit_groups=hit_groups,
            top_k=self.top_k,
            min_score=self.min_score,
        )

    def _confidence_fallback_response(self) -> str:
        return "I can give general guidance, but I could not verify this from the textbooks."

    @staticmethod
    def _should_retrieve(intent: RAGIntent) -> bool:
        return intent in RETRIEVAL_INTENTS

    @staticmethod
    def _should_expand_neighbors(intent: RAGIntent) -> bool:
        return intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}

    @classmethod
    def _select_unambiguous_toc_route(cls, toc_hits: Sequence[Hit]) -> Hit | None:
        if not toc_hits:
            return None
        source_ids = {
            str(hit.document.metadata.get("source_id", "")).strip()
            for hit in toc_hits
            if str(hit.document.metadata.get("source_id", "")).strip()
        }
        if len(source_ids) == 1:
            return toc_hits[0]
        if len(toc_hits) >= 2:
            top = toc_hits[0]
            second = toc_hits[1]
            match_kind = str(top.document.metadata.get("toc_match_kind", "")).strip().lower()
            if match_kind == "title" and float(top.score) >= float(second.score) + 0.05:
                return top
        return None

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
    ) -> Iterator[dict]:
        route = self.intent_router.route(question=question, history=history)
        intent = route.intent

        if intent == RAGIntent.SMALLTALK:
            smalltalk = self._smalltalk_response(question) or "سلام! چگونه می‌توانم برای آمادگی کانکور کمک کنم؟"
            yield {'type': 'sources', 'data': []}
            yield {'type': 'delta', 'data': {'text': smalltalk}}
            return

        retrieval_assessment: RetrievalAssessment | None = None
        hits: list[Hit] = []
        if self._should_retrieve(intent):
            queries = route.retrieval_queries or [question]
            toc_hits: list[Hit] = []
            toc_route: Hit | None = None
            toc_short_circuit = False
            if self.toc_index is not None:
                if intent == RAGIntent.TOPIC_LOCATOR:
                    toc_hits = self.toc_index.search(question=question, top_k=self.top_k)
                elif intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}:
                    # Always consult the TOC for grounded intents. Many lesson/topic titles only exist
                    # in the manifest (not in window-text chunks), so TOC routing is a retrieval-quality win.
                    toc_hits = self.toc_index.search(question=question, top_k=self.top_k)
                toc_route = self._select_unambiguous_toc_route(toc_hits)

            if intent == RAGIntent.TOPIC_LOCATOR and toc_route is not None:
                toc_short_circuit = True
                hits = list(toc_hits)[: max(1, int(self.top_k))]
            elif toc_route is not None and intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}:
                route_meta = toc_route.document.metadata
                source_id = str(route_meta.get("source_id", "")).strip()
                span_start = self._coerce_positive_int(route_meta.get("start_page")) or self._coerce_positive_int(
                    route_meta.get("page")
                )
                span_end = self._coerce_positive_int(route_meta.get("end_page")) or span_start
                filters = {"source_id": source_id} if source_id else None
                page_span = (span_start, span_end) if span_start and span_end else None

                hits = self.retrieve_many(
                    queries,
                    filters=filters,
                    page_span=page_span,
                )
                # If the chapter span is mismatched or too narrow, fall back to the full book.
                if not hits and filters is not None:
                    hits = self.retrieve_many(
                        queries,
                        filters=filters,
                        page_span=None,
                    )
                # Last resort: global retrieval.
                if not hits:
                    hits = self.retrieve_many(queries)
            else:
                hits = self.retrieve_many(queries)
            if intent == RAGIntent.TOPIC_LOCATOR:
                if not toc_short_circuit:
                    if toc_hits:
                        hits = merge_retrieval_hits(
                            hit_groups=[toc_hits, hits],
                            top_k=self.top_k,
                            min_score=self.min_score,
                        )
                    chapter_hits = find_topic_locator_chapter_hits(
                        question=question,
                        vector_store=self.vector_store,
                        top_k=self.top_k,
                    )
                    if chapter_hits:
                        hits = merge_retrieval_hits(
                            hit_groups=[hits, chapter_hits],
                            top_k=self.top_k,
                            min_score=self.min_score,
                        )
                    hits = filter_topic_locator_hits(
                        hits=hits,
                        question=question,
                        top_k=self.top_k,
                        front_matter_policy=self.topic_locator_front_matter_policy,
                    )
            if self._should_expand_neighbors(intent):
                hits = expand_local_window_hits(
                    hits=hits,
                    vector_store=self.vector_store,
                    neighbors_per_side=self.local_expansion_neighbors,
                    max_hits=self.top_k + (self.local_expansion_neighbors * 2),
                )
            retrieval_assessment = assess_retrieval_confidence(
                hits=hits,
                min_top_score=self.retrieval_confidence_top_score,
                min_hits=self.retrieval_confidence_min_hits,
            )

        if (
            self._should_retrieve(intent)
            and self.retrieval_oos_top_score_threshold is not None
            and intent in {RAGIntent.GROUNDED_TEXTBOOK, RAGIntent.PRACTICE_GENERATION}
        ):
            top_score = float(hits[0].score) if hits else None
            if top_score is None or top_score < self.retrieval_oos_top_score_threshold:
                yield {'type': 'sources', 'data': []}
                yield {
                    'type': 'delta',
                    'data': {
                        'text': (
                            "این سوال احتمالاً در محدودهٔ کتاب‌های درسی موجود در سیستم نیست. "
                            "اگر سوال شما از کتاب‌های کانکور است، نام مضمون/صنف و چند کلیدواژهٔ دقیق‌تر را اضافه کنید."
                        )
                    },
                }
                return

        source_payload = (
            hits_to_source_payload(
                hits,
                self.corpus_version,
                source_pdf_url_template=self.source_pdf_url_template,
            )
            if self._should_retrieve(intent)
            else []
        )
        yield {'type': 'sources', 'data': source_payload}

        if (
            intent == RAGIntent.TOPIC_LOCATOR
            and self._should_retrieve(intent)
            and self.topic_locator_response_mode != "llm"
            and hits
        ):
            raw_answer = render_topic_locator_answer(hits=hits, max_candidates=min(3, self.top_k))
            cleaned_answer = strip_inline_citation_markers(raw_answer)
            if cleaned_answer:
                yield {'type': 'delta', 'data': {'text': cleaned_answer}}
            if source_payload:
                selected_sources, strategy = select_reference_sources_for_answer(
                    raw_answer=raw_answer,
                    cleaned_answer=cleaned_answer,
                    sources=source_payload,
                    max_sources=self.references_max_sources,
                )
                yield {
                    'type': 'references',
                    'data': {
                        'sources': selected_sources,
                        'strategy': strategy,
                        'answer_has_references_heading': answer_includes_references_heading(raw_answer),
                    },
                }
            return

        if self._should_retrieve(intent) and retrieval_assessment is not None and retrieval_assessment.weak:
            raw_answer = self._confidence_fallback_response()
            for token in raw_answer.split(' '):
                yield {'type': 'delta', 'data': {'text': token + ' '}}
            if source_payload:
                selected_sources, strategy = select_reference_sources_for_answer(
                    raw_answer=raw_answer,
                    cleaned_answer=raw_answer,
                    sources=source_payload,
                    max_sources=self.references_max_sources,
                )
                yield {
                    'type': 'references',
                    'data': {
                        'sources': selected_sources,
                        'strategy': strategy,
                        'answer_has_references_heading': answer_includes_references_heading(raw_answer),
                    },
                }
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
        )
        request = self.grounding_context_plugin.build(
            question=question,
            history=history,
            hits=hits,
            grounded=self._should_retrieve(intent),
            intent=intent.value,
            task_directive=task_directive,
            corpus_version=self.corpus_version,
        )
        if request.attachments and not self.llm.supports_attachments:
            request = TextGroundingContextPlugin().build(
                question=question,
                history=history,
                hits=hits,
                grounded=self._should_retrieve(intent),
                intent=intent.value,
                task_directive=task_directive,
                corpus_version=self.corpus_version,
            )
        raw_chunks: list[str] = []
        cleaned_chunks: list[str] = []
        stripper = InlineCitationStripper()
        for token in self.llm.stream_chat(
            messages=request.messages,
            system_prompt=system_prompt,
            max_new_tokens=resolved_max_new_tokens,
            temperature=resolved_temperature,
            attachments=request.attachments,
        ):
            raw_chunks.append(token)
            cleaned_piece = stripper.feed(token)
            if cleaned_piece:
                cleaned_chunks.append(cleaned_piece)
                yield {'type': 'delta', 'data': {'text': cleaned_piece}}

        tail = stripper.flush()
        if tail:
            cleaned_chunks.append(tail)
            yield {'type': 'delta', 'data': {'text': tail}}

        if self._should_retrieve(intent) and source_payload:
            raw_answer = ''.join(raw_chunks)
            cleaned_answer = ''.join(cleaned_chunks)
            selected_sources, strategy = select_reference_sources_for_answer(
                raw_answer=raw_answer,
                cleaned_answer=cleaned_answer,
                sources=source_payload,
                max_sources=self.references_max_sources,
            )
            yield {
                'type': 'references',
                'data': {
                    'sources': selected_sources,
                    'strategy': strategy,
                    'answer_has_references_heading': answer_includes_references_heading(raw_answer),
                },
            }
