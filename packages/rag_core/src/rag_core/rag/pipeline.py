from __future__ import annotations

from collections.abc import Iterator, Sequence

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.citations import (
    DEFAULT_SOURCE_PDF_URL_TEMPLATE,
    build_references_suffix,
    hits_to_source_payload,
)
from rag_core.rag.prompts import build_chat_messages, build_context_block, build_system_prompt
from rag_core.types import ChatTurn, Hit


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

    def retrieve(self, question: str) -> list[Hit]:
        query_vector = self.embedder.embed_query(question)
        hits = self.vector_store.search(query_vector, top_k=self.top_k)
        return [hit for hit in hits if hit.score >= self.min_score]

    def _fallback_response(self, question: str) -> str:
        return 'I could not find strong supporting evidence in the current indexed corpus for this question. Please try a more specific phrasing, switch to a closer subject category, or refresh the corpus/index version. ' + f'Current corpus version: {self.corpus_version}. Question: {question}'

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
        hits = self.retrieve(question)
        source_payload = hits_to_source_payload(
            hits,
            self.corpus_version,
            source_pdf_url_template=self.source_pdf_url_template,
        )
        yield {
            'type': 'sources',
            'data': source_payload,
        }
        if not hits:
            for token in self._fallback_response(question).split(' '):
                yield {'type': 'delta', 'data': {'text': token + ' '}}
            return
        resolved_max_new_tokens, resolved_temperature = self.resolve_generation_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        system_prompt = build_system_prompt(question=question, hits=hits, corpus_version=self.corpus_version, default_language=self.default_language)
        context_block = build_context_block(hits)
        messages = build_chat_messages(question=question, history=history, context_block=context_block)
        generated_chunks: list[str] = []
        for token in self.llm.stream_chat(messages=messages, system_prompt=system_prompt, max_new_tokens=resolved_max_new_tokens, temperature=resolved_temperature):
            generated_chunks.append(token)
            yield {'type': 'delta', 'data': {'text': token}}
        references_suffix = build_references_suffix(
            answer_markdown=''.join(generated_chunks),
            sources=source_payload,
        )
        if references_suffix:
            yield {'type': 'delta', 'data': {'text': references_suffix}}
