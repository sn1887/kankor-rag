from __future__ import annotations
from collections.abc import Iterator, Sequence
from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.llm import LLMProvider
from rag_core.contracts.vector_store import VectorStore
from rag_core.rag.citations import hits_to_source_payload
from rag_core.rag.prompts import build_chat_messages, build_context_block, build_system_prompt
from rag_core.types import ChatTurn, Hit

class RAGPipeline:
    def __init__(self, *, llm: LLMProvider, embedder: Embedder, vector_store: VectorStore, corpus_version: str, top_k: int = 5, min_score: float = 0.15, max_new_tokens: int = 256, temperature: float = 0.2, default_language: str = 'auto') -> None:
        self.llm = llm; self.embedder = embedder; self.vector_store = vector_store; self.corpus_version = corpus_version; self.top_k = top_k; self.min_score = min_score; self.max_new_tokens = max_new_tokens; self.temperature = temperature; self.default_language = default_language
    def retrieve(self, question: str) -> list[Hit]:
        query_vector = self.embedder.embed_query(question)
        hits = self.vector_store.search(query_vector, top_k=self.top_k)
        return [hit for hit in hits if hit.score >= self.min_score]
    def _fallback_response(self, question: str) -> str:
        return 'I could not find strong supporting evidence in the current indexed corpus for this question. Please try a more specific phrasing, switch to a closer subject category, or refresh the corpus/index version. ' + f'Current corpus version: {self.corpus_version}. Question: {question}'
    def stream_answer(self, *, question: str, history: Sequence[ChatTurn]) -> Iterator[dict]:
        hits = self.retrieve(question)
        yield {'type': 'sources', 'data': hits_to_source_payload(hits, self.corpus_version)}
        if not hits:
            for token in self._fallback_response(question).split(' '):
                yield {'type': 'delta', 'data': {'text': token + ' '}}
            return
        system_prompt = build_system_prompt(question=question, hits=hits, corpus_version=self.corpus_version, default_language=self.default_language)
        context_block = build_context_block(hits)
        messages = build_chat_messages(question=question, history=history, context_block=context_block)
        for token in self.llm.stream_chat(messages=messages, system_prompt=system_prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature):
            yield {'type': 'delta', 'data': {'text': token}}
