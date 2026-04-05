from __future__ import annotations

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.contracts.reranker import Reranker
from rag_core.contracts.vector_store import VectorStore
from rag_core.eval.retrieval_benchmark import QueryQrels, evaluate_retrieval_rows
from rag_core.eval.retrieval_benchmark import benchmark_hybrid_reranked_retrieval
from rag_core.rag.retrieval import DenseRetriever, LexicalRetriever, PageRetrievalEngine, RRFFusionPolicy
from rag_core.types import Document, Hit


class _StubEmbedder(Embedder):
    def embed_documents(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_query(self, text):
        _ = text
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_queries(self, texts):
        return self.embed_documents(texts)


class _StaticVectorStore(VectorStore):
    def __init__(self, hits):
        self._hits = list(hits)
        self.documents = [hit.document for hit in hits]

    @property
    def size(self) -> int:
        return len(self.documents)

    def search(self, query_vector: np.ndarray, *, top_k: int, filters=None) -> list[Hit]:
        _ = query_vector, filters
        return list(self._hits[: max(1, int(top_k))])


class _PriorityReranker(Reranker):
    def __init__(self, scores_by_doc_id: dict[str, float]) -> None:
        self._scores_by_doc_id = dict(scores_by_doc_id)

    def rerank(self, *, query: str, hits, top_k: int | None = None) -> list[Hit]:
        _ = query
        limit = len(hits) if top_k is None else max(1, int(top_k))
        ranked = sorted(
            hits,
            key=lambda hit: self._scores_by_doc_id.get(hit.document.id, 0.0),
            reverse=True,
        )
        return list(ranked[:limit])


def test_evaluate_retrieval_rows_computes_expected_metrics() -> None:
    retrieval_rows = [
        {
            "query_id": "q1",
            "query": "sample",
            "intent": "grounded_textbook",
            "language": "fa",
            "origin": "chunk_test",
            "expected_subjects": ["history"],
            "hits": [
                {
                    "rank": 1,
                    "doc_id": "doc-1",
                    "source_id": "G10-Dr-History",
                    "start_page": 3,
                    "end_page": 3,
                }
            ],
        }
    ]
    qrels_by_query = {
        "q1": QueryQrels(
            doc_key_gains={"G10-Dr-History": 1.0},
            page_range_gains=[(3, 4, 1.0)],
        )
    }

    result = evaluate_retrieval_rows(
        retrieval_rows,
        qrels_by_query=qrels_by_query,
        k_values=(1, 5),
    )

    aggregate_at_1 = result["aggregate_by_k"]["1"]
    assert aggregate_at_1["mean_hit_rate_at_k"] == 1.0
    assert aggregate_at_1["mean_recall_at_k"] == 1.0
    assert aggregate_at_1["mean_mrr_at_k"] == 1.0

    per_query = result["per_query"][0]
    assert per_query["matched_target_count_at_1"] == 2
    assert per_query["first_relevant_rank_at_1"] == 1


def test_evaluate_retrieval_rows_ndcg_does_not_exceed_one_for_duplicate_hits() -> None:
    retrieval_rows = [
        {
            "query_id": "q1",
            "query": "sample",
            "intent": "grounded_textbook",
            "language": "fa",
            "origin": "chunk_test",
            "expected_subjects": ["history"],
            "hits": [
                {
                    "rank": rank,
                    "doc_id": f"doc-{rank}",
                    "source_id": "G10-Dr-History",
                    "start_page": 3,
                    "end_page": 3,
                }
                for rank in range(1, 6)
            ],
        }
    ]
    qrels_by_query = {
        "q1": QueryQrels(
            doc_key_gains={"G10-Dr-History": 1.0},
            page_range_gains=[(3, 4, 1.0)],
        )
    }

    result = evaluate_retrieval_rows(
        retrieval_rows,
        qrels_by_query=qrels_by_query,
        k_values=(5,),
    )

    aggregate_at_5 = result["aggregate_by_k"]["5"]
    assert aggregate_at_5["mean_ndcg_at_k"] <= 1.0
    per_query = result["per_query"][0]
    assert per_query["matched_target_count_at_5"] == 2


def test_benchmark_hybrid_reranked_retrieval_reorders_hits_before_eval() -> None:
    history_doc = Document(
        id="history-page-3",
        text="Afghanistan history lesson page three",
        metadata={
            "source_id": "G10-Dr-History",
            "subject": "history",
            "grade_band": "G10",
            "page": 3,
            "start_page": 3,
            "end_page": 3,
        },
    )
    chemistry_doc = Document(
        id="chemistry-page-8",
        text="Chemistry glossary and equations",
        metadata={
            "source_id": "G10-Dr-Chemistry",
            "subject": "chemistry",
            "grade_band": "G10",
            "page": 8,
            "start_page": 8,
            "end_page": 8,
        },
    )
    vector_store = _StaticVectorStore(
        [
            Hit(document=chemistry_doc, score=0.95),
            Hit(document=history_doc, score=0.80),
        ]
    )
    retrieval_engine = PageRetrievalEngine(
        dense_retriever=DenseRetriever(
            embedder=_StubEmbedder(),
            vector_store=vector_store,
            min_score=0.0,
        ),
        lexical_retriever=LexicalRetriever(
            vector_store=vector_store,
            min_score=0.0,
        ),
        fusion_policy=RRFFusionPolicy(rrf_k=20),
    )

    queries = [
        {
            "id": "q1",
            "query": "afghanistan history",
            "intent": "grounded_textbook",
            "language": "fa",
            "origin": "chunk_test",
            "expected_subjects": ["history"],
        }
    ]
    qrels_by_query = {
        "q1": QueryQrels(
            doc_key_gains={"G10-Dr-History": 1.0},
            page_range_gains=[(3, 3, 1.0)],
        )
    }
    reranker = _PriorityReranker({"history-page-3": 1.0, "chemistry-page-8": 0.0})

    result = benchmark_hybrid_reranked_retrieval(
        queries=queries,
        qrels_by_query=qrels_by_query,
        retrieval_engine=retrieval_engine,
        reranker=reranker,
        k_values=(1, 5),
        reranker_candidate_pool_size=2,
    )

    top_hit = result["retrieval_rows"][0]["hits"][0]
    assert top_hit["doc_id"] == "history-page-3"
    assert result["aggregate_by_k"]["1"]["mean_hit_rate_at_k"] == 1.0
