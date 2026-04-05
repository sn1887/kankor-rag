from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from rag_core.contracts.reranker import Reranker
from rag_core.rag.retrieval import DenseRetriever, PageRetrievalEngine


@dataclass(frozen=True, slots=True)
class QueryQrels:
    doc_key_gains: dict[str, float]
    page_range_gains: list[tuple[int, int, float]]

    @property
    def target_count(self) -> int:
        return len(self.doc_key_gains) + len(self.page_range_gains)

    @property
    def target_gains(self) -> list[float]:
        return list(self.doc_key_gains.values()) + [gain for _, _, gain in self.page_range_gains]


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")
    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{resolved_path} must contain a JSON list.")
        return [dict(item) for item in payload if isinstance(item, dict)]

    rows: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {resolved_path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Each row in {resolved_path} must be a JSON object.")
            rows.append(dict(item))
    return rows


def load_query_suite(
    path: str | Path,
    *,
    allowed_intents: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _load_json_rows(path)
    allowed = {str(value).strip() for value in (allowed_intents or ()) if str(value).strip()}
    queries: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        query = " ".join(str(row.get("query") or "").split()).strip()
        if not query:
            continue
        intent = str(row.get("intent") or "").strip()
        if allowed and intent not in allowed:
            continue
        query_id = str(row.get("id") or "").strip() or f"query_{index:04d}"
        query_row = dict(row)
        query_row["id"] = query_id
        query_row["query"] = query
        query_row["intent"] = intent
        queries.append(query_row)
    return queries


def load_qrels(path: str | Path) -> dict[str, QueryQrels]:
    rows = _load_json_rows(path)
    qrels_by_query: dict[str, QueryQrels] = {}
    for index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or row.get("id") or "").strip()
        if not query_id:
            raise ValueError(f"qrels row {index} is missing query_id.")

        doc_key_gains: dict[str, float] = {}
        relevant_doc_keys = row.get("relevant_doc_keys")
        if isinstance(relevant_doc_keys, list):
            for raw_key in relevant_doc_keys:
                key = str(raw_key or "").strip()
                if key:
                    doc_key_gains[key] = 1.0
        elif isinstance(relevant_doc_keys, dict):
            for raw_key, raw_gain in relevant_doc_keys.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                doc_key_gains[key] = float(raw_gain)
        elif relevant_doc_keys is not None:
            raise ValueError(f"qrels query_id={query_id}: relevant_doc_keys must be a list or object.")

        page_range_gains: list[tuple[int, int, float]] = []
        relevant_ranges = row.get("relevant_ranges") or []
        if not isinstance(relevant_ranges, list):
            raise ValueError(f"qrels query_id={query_id}: relevant_ranges must be a list.")
        for item in relevant_ranges:
            if not isinstance(item, list | tuple) or len(item) not in {2, 3}:
                raise ValueError(
                    f"qrels query_id={query_id}: each relevant_ranges item must be [start,end] or [start,end,gain]."
                )
            start_page = _coerce_positive_int(item[0])
            end_page = _coerce_positive_int(item[1])
            if start_page is None or end_page is None or start_page > end_page:
                raise ValueError(f"qrels query_id={query_id}: invalid range [{item[0]}, {item[1]}].")
            gain = float(item[2]) if len(item) == 3 else 1.0
            page_range_gains.append((start_page, end_page, gain))

        if not doc_key_gains and not page_range_gains:
            raise ValueError(f"qrels query_id={query_id}: provide at least one relevance target.")

        qrels_by_query[query_id] = QueryQrels(
            doc_key_gains=doc_key_gains,
            page_range_gains=page_range_gains,
        )
    return qrels_by_query


def _hit_to_row(hit, *, rank: int) -> dict[str, Any]:
    metadata = hit.document.metadata
    page = _coerce_positive_int(metadata.get("page"))
    start_page = _coerce_positive_int(metadata.get("start_page", page))
    end_page = _coerce_positive_int(metadata.get("end_page", page))
    return {
        "rank": rank,
        "score": float(hit.score),
        "doc_id": hit.document.id,
        "source_id": str(metadata.get("source_id") or "").strip(),
        "subject": str(metadata.get("subject") or "").strip(),
        "grade_band": str(metadata.get("grade_band") or "").strip(),
        "page": page,
        "start_page": start_page,
        "end_page": end_page,
        "chapter_title": str(metadata.get("resolved_chapter_title") or metadata.get("chapter_title") or "").strip(),
        "topic_title": str(metadata.get("resolved_topic_title") or metadata.get("topic_title") or "").strip(),
        "is_glossary": bool(metadata.get("is_glossary")),
        "structure_source": str(metadata.get("structure_source") or "").strip(),
    }


def run_dense_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    dense_retriever: DenseRetriever,
    top_k: int,
    query_batch_size: int = 64,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hit_groups, search_calls = dense_retriever.search_per_query(
        questions=[str(query["query"]) for query in queries],
        top_k=top_k,
        batch_size=query_batch_size,
    )
    for query, hits in zip(queries, hit_groups, strict=False):
        rows.append(
            {
                "query_id": str(query["id"]),
                "query": str(query["query"]),
                "intent": str(query.get("intent") or ""),
                "language": str(query.get("language") or ""),
                "origin": str(query.get("origin") or ""),
                "expected_subjects": list(query.get("expected_subjects") or []),
                "dense_search_calls": 1 if str(query["query"]).strip() else 0,
                "dense_batch_search_calls_total": search_calls,
                "hits": [_hit_to_row(hit, rank=rank) for rank, hit in enumerate(hits, start=1)],
            }
        )
    return rows


def run_hybrid_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    retrieval_engine: PageRetrievalEngine,
    top_k: int,
    query_batch_size: int = 64,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dense_hit_groups, search_calls = retrieval_engine._dense_retriever.search_per_query(
        questions=[str(query["query"]) for query in queries],
        top_k=top_k,
        batch_size=query_batch_size,
    )
    for query, dense_hits in zip(queries, dense_hit_groups, strict=False):
        lexical_hits = retrieval_engine._lexical_retriever.search_many(
            questions=[str(query["query"])],
            top_k=top_k,
        )
        hits = retrieval_engine._fusion_policy.fuse(
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            top_k=top_k,
        )
        rows.append(
            {
                "query_id": str(query["id"]),
                "query": str(query["query"]),
                "intent": str(query.get("intent") or ""),
                "language": str(query.get("language") or ""),
                "origin": str(query.get("origin") or ""),
                "expected_subjects": list(query.get("expected_subjects") or []),
                "dense_search_calls": 1 if str(query["query"]).strip() else 0,
                "dense_batch_search_calls_total": search_calls,
                "hits": [_hit_to_row(hit, rank=rank) for rank, hit in enumerate(hits, start=1)],
            }
        )
    return rows


def run_hybrid_reranked_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    retrieval_engine: PageRetrievalEngine,
    reranker: Reranker,
    top_k: int,
    reranker_candidate_pool_size: int | None = None,
    query_batch_size: int = 64,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_pool_size = max(
        max(1, int(top_k)),
        max(1, int(reranker_candidate_pool_size or top_k)),
    )
    dense_hit_groups, search_calls = retrieval_engine._dense_retriever.search_per_query(
        questions=[str(query["query"]) for query in queries],
        top_k=candidate_pool_size,
        batch_size=query_batch_size,
    )
    for query, dense_hits in zip(queries, dense_hit_groups, strict=False):
        normalized_query = str(query["query"])
        lexical_hits = retrieval_engine._lexical_retriever.search_many(
            questions=[normalized_query],
            top_k=candidate_pool_size,
        )
        fused_hits = retrieval_engine._fusion_policy.fuse(
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            top_k=candidate_pool_size,
        )
        rerank_started = time.perf_counter()
        reranked_hits = reranker.rerank(
            query=normalized_query,
            hits=fused_hits,
            top_k=top_k,
        )
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
        rows.append(
            {
                "query_id": str(query["id"]),
                "query": normalized_query,
                "intent": str(query.get("intent") or ""),
                "language": str(query.get("language") or ""),
                "origin": str(query.get("origin") or ""),
                "expected_subjects": list(query.get("expected_subjects") or []),
                "dense_search_calls": 1 if normalized_query.strip() else 0,
                "dense_batch_search_calls_total": search_calls,
                "reranker_candidate_pool_size": candidate_pool_size,
                "rerank_input_count": len(fused_hits),
                "rerank_ms": rerank_ms,
                "hits": [_hit_to_row(hit, rank=rank) for rank, hit in enumerate(reranked_hits, start=1)],
            }
        )
    return rows


def _range_overlap(start_page: int, end_page: int, target_start: int, target_end: int) -> bool:
    return end_page >= target_start and start_page <= target_end


def _hit_doc_keys(hit: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("source_id", "doc_id", "id"):
        value = str(hit.get(field) or "").strip()
        if value:
            keys.append(value)
    return keys


def _dcg(values: Sequence[float]) -> float:
    total = 0.0
    for rank, value in enumerate(values, start=1):
        if value <= 0.0:
            continue
        total += float(value) / math.log2(rank + 1.0)
    return total


def _new_match_gain_for_hit(
    hit: dict[str, Any],
    *,
    qrel: QueryQrels,
    matched_targets: set[tuple[str, object]],
) -> tuple[float, set[tuple[str, object]]]:
    candidate_matches: list[tuple[tuple[str, object], float]] = []

    for key in _hit_doc_keys(hit):
        target_gain = qrel.doc_key_gains.get(key)
        if target_gain is None:
            continue
        candidate_matches.append((("doc", key), float(target_gain)))

    start_page = _coerce_positive_int(hit.get("start_page"))
    end_page = _coerce_positive_int(hit.get("end_page"))
    if start_page is not None and end_page is not None:
        for index, (target_start, target_end, target_gain) in enumerate(qrel.page_range_gains):
            if not _range_overlap(start_page, end_page, target_start, target_end):
                continue
            candidate_matches.append((("range", index), float(target_gain)))

    new_matches = {
        target_key
        for target_key, _ in candidate_matches
        if target_key not in matched_targets
    }
    if not new_matches:
        return 0.0, set()

    gain = max(
        gain_value
        for target_key, gain_value in candidate_matches
        if target_key in new_matches
    )
    return gain, new_matches


def evaluate_retrieval_rows(
    retrieval_rows: Sequence[dict[str, Any]],
    *,
    qrels_by_query: dict[str, QueryQrels],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    normalized_k_values = tuple(sorted({max(1, int(value)) for value in k_values}))
    per_query: list[dict[str, Any]] = []

    aggregate_buckets: dict[int, dict[str, list[float] | int]] = {
        k: {
            "hit_rate": [],
            "recall": [],
            "mrr": [],
            "ndcg": [],
            "skipped_no_hits": 0,
        }
        for k in normalized_k_values
    }

    for row in retrieval_rows:
        query_id = str(row.get("query_id") or "").strip()
        qrel = qrels_by_query.get(query_id)
        if qrel is None:
            continue

        hits = row.get("hits")
        ranked_hits = hits if isinstance(hits, list) else []
        base_row = {
            "query_id": query_id,
            "query": str(row.get("query") or "").strip(),
            "intent": str(row.get("intent") or "").strip(),
            "language": str(row.get("language") or "").strip(),
            "origin": str(row.get("origin") or "").strip(),
            "expected_subjects": list(row.get("expected_subjects") or []),
            "relevant_target_count": qrel.target_count,
        }

        for k in normalized_k_values:
            gains_at_k: list[float] = []
            matched_targets: set[tuple[str, object]] = set()
            first_relevant_rank: int | None = None

            for rank, hit in enumerate(ranked_hits[:k], start=1):
                typed_hit = dict(hit) if isinstance(hit, dict) else {}
                gain, new_matches = _new_match_gain_for_hit(
                    typed_hit,
                    qrel=qrel,
                    matched_targets=matched_targets,
                )
                matched_targets.update(new_matches)
                gains_at_k.append(gain)
                if first_relevant_rank is None and gain > 0.0:
                    first_relevant_rank = rank

            hit_rate_at_k = 1.0 if first_relevant_rank is not None else 0.0
            recall_at_k = (len(matched_targets) / qrel.target_count) if qrel.target_count > 0 else 0.0
            mrr_at_k = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
            ideal_gains = sorted(qrel.target_gains, reverse=True)[:k]
            ndcg_at_k = (_dcg(gains_at_k) / _dcg(ideal_gains)) if ideal_gains and _dcg(ideal_gains) > 0.0 else 0.0

            base_row[f"hit_rate_at_{k}"] = hit_rate_at_k
            base_row[f"recall_at_{k}"] = recall_at_k
            base_row[f"mrr_at_{k}"] = mrr_at_k
            base_row[f"ndcg_at_{k}"] = ndcg_at_k
            base_row[f"first_relevant_rank_at_{k}"] = first_relevant_rank
            base_row[f"matched_target_count_at_{k}"] = len(matched_targets)

            aggregate_bucket = aggregate_buckets[k]
            cast_hit_rate = aggregate_bucket["hit_rate"]
            cast_recall = aggregate_bucket["recall"]
            cast_mrr = aggregate_bucket["mrr"]
            cast_ndcg = aggregate_bucket["ndcg"]
            assert isinstance(cast_hit_rate, list)
            assert isinstance(cast_recall, list)
            assert isinstance(cast_mrr, list)
            assert isinstance(cast_ndcg, list)
            cast_hit_rate.append(hit_rate_at_k)
            cast_recall.append(recall_at_k)
            cast_mrr.append(mrr_at_k)
            cast_ndcg.append(ndcg_at_k)
            if not ranked_hits:
                aggregate_bucket["skipped_no_hits"] = int(aggregate_bucket["skipped_no_hits"]) + 1

        per_query.append(base_row)

    aggregate_by_k: dict[str, dict[str, Any]] = {}
    for k in normalized_k_values:
        bucket = aggregate_buckets[k]
        hit_rate_values = bucket["hit_rate"]
        recall_values = bucket["recall"]
        mrr_values = bucket["mrr"]
        ndcg_values = bucket["ndcg"]
        assert isinstance(hit_rate_values, list)
        assert isinstance(recall_values, list)
        assert isinstance(mrr_values, list)
        assert isinstance(ndcg_values, list)
        aggregate_by_k[str(k)] = {
            "k": k,
            "query_count_evaluated": len(hit_rate_values),
            "query_count_skipped_no_hits": int(bucket["skipped_no_hits"]),
            "mean_hit_rate_at_k": (sum(hit_rate_values) / len(hit_rate_values)) if hit_rate_values else None,
            "mean_recall_at_k": (sum(recall_values) / len(recall_values)) if recall_values else None,
            "mean_mrr_at_k": (sum(mrr_values) / len(mrr_values)) if mrr_values else None,
            "mean_ndcg_at_k": (sum(ndcg_values) / len(ndcg_values)) if ndcg_values else None,
        }

    return {
        "aggregate_by_k": aggregate_by_k,
        "per_query": per_query,
    }


def benchmark_dense_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    qrels_by_query: dict[str, QueryQrels],
    dense_retriever: DenseRetriever,
    k_values: Sequence[int] = (1, 3, 5, 10),
    query_batch_size: int = 64,
) -> dict[str, Any]:
    retrieval_rows = run_dense_retrieval(
        queries=queries,
        dense_retriever=dense_retriever,
        top_k=max(max(1, int(value)) for value in k_values),
        query_batch_size=query_batch_size,
    )
    evaluation = evaluate_retrieval_rows(
        retrieval_rows,
        qrels_by_query=qrels_by_query,
        k_values=k_values,
    )
    return {
        "retrieval_rows": retrieval_rows,
        **evaluation,
    }


def benchmark_hybrid_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    qrels_by_query: dict[str, QueryQrels],
    retrieval_engine: PageRetrievalEngine,
    k_values: Sequence[int] = (1, 3, 5, 10),
    query_batch_size: int = 64,
) -> dict[str, Any]:
    retrieval_rows = run_hybrid_retrieval(
        queries=queries,
        retrieval_engine=retrieval_engine,
        top_k=max(max(1, int(value)) for value in k_values),
        query_batch_size=query_batch_size,
    )
    evaluation = evaluate_retrieval_rows(
        retrieval_rows,
        qrels_by_query=qrels_by_query,
        k_values=k_values,
    )
    return {
        "retrieval_rows": retrieval_rows,
        **evaluation,
    }


def benchmark_hybrid_reranked_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    qrels_by_query: dict[str, QueryQrels],
    retrieval_engine: PageRetrievalEngine,
    reranker: Reranker,
    k_values: Sequence[int] = (1, 3, 5, 10),
    reranker_candidate_pool_size: int | None = None,
    query_batch_size: int = 64,
) -> dict[str, Any]:
    retrieval_rows = run_hybrid_reranked_retrieval(
        queries=queries,
        retrieval_engine=retrieval_engine,
        reranker=reranker,
        top_k=max(max(1, int(value)) for value in k_values),
        reranker_candidate_pool_size=reranker_candidate_pool_size,
        query_batch_size=query_batch_size,
    )
    evaluation = evaluate_retrieval_rows(
        retrieval_rows,
        qrels_by_query=qrels_by_query,
        k_values=k_values,
    )
    return {
        "retrieval_rows": retrieval_rows,
        **evaluation,
    }
