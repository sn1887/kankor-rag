from __future__ import annotations

from collections.abc import Sequence
import logging
import math
from typing import Any

from rag_core.contracts.reranker import Reranker
from rag_core.types import Document, Hit


logger = logging.getLogger(__name__)


class HFSequenceClassificationReranker(Reranker):
    def __init__(
        self,
        *,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 8,
        trust_remote_code: bool = True,
    ) -> None:
        normalized_model_name = str(model_name or "").strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")

        self.model_name = normalized_model_name
        self.max_length = max(32, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.trust_remote_code = bool(trust_remote_code)

        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        if self._torch is not None and self._tokenizer is not None and self._model is not None:
            return self._torch, self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "HFSequenceClassificationReranker requires 'torch' and 'transformers'. "
                "Install the serve extras before enabling RAG_RERANKER_ENABLED."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        model = model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        logger.info(
            "Loaded HF reranker model=%s device=%s max_length=%s batch_size=%s",
            self.model_name,
            device,
            self.max_length,
            self.batch_size,
        )
        return torch, tokenizer, model

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_value = math.exp(-value)
            return 1.0 / (1.0 + exp_value)
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _score_pairs(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        if not pairs:
            return []

        torch, tokenizer, model = self._load_runtime()
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = list(pairs[start : start + self.batch_size])
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(model.device)
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits.view(-1).detach().float().cpu().tolist()
                scores.extend(self._sigmoid(float(logit)) for logit in logits)
        return scores

    @staticmethod
    def _clone_hit_with_rerank_score(hit: Hit, *, rerank_score: float) -> Hit:
        metadata = dict(hit.document.metadata or {})
        metadata["rerank_score"] = float(rerank_score)
        document = Document(
            id=hit.document.id,
            text=hit.document.text,
            metadata=metadata,
        )
        return Hit(document=document, score=float(hit.score))

    def rerank(
        self,
        *,
        query: str,
        hits: Sequence[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        normalized_query = str(query or "").strip()
        if not normalized_query or not hits:
            return []

        limit = len(hits) if top_k is None else max(1, int(top_k))
        pairs = [(normalized_query, hit.document.text) for hit in hits]
        scores = self._score_pairs(pairs)
        if len(scores) != len(hits):
            raise RuntimeError(
                f"HF reranker score count mismatch: expected {len(hits)}, got {len(scores)}."
            )

        ranked_items: list[tuple[float, float, Hit]] = []
        for hit, rerank_score in zip(hits, scores, strict=False):
            ranked_items.append(
                (
                    float(rerank_score),
                    float(hit.score),
                    self._clone_hit_with_rerank_score(hit, rerank_score=float(rerank_score)),
                )
            )
        ranked_items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked_items[:limit]]
