from __future__ import annotations

from collections.abc import Sequence
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from rag_core.contracts.reranker import Reranker
from rag_core.types import Document, Hit


logger = logging.getLogger(__name__)


class ONNXSequenceClassificationReranker(Reranker):
    _MODEL_FILENAME_CANDIDATES = (
        "onnx/model_int8.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_uint8.onnx",
        "onnx/model_fp16.onnx",
        "onnx/model.onnx",
    )

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str | None = None,
        max_length: int = 256,
        batch_size: int = 8,
    ) -> None:
        normalized_model_name = str(model_name or "").strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")

        self.model_name = normalized_model_name
        self.model_revision = str(model_revision or "").strip() or None
        self.max_length = max(32, int(max_length))
        self.batch_size = max(1, int(batch_size))

        self._onnxruntime: Any | None = None
        self._tokenizer: Any | None = None
        self._session: Any | None = None
        self._session_input_names: tuple[str, ...] = ()
        self._session_output_name: str | None = None
        self._model_path: str | None = None

        self.last_runtime_load_ms = 0
        self.last_call_load_ms = 0
        self.last_call_inference_ms = 0
        self.last_call_total_ms = 0

    def _resolve_model_path(self) -> str:
        if self._model_path is not None:
            return self._model_path

        cached_path = self._resolve_cached_model_path()
        if cached_path is not None:
            self._model_path = cached_path
            logger.info(
                "Resolved ONNX reranker model from local cache model=%s revision=%s path=%s",
                self.model_name,
                self.model_revision or "main",
                cached_path,
            )
            return self._model_path

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "ONNXSequenceClassificationReranker requires 'huggingface_hub'. "
                "Install the serve extras before enabling the ONNX reranker."
            ) from exc

        for filename in self._MODEL_FILENAME_CANDIDATES:
            try:
                self._model_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=filename,
                    revision=self.model_revision,
                    repo_type="model",
                )
                logger.info(
                    "Resolved ONNX reranker model=%s revision=%s filename=%s",
                    self.model_name,
                    self.model_revision or "main",
                    filename,
                )
                return self._model_path
            except EntryNotFoundError:
                continue
        raise RuntimeError(
            "Unable to locate an ONNX model file in repo "
            f"{self.model_name!r} revision={self.model_revision or 'main'!r}. "
            f"Tried: {', '.join(self._MODEL_FILENAME_CANDIDATES)}."
        )

    def _resolve_cached_model_path(self) -> str | None:
        cache_root = Path(
            os.getenv("HF_HOME")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or (Path.home() / ".cache" / "huggingface")
        )
        repo_slug = self.model_name.replace("/", "--")
        if self.model_revision:
            snapshot_dir = cache_root / "hub" / f"models--{repo_slug}" / "snapshots" / self.model_revision
            for filename in self._MODEL_FILENAME_CANDIDATES:
                candidate = snapshot_dir / filename
                if candidate.exists():
                    return str(candidate)
        return None

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        if self._onnxruntime is not None and self._tokenizer is not None and self._session is not None:
            self.last_runtime_load_ms = 0
            return self._onnxruntime, self._tokenizer, self._session

        load_started = time.perf_counter()
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "ONNXSequenceClassificationReranker requires 'onnxruntime' and 'transformers'. "
                "Install the serve extras before enabling the ONNX reranker."
            ) from exc

        model_path = self._resolve_model_path()
        tokenizer = self._load_tokenizer(
            AutoTokenizer=AutoTokenizer,
            model_path=model_path,
        )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self._onnxruntime = ort
        self._tokenizer = tokenizer
        self._session = session
        self._session_input_names = tuple(meta.name for meta in session.get_inputs())
        first_output = session.get_outputs()[0] if session.get_outputs() else None
        self._session_output_name = first_output.name if first_output is not None else None
        self.last_runtime_load_ms = int((time.perf_counter() - load_started) * 1000)
        logger.info(
            "Loaded ONNX reranker model=%s revision=%s max_length=%s batch_size=%s inputs=%s",
            self.model_name,
            self.model_revision or "main",
            self.max_length,
            self.batch_size,
            list(self._session_input_names),
        )
        return ort, tokenizer, session

    def _tokenizer_sources(self, *, model_path: str) -> list[str]:
        sources: list[str] = []
        snapshot_root = str(Path(model_path).parents[1])
        for source in (snapshot_root, self.model_name):
            normalized = str(source or "").strip()
            if normalized and normalized not in sources:
                sources.append(normalized)
        return sources

    def _tokenizer_source_options(self, *, source: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "trust_remote_code": False,
        }
        if source == self.model_name:
            if self.model_revision:
                options["revision"] = self.model_revision
        else:
            options["local_files_only"] = True
        return options

    def _load_tokenizer_from_source(self, *, AutoTokenizer: Any, source: str, use_fast: bool) -> Any:
        return AutoTokenizer.from_pretrained(
            source,
            use_fast=use_fast,
            **self._tokenizer_source_options(source=source),
        )

    def _load_tokenizer(self, *, AutoTokenizer: Any, model_path: str) -> Any:
        sources = self._tokenizer_sources(model_path=model_path)
        attempts: list[str] = []
        last_error: Exception | None = None

        for source in sources:
            try:
                tokenizer = self._load_tokenizer_from_source(
                    AutoTokenizer=AutoTokenizer,
                    source=source,
                    use_fast=True,
                )
                logger.info(
                    "Loaded reranker tokenizer model=%s revision=%s source=%s use_fast=true",
                    self.model_name,
                    self.model_revision or "main",
                    source,
                )
                return tokenizer
            except Exception as exc:
                last_error = exc
                attempts.append(f"{source} (use_fast=True): {exc}")
                logger.warning(
                    "Fast tokenizer load failed for reranker model=%s revision=%s source=%s; trying slow tokenizer fallback",
                    self.model_name,
                    self.model_revision or "main",
                    source,
                )
            try:
                tokenizer = self._load_tokenizer_from_source(
                    AutoTokenizer=AutoTokenizer,
                    source=source,
                    use_fast=False,
                )
                logger.info(
                    "Loaded reranker tokenizer model=%s revision=%s source=%s use_fast=false",
                    self.model_name,
                    self.model_revision or "main",
                    source,
                )
                return tokenizer
            except Exception as exc:
                last_error = exc
                attempts.append(f"{source} (use_fast=False): {exc}")

        detail = "; ".join(attempts) if attempts else "no tokenizer sources were available"
        raise RuntimeError(
            "Unable to load tokenizer for ONNX reranker "
            f"model={self.model_name!r} revision={self.model_revision or 'main'!r}. "
            f"Attempted sources: {', '.join(sources)}. "
            f"Failures: {detail}"
        ) from last_error

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

        _, tokenizer, session = self._load_runtime()
        if self._session_output_name is None:
            raise RuntimeError("ONNX reranker session did not expose an output tensor.")

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = list(pairs[start : start + self.batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=self.max_length,
            )
            ort_inputs: dict[str, np.ndarray] = {}
            for name in self._session_input_names:
                value = encoded.get(name)
                if value is None:
                    value = self._default_missing_input(name=name, encoded=encoded)
                if value is None:
                    continue
                value = np.asarray(value)
                if value.dtype.kind in {"i", "u", "b"}:
                    value = value.astype(np.int64, copy=False)
                ort_inputs[name] = value
            if not ort_inputs:
                raise RuntimeError(
                    f"ONNX reranker inputs were empty for session inputs={self._session_input_names!r}."
                )
            outputs = session.run([self._session_output_name], ort_inputs)
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1).tolist()
            scores.extend(self._sigmoid(float(logit)) for logit in logits)
        return scores

    @staticmethod
    def _default_missing_input(*, name: str, encoded: Any) -> np.ndarray | None:
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            return None

        input_ids_array = np.asarray(input_ids)
        if name == "token_type_ids":
            return np.zeros_like(input_ids_array, dtype=np.int64)
        if name == "position_ids":
            if input_ids_array.ndim == 1:
                return np.arange(input_ids_array.shape[0], dtype=np.int64)
            seq_len = int(input_ids_array.shape[1])
            base = np.arange(seq_len, dtype=np.int64)
            return np.broadcast_to(base, input_ids_array.shape).copy()
        return None

    def warmup(self) -> None:
        self._score_pairs([("warmup", "warmup")])

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

        runtime_loaded_before = self._onnxruntime is not None and self._tokenizer is not None and self._session is not None
        call_started = time.perf_counter()
        limit = len(hits) if top_k is None else max(1, int(top_k))
        pairs = [(normalized_query, hit.document.text) for hit in hits]
        scores = self._score_pairs(pairs)
        self.last_call_total_ms = int((time.perf_counter() - call_started) * 1000)
        self.last_call_load_ms = 0 if runtime_loaded_before else int(self.last_runtime_load_ms)
        self.last_call_inference_ms = max(0, self.last_call_total_ms - self.last_call_load_ms)
        if len(scores) != len(hits):
            raise RuntimeError(
                f"ONNX reranker score count mismatch: expected {len(hits)}, got {len(scores)}."
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
