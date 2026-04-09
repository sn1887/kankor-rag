from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
import re
import time
from typing import Any

from rag_core.rag.prompts import build_chat_messages, build_context_block
from rag_core.types import ChatAttachment, ChatTurn, Hit

logger = logging.getLogger(__name__)

_ADAPTIVE_ATTACHMENT_INTENTS = frozenset({"grounded_textbook", "practice_generation"})
DEFAULT_PDF_WINDOW_MAX_ATTACHMENTS = 3
DEFAULT_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS = 1
DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW = 0.42
DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW = 0.30
DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW = 0.003
DEFAULT_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS = 25
_QUESTION_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)
_COMPLEXITY_KEYWORDS = frozenset(
    {
        "مقایسه",
        "تفاوت",
        "فرق",
        "بین",
        "همچنین",
        "علاوه بر",
        "از یک طرف",
        "چند",
        "انواع",
        "دلایل",
        "compare",
        "difference",
    }
)


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list[ChatTurn]
    attachments: list[ChatAttachment] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdaptiveAttachmentRetrievalSignals:
    top_score_pre_expansion: float | None = None
    score_gap_pre_expansion: float | None = None
    top_score_post_expansion: float | None = None
    score_gap_post_expansion: float | None = None


class GroundingContextPlugin(ABC):
    @property
    def requires_attachments(self) -> bool:
        return False

    @abstractmethod
    def build(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        hits: Sequence[Hit],
        grounded: bool,
        intent: str,
        task_directive: str,
        corpus_version: str,
        retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None,
        supportive_grounding: bool = False,
    ) -> PreparedChatRequest:
        raise NotImplementedError


@dataclass(slots=True)
class AdaptiveAttachmentDecision:
    count: int
    target: int
    top_score: float | None
    score_gap: float | None
    unique_sources_top3: int
    question_complexity: bool
    reason_flags: list[str] = field(default_factory=list)


def _question_token_count(question: str) -> int:
    return len(_QUESTION_TOKEN_PATTERN.findall(question or ""))


def is_question_complex_for_pdf_attachments(
    question: str,
    *,
    length_threshold_tokens: int = 25,
) -> bool:
    normalized = " ".join(str(question or "").casefold().split())
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _COMPLEXITY_KEYWORDS):
        return True
    return _question_token_count(normalized) > max(1, int(length_threshold_tokens))


def _append_reason_flag(reason_flags: list[str], flag: str) -> None:
    if flag not in reason_flags:
        reason_flags.append(flag)


def _resolve_top_score(hits: Sequence[Hit]) -> float | None:
    if not hits:
        return None
    try:
        return float(hits[0].score)
    except (TypeError, ValueError):
        return None


def _resolve_score_gap(hits: Sequence[Hit]) -> float | None:
    if len(hits) < 2:
        return None
    try:
        return float(hits[0].score) - float(hits[1].score)
    except (TypeError, ValueError):
        return None


def _resolve_unique_sources_top3(hits: Sequence[Hit]) -> int:
    source_ids = {
        str(hit.document.metadata.get("source_id", "")).strip()
        for hit in hits[:3]
        if str(hit.document.metadata.get("source_id", "")).strip()
    }
    return len(source_ids)


def _coerce_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return high
    return max(low, min(value, high))


def select_adaptive_pdf_attachment_count(
    *,
    question: str,
    hits: Sequence[Hit],
    effective_max: int,
    max_attachments: int,
    min_attachments: int,
    adaptive_enabled: bool,
    adaptive_applicable: bool,
    top_score_low: float,
    top_score_very_low: float,
    score_gap_low: float,
    complexity_length_tokens: int,
    top_score_override: float | None = None,
    score_gap_override: float | None = None,
) -> AdaptiveAttachmentDecision:
    resolved_max = max(1, int(max_attachments))
    resolved_min = max(1, int(min_attachments))
    resolved_effective_max = max(0, min(int(effective_max), resolved_max))

    derived_top_score = _resolve_top_score(hits)
    derived_score_gap = _resolve_score_gap(hits)
    top_score = _coerce_optional_float(top_score_override)
    if top_score is None:
        top_score = derived_top_score
    score_gap = _coerce_optional_float(score_gap_override)
    if score_gap is None:
        score_gap = derived_score_gap
    unique_sources_top3 = _resolve_unique_sources_top3(hits)
    question_complexity = is_question_complex_for_pdf_attachments(
        question,
        length_threshold_tokens=complexity_length_tokens,
    )

    if not hits or resolved_effective_max == 0:
        return AdaptiveAttachmentDecision(
            count=0,
            target=0,
            top_score=top_score,
            score_gap=score_gap,
            unique_sources_top3=unique_sources_top3,
            question_complexity=question_complexity,
            reason_flags=[],
        )

    reason_flags: list[str] = []
    if not adaptive_enabled or not adaptive_applicable:
        target = resolved_max
        count = min(target, resolved_effective_max)
        _append_reason_flag(reason_flags, "adaptive_disabled")
        if count < target:
            _append_reason_flag(reason_flags, "clamped_by_effective_max")
        return AdaptiveAttachmentDecision(
            count=count,
            target=target,
            top_score=top_score,
            score_gap=score_gap,
            unique_sources_top3=unique_sources_top3,
            question_complexity=question_complexity,
            reason_flags=reason_flags,
        )

    target = 1
    if top_score is not None and top_score < float(top_score_low):
        target = max(target, 2)
        _append_reason_flag(reason_flags, "low_top_score")
    if score_gap is not None and score_gap < float(score_gap_low):
        target = max(target, 2)
        _append_reason_flag(reason_flags, "flat_score_gap")
    if unique_sources_top3 >= 2:
        target = max(target, 2)
        _append_reason_flag(reason_flags, "multi_source_top3")
    if top_score is not None and top_score < float(top_score_very_low):
        target = max(target, 3)
        _append_reason_flag(reason_flags, "very_low_top_score")
    if unique_sources_top3 >= 3:
        target = max(target, 3)
    if question_complexity and unique_sources_top3 >= 2:
        target = max(target, 3)
        _append_reason_flag(reason_flags, "complex_query_with_source_spread")

    count = _clamp(target, resolved_min, resolved_effective_max)
    if count < target:
        _append_reason_flag(reason_flags, "clamped_by_effective_max")

    return AdaptiveAttachmentDecision(
        count=count,
        target=target,
        top_score=top_score,
        score_gap=score_gap,
        unique_sources_top3=unique_sources_top3,
        question_complexity=question_complexity,
        reason_flags=reason_flags,
    )


def _runtime_hints_block(
    *,
    task_directive: str,
    corpus_version: str,
    hit_count: int,
    supportive_grounding: bool = False,
) -> str:
    runtime_hints: list[str] = []
    if corpus_version.strip():
        runtime_hints.append(f"- corpus_version: {corpus_version.strip()}")
    runtime_hints.append(f"- retrieved_sources: {max(0, int(hit_count))}")
    runtime_hints.append(
        f"- grounding_mode: {'supportive_stem' if supportive_grounding else 'strict_grounded'}"
    )
    if task_directive.strip():
        runtime_hints.append(f"- task_mode: {task_directive.strip()}")
    block = "\n".join(runtime_hints)
    if not block:
        return ""
    return f"پیکربندی اجرای همین درخواست:\n{block}\n\n"


class TextGroundingContextPlugin(GroundingContextPlugin):
    def build(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        hits: Sequence[Hit],
        grounded: bool,
        intent: str,
        task_directive: str,
        corpus_version: str,
        retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None,
        supportive_grounding: bool = False,
    ) -> PreparedChatRequest:
        _ = retrieval_signals
        context_block = build_context_block(hits) if grounded else ""
        messages = build_chat_messages(
            question=question,
            history=history,
            context_block=context_block,
            grounded=grounded,
            intent=intent,
            task_directive=task_directive,
            corpus_version=corpus_version,
            hit_count=len(hits) if grounded else None,
            supportive_grounding=supportive_grounding,
        )
        return PreparedChatRequest(messages=messages)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _build_pdf_outline(hits: Sequence[tuple[str, Hit]]) -> str:
    lines: list[str] = []
    for label, hit in hits:
        meta = hit.document.metadata
        start_page = meta.get("start_page", meta.get("page", "unknown"))
        end_page = meta.get("end_page", meta.get("page", "unknown"))
        lines.append(
            f"[{label}] source_id={meta.get('source_id', 'unknown')} "
            f"title={meta.get('title', 'unknown')} "
            f"start_page={start_page} "
            f"end_page={end_page} "
            f"subject={meta.get('subject', 'unknown')} "
            f"grade={meta.get('grade_band', 'mixed')}"
        )
    return "\n".join(lines).strip()


class PdfWindowGroundingContextPlugin(GroundingContextPlugin):
    def __init__(
        self,
        *,
        max_attachments: int = DEFAULT_PDF_WINDOW_MAX_ATTACHMENTS,
        max_pages_per_attachment: int = 4,
        adaptive_enabled: bool = False,
        adaptive_min_attachments: int = DEFAULT_PDF_WINDOW_ADAPTIVE_MIN_ATTACHMENTS,
        adaptive_top_score_low: float = DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_LOW,
        adaptive_top_score_very_low: float = DEFAULT_PDF_WINDOW_ADAPTIVE_TOP_SCORE_VERY_LOW,
        adaptive_score_gap_low: float = DEFAULT_PDF_WINDOW_ADAPTIVE_SCORE_GAP_LOW,
        adaptive_complexity_length_tokens: int = DEFAULT_PDF_WINDOW_ADAPTIVE_COMPLEXITY_LENGTH_TOKENS,
    ) -> None:
        self.max_attachments = max(1, int(max_attachments))
        self.max_pages_per_attachment = max(1, int(max_pages_per_attachment))
        self.adaptive_enabled = bool(adaptive_enabled)
        self.adaptive_min_attachments = max(1, int(adaptive_min_attachments))
        self.adaptive_top_score_low = float(adaptive_top_score_low)
        self.adaptive_top_score_very_low = min(float(adaptive_top_score_very_low), self.adaptive_top_score_low)
        self.adaptive_score_gap_low = max(0.0, float(adaptive_score_gap_low))
        self.adaptive_complexity_length_tokens = max(1, int(adaptive_complexity_length_tokens))

    @property
    def requires_attachments(self) -> bool:
        return True

    def _extract_pdf_window(
        self,
        *,
        reader_cache: dict[Path, Any],
        pdf_path: Path,
        start_page: int,
        end_page: int,
    ) -> bytes | None:
        try:
            from pypdf import PdfReader, PdfWriter
        except Exception:
            return None

        try:
            reader = reader_cache.get(pdf_path)
            if reader is None:
                reader = PdfReader(str(pdf_path))
                reader_cache[pdf_path] = reader
            total_pages = len(reader.pages)
            if start_page < 1 or end_page < start_page or end_page > total_pages:
                return None
            writer = PdfWriter()
            for page_idx in range(start_page - 1, end_page):
                writer.add_page(reader.pages[page_idx])
            buffer = io.BytesIO()
            writer.write(buffer)
            return buffer.getvalue()
        except Exception:
            return None

    @staticmethod
    def _adaptive_applicable_for_intent(intent: str) -> bool:
        return str(intent or "").strip().lower() in _ADAPTIVE_ATTACHMENT_INTENTS

    def _attachments_for_hits(
        self,
        *,
        question: str,
        intent: str,
        hits: Sequence[Hit],
        retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None,
    ) -> tuple[list[tuple[str, Hit]], list[ChatAttachment]]:
        attachments: list[ChatAttachment] = []
        attached_hits: list[tuple[str, Hit]] = []
        candidates: list[tuple[int, Hit, Path, int, int]] = []
        reader_cache: dict[Path, Any] = {}
        for idx, hit in enumerate(hits, start=1):
            meta = hit.document.metadata
            path_value = str(meta.get("source_pdf_path", "")).strip()
            if not path_value:
                continue
            pdf_path = Path(path_value)
            if not pdf_path.exists():
                continue
            start_page = _positive_int(meta.get("start_page", meta.get("page")))
            span_end_page = _positive_int(meta.get("end_page", meta.get("page")))
            end_page = span_end_page
            if start_page is None or end_page is None:
                continue
            end_page = min(end_page, start_page + self.max_pages_per_attachment - 1)
            candidates.append((idx, hit, pdf_path, start_page, end_page))

        decision = select_adaptive_pdf_attachment_count(
            question=question,
            hits=hits,
            effective_max=len(candidates),
            max_attachments=self.max_attachments,
            min_attachments=self.adaptive_min_attachments,
            adaptive_enabled=self.adaptive_enabled,
            adaptive_applicable=self._adaptive_applicable_for_intent(intent),
            top_score_low=self.adaptive_top_score_low,
            top_score_very_low=self.adaptive_top_score_very_low,
            score_gap_low=self.adaptive_score_gap_low,
            complexity_length_tokens=self.adaptive_complexity_length_tokens,
            top_score_override=(
                retrieval_signals.top_score_pre_expansion
                if retrieval_signals is not None
                else None
            ),
            score_gap_override=(
                retrieval_signals.score_gap_pre_expansion
                if retrieval_signals is not None
                else None
            ),
        )
        if decision.top_score is not None and decision.top_score > 1.0:
            logger.warning(
                "Adaptive PDF attachment thresholds are tuned for normalized retrieval scores; "
                "top_score %.4f exceeded 1.0. Recalibration may be required for this retriever/index.",
                decision.top_score,
            )
        logger.debug(
            "adaptive_pdf_window_attachment_decision %s",
            {
                "adaptive_attachment_count": decision.count,
                "intent": str(intent),
                "top_score": decision.top_score,
                "score_gap": decision.score_gap,
                "top_score_pre_expansion": (
                    retrieval_signals.top_score_pre_expansion
                    if retrieval_signals is not None
                    else None
                ),
                "score_gap_pre_expansion": (
                    retrieval_signals.score_gap_pre_expansion
                    if retrieval_signals is not None
                    else None
                ),
                "top_score_post_expansion": (
                    retrieval_signals.top_score_post_expansion
                    if retrieval_signals is not None
                    else _resolve_top_score(hits)
                ),
                "score_gap_post_expansion": (
                    retrieval_signals.score_gap_post_expansion
                    if retrieval_signals is not None
                    else _resolve_score_gap(hits)
                ),
                "gap_signal_source": (
                    "pre_expansion"
                    if retrieval_signals is not None and retrieval_signals.score_gap_pre_expansion is not None
                    else "post_expansion"
                ),
                "unique_sources_top3": decision.unique_sources_top3,
                "question_complexity": decision.question_complexity,
                "reason_flags": list(decision.reason_flags),
            },
        )

        if decision.count <= 0:
            return attached_hits, attachments

        for idx, hit, pdf_path, start_page, end_page in candidates:
            if len(attachments) >= decision.count:
                break
            meta = hit.document.metadata
            bytes_data = self._extract_pdf_window(
                reader_cache=reader_cache,
                pdf_path=pdf_path,
                start_page=start_page,
                end_page=end_page,
            )
            if not bytes_data:
                continue
            label = f"S{idx}"
            attachments.append(
                ChatAttachment(
                    media_type="application/pdf",
                    data=bytes_data,
                    label=label,
                    metadata={
                        "source_id": str(meta.get("source_id", "")),
                        "start_page": start_page,
                        "end_page": end_page,
                    },
                )
            )
            attached_hits.append((label, hit))
        return attached_hits, attachments

    def build(
        self,
        *,
        question: str,
        history: Sequence[ChatTurn],
        hits: Sequence[Hit],
        grounded: bool,
        intent: str,
        task_directive: str,
        corpus_version: str,
        retrieval_signals: AdaptiveAttachmentRetrievalSignals | None = None,
        supportive_grounding: bool = False,
    ) -> PreparedChatRequest:
        if not grounded:
            messages = build_chat_messages(
                question=question,
                history=history,
                context_block="",
                grounded=False,
                intent=intent,
                task_directive=task_directive,
                corpus_version=corpus_version,
                hit_count=None,
                supportive_grounding=supportive_grounding,
            )
            return PreparedChatRequest(messages=messages)

        attachment_started = time.perf_counter()
        attached_hits, attachments = self._attachments_for_hits(
            question=question,
            intent=intent,
            hits=hits,
            retrieval_signals=retrieval_signals,
        )
        attachment_prep_ms = int((time.perf_counter() - attachment_started) * 1000)
        attachment_diagnostics = {
            "attachment_prep_ms": attachment_prep_ms,
            "attachment_count": len(attachments),
            "attachment_bytes_total": sum(len(attachment.data) for attachment in attachments),
            "attachment_fallback_to_text": False,
        }
        if not attachments:
            request = TextGroundingContextPlugin().build(
                question=question,
                history=history,
                hits=hits,
                grounded=grounded,
                intent=intent,
                task_directive=task_directive,
                corpus_version=corpus_version,
                retrieval_signals=retrieval_signals,
                supportive_grounding=supportive_grounding,
            )
            attachment_diagnostics["attachment_fallback_to_text"] = True
            request.diagnostics.update(attachment_diagnostics)
            return request

        context_outline = _build_pdf_outline(attached_hits)
        runtime_prefix = _runtime_hints_block(
            task_directive=task_directive,
            corpus_version=corpus_version,
            hit_count=len(hits),
            supportive_grounding=supportive_grounding,
        )
        grounding_instruction = (
            "برای ادعاهای factual فقط از PDFهای ضمیمه‌شده استفاده کن. "
            "اگر شواهد کافی نبود، محدودیت را صریح بگو. "
        )
        if supportive_grounding:
            grounding_instruction = (
                "اگر PDFهای ضمیمه‌شده برای فرمول، تعریف یا مثال مفید بودند از آن‌ها استفاده کن. "
                "اگر PDFها دقیقاً همان سوال را پوشش نمی‌دادند، مسئله را مستقیم حل کن و فقط ادعاهای "
                "مبتنی بر PDF را به عنوان شواهد کتابی در نظر بگیر. "
            )
        user_prompt = (
            f"پنجره‌های PDF ضمیمه‌شده:\n{context_outline}\n\n"
            f"پرسش کاربر: {question}\n\n"
            f"{runtime_prefix}"
            f"{grounding_instruction}"
            "برای ادعاهای مستند از ارجاع کوتاه درون‌متنی مانند [۱] استفاده کن. "
            "از قالب‌های فنی مانند [S1] استفاده نکن و بخش «منابع/References» هم نساز. "
            "پاسخ را Markdown و آموزشی نگه دار."
        )
        messages = list(history)
        messages.append(ChatTurn(role="user", content=user_prompt))
        return PreparedChatRequest(
            messages=messages,
            attachments=attachments,
            diagnostics=attachment_diagnostics,
        )
