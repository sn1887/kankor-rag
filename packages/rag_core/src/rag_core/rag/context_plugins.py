from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any

from rag_core.rag.prompts import build_chat_messages, build_context_block
from rag_core.types import ChatAttachment, ChatTurn, Hit


@dataclass(slots=True)
class PreparedChatRequest:
    messages: list[ChatTurn]
    attachments: list[ChatAttachment] = field(default_factory=list)


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
    ) -> PreparedChatRequest:
        raise NotImplementedError


def _runtime_hints_block(
    *,
    task_directive: str,
    corpus_version: str,
    hit_count: int,
) -> str:
    runtime_hints: list[str] = []
    if corpus_version.strip():
        runtime_hints.append(f"- corpus_version: {corpus_version.strip()}")
    runtime_hints.append(f"- retrieved_sources: {max(0, int(hit_count))}")
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
    ) -> PreparedChatRequest:
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
        max_attachments: int = 3,
        max_pages_per_attachment: int = 4,
    ) -> None:
        self.max_attachments = max(1, int(max_attachments))
        self.max_pages_per_attachment = max(1, int(max_pages_per_attachment))

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

    def _attachments_for_hits(self, hits: Sequence[Hit]) -> tuple[list[tuple[str, Hit]], list[ChatAttachment]]:
        attachments: list[ChatAttachment] = []
        attached_hits: list[tuple[str, Hit]] = []
        reader_cache: dict[Path, Any] = {}
        for idx, hit in enumerate(hits, start=1):
            if len(attachments) >= self.max_attachments:
                break
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
            )
            return PreparedChatRequest(messages=messages)

        attached_hits, attachments = self._attachments_for_hits(hits)
        if not attachments:
            return TextGroundingContextPlugin().build(
                question=question,
                history=history,
                hits=hits,
                grounded=grounded,
                intent=intent,
                task_directive=task_directive,
                corpus_version=corpus_version,
            )

        context_outline = _build_pdf_outline(attached_hits)
        runtime_prefix = _runtime_hints_block(
            task_directive=task_directive,
            corpus_version=corpus_version,
            hit_count=len(hits),
        )
        user_prompt = (
            f"پنجره‌های PDF ضمیمه‌شده:\n{context_outline}\n\n"
            f"پرسش کاربر: {question}\n\n"
            f"{runtime_prefix}"
            "برای ادعاهای factual فقط از PDFهای ضمیمه‌شده استفاده کن. "
            "در متن پاسخ هیچ ارجاع درون‌متنی مانند [S1] تولید نکن و بخش «منابع/References» هم نساز. "
            "پاسخ را Markdown و آموزشی نگه دار."
        )
        messages = list(history)
        messages.append(ChatTurn(role="user", content=user_prompt))
        return PreparedChatRequest(
            messages=messages,
            attachments=attachments,
        )
