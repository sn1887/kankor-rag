from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rag_core.rag.citations import to_persian_digits
from rag_core.rag.pipeline import RAGPipeline
from rag_core.types import ChatTurn

from .contracts import ConversationStore, MediaProvider, OCRProvider, OutboundMessenger, WhatsAppJob

logger = logging.getLogger(__name__)


class WhatsAppMessageProcessor:
    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        messenger: OutboundMessenger,
        conversation_store: ConversationStore,
        media_provider: MediaProvider,
        ocr_provider: OCRProvider,
        max_reply_chars: int = 1400,
    ) -> None:
        self.pipeline = pipeline
        self.messenger = messenger
        self.conversation_store = conversation_store
        self.media_provider = media_provider
        self.ocr_provider = ocr_provider
        self.max_reply_chars = max(200, int(max_reply_chars))

    async def process(self, job: WhatsAppJob) -> None:
        message = job.message
        try:
            question = await self._resolve_question(message=message)
            if not question:
                await self.messenger.send_text(
                    to=message.from_wa_id,
                    text=(
                        "I could not read your question yet. Please send the question as text "
                        "or upload a clearer image."
                    ),
                    in_reply_to_message_id=message.message_id,
                )
                return

            history = list(self.conversation_store.get_history(message.from_wa_id))
            body, reference_sources = self._generate_answer_sync(question, history)
            if not body.strip():
                body = (
                    "I could not generate a full answer right now. "
                    "Please try again with a slightly more specific question."
                )

            chunks = _split_for_whatsapp(body, limit=self.max_reply_chars)
            for index, chunk in enumerate(chunks):
                await self.messenger.send_text(
                    to=message.from_wa_id,
                    text=chunk,
                    in_reply_to_message_id=message.message_id if index == 0 else None,
                )

            # Send references as a separate final message (WhatsApp-friendly, raw URLs).
            if reference_sources:
                refs_text = _render_whatsapp_references(reference_sources)
                if refs_text.strip():
                    refs_chunks = _split_for_whatsapp(refs_text, limit=self.max_reply_chars)
                    try:
                        for chunk in refs_chunks:
                            await self.messenger.send_text(
                                to=message.from_wa_id,
                                text=chunk,
                                in_reply_to_message_id=None,
                            )
                    except Exception:
                        logger.warning(
                            "Failed to send WhatsApp references for message %s; retrying once.",
                            message.message_id,
                            exc_info=True,
                        )
                        try:
                            for chunk in refs_chunks:
                                await self.messenger.send_text(
                                    to=message.from_wa_id,
                                    text=chunk,
                                    in_reply_to_message_id=None,
                                )
                        except Exception:
                            logger.warning(
                                "Second attempt to send WhatsApp references failed for message %s.",
                                message.message_id,
                                exc_info=True,
                            )
                            # Best-effort: notify the user without leaking long URLs into history.
                            try:
                                await self.messenger.send_text(
                                    to=message.from_wa_id,
                                    text="منابع ارسال نشد. لطفاً دوباره تلاش کنید.",
                                    in_reply_to_message_id=None,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to send WhatsApp references failure notice for message %s.",
                                    message.message_id,
                                )
            self.conversation_store.append_turn(
                message.from_wa_id,
                user_content=question,
                assistant_content=body,
            )
        except Exception:
            logger.exception("Failed to process WhatsApp message %s", message.message_id)
            try:
                await self.messenger.send_text(
                    to=message.from_wa_id,
                    text=(
                        "The service is temporarily busy. Please try again in a minute."
                    ),
                    in_reply_to_message_id=message.message_id,
                )
            except Exception:
                logger.exception(
                    "Failed to send WhatsApp fallback message %s",
                    message.message_id,
                )

    async def _resolve_question(self, *, message) -> str:
        if message.message_type == "text" and message.text is not None:
            return message.text.body.strip()

        if message.message_type == "image" and message.image is not None:
            caption = (message.image.caption or "").strip()
            extracted = ""
            try:
                media = await self.media_provider.fetch(
                    media_id=message.image.media_id,
                    hinted_mime_type=message.image.mime_type,
                )
                ocr = await self.ocr_provider.extract_text(media)
                extracted = ocr.text.strip()
            except Exception:
                logger.exception("Failed OCR path for message %s", message.message_id)

            if caption and extracted:
                return f"{caption}\n\nExtracted text:\n{extracted}"
            if caption:
                return caption
            if extracted:
                return extracted
            return ""

        return ""

    def _generate_answer_sync(self, question: str, history: Sequence[ChatTurn]) -> tuple[str, list[dict]]:
        deltas: list[str] = []
        reference_sources: list[dict] = []
        for event in self.pipeline.stream_answer(question=question, history=history):
            event_type = event.get("type")
            if event_type == "delta":
                text = str(event.get("data", {}).get("text", ""))
                if text:
                    deltas.append(text)
            elif event_type == "references":
                data = dict(event.get("data") or {})
                reference_sources = list(data.get("sources") or [])
        return "".join(deltas).strip(), reference_sources


def _split_for_whatsapp(text: str, *, limit: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return ["No answer was generated."]
    if len(normalized) <= limit:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    min_breakpoint = max(40, int(limit * 0.5))
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = max(remaining.rfind("\n", 0, limit), remaining.rfind(" ", 0, limit))
        if split_at < min_breakpoint:
            split_at = limit

        # Avoid splitting inside URLs when possible.
        url_match = None
        for match in _URL_PATTERN.finditer(remaining):
            if match.start() < split_at < match.end():
                url_match = match
        if url_match is not None and url_match.start() >= min_breakpoint:
            split_at = url_match.start()

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
        chunks.append(chunk)

        remaining = remaining[len(chunk):].lstrip()

    return chunks


_URL_PATTERN = re.compile(r"https?://\\S+")


def _render_whatsapp_references(sources: Sequence[dict]) -> str:
    # Compact plain-text Dari, WhatsApp-friendly.
    lines: list[str] = ["منابع:"]
    for idx, source in enumerate(sources, start=1):
        title = str(source.get("title") or source.get("sourceId") or "منبع نامشخص").strip()
        page = source.get("page")
        try:
            page_num = int(page) if page is not None else None
        except Exception:
            page_num = None
        if page_num is not None and page_num > 0:
            item = f"{to_persian_digits(idx)}. {title}، صفحه {to_persian_digits(page_num)}"
        else:
            item = f"{to_persian_digits(idx)}. {title}"
        lines.append(item)

        url = str(source.get("pdfUrl") or "").strip()
        if url:
            lines.append(url)
    return "\n".join(lines).strip()
