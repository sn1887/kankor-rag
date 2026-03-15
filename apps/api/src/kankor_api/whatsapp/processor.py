from __future__ import annotations

import logging
from collections.abc import Sequence

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
            answer = self._generate_answer_sync(question, history)
            if not answer.strip():
                answer = (
                    "I could not generate a full answer right now. "
                    "Please try again with a slightly more specific question."
                )

            chunks = _split_for_whatsapp(answer, limit=self.max_reply_chars)
            for index, chunk in enumerate(chunks):
                await self.messenger.send_text(
                    to=message.from_wa_id,
                    text=chunk,
                    in_reply_to_message_id=message.message_id if index == 0 else None,
                )
            self.conversation_store.append_turn(
                message.from_wa_id,
                user_content=question,
                assistant_content=answer,
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

    def _generate_answer_sync(self, question: str, history: Sequence[ChatTurn]) -> str:
        deltas: list[str] = []
        for event in self.pipeline.stream_answer(question=question, history=history):
            if event.get("type") != "delta":
                continue
            text = str(event.get("data", {}).get("text", ""))
            if text:
                deltas.append(text)
        return "".join(deltas).strip()


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

        split_at = max(
            remaining.rfind("\n", 0, limit),
            remaining.rfind(" ", 0, limit),
        )
        if split_at < min_breakpoint:
            split_at = limit

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
        chunks.append(chunk)

        remaining = remaining[len(chunk):].lstrip()

    return chunks
