from __future__ import annotations

import asyncio

from kankor_api.whatsapp.contracts import MediaBlob, OCRResult, WhatsAppJob
from kankor_api.whatsapp.events import (
    WhatsAppImagePayload,
    WhatsAppInboundMessage,
    WhatsAppTextPayload,
)
from kankor_api.whatsapp.messaging import LoggingMessenger
from kankor_api.whatsapp.processor import WhatsAppMessageProcessor
from kankor_api.whatsapp.stores import InMemoryConversationStore


class _FakePipeline:
    def stream_answer(self, *, question: str, history):
        yield {"type": "sources", "data": [{"badge": "S1"}]}
        yield {"type": "delta", "data": {"text": f"answer:{question}"}}


class _FakeMediaProvider:
    async def fetch(self, *, media_id: str, hinted_mime_type: str | None = None) -> MediaBlob:
        _ = media_id, hinted_mime_type
        return MediaBlob(content=b"fake-bytes", mime_type="image/jpeg")


class _FakeOCRProvider:
    async def extract_text(self, media: MediaBlob) -> OCRResult:
        _ = media
        return OCRResult(text="What is the derivative of x^2?", confidence=0.9, engine="fake")


def test_whatsapp_processor_handles_text_message() -> None:
    messenger = LoggingMessenger()
    store = InMemoryConversationStore(max_turns=4)
    processor = WhatsAppMessageProcessor(
        pipeline=_FakePipeline(),  # type: ignore[arg-type]
        messenger=messenger,
        conversation_store=store,
        media_provider=_FakeMediaProvider(),  # type: ignore[arg-type]
        ocr_provider=_FakeOCRProvider(),  # type: ignore[arg-type]
    )

    job = WhatsAppJob(
        message=WhatsAppInboundMessage(
            message_id="wamid-1",
            from_wa_id="93700111222",
            timestamp=1711000000,
            message_type="text",
            text=WhatsAppTextPayload(body="Define photosynthesis"),
        )
    )

    asyncio.run(processor.process(job))

    assert len(messenger.sent_messages) == 1
    assert messenger.sent_messages[0].text == "answer:Define photosynthesis"
    history = store.get_history("93700111222")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Define photosynthesis"


def test_whatsapp_processor_handles_image_ocr_path() -> None:
    messenger = LoggingMessenger()
    store = InMemoryConversationStore(max_turns=4)
    processor = WhatsAppMessageProcessor(
        pipeline=_FakePipeline(),  # type: ignore[arg-type]
        messenger=messenger,
        conversation_store=store,
        media_provider=_FakeMediaProvider(),  # type: ignore[arg-type]
        ocr_provider=_FakeOCRProvider(),  # type: ignore[arg-type]
    )

    job = WhatsAppJob(
        message=WhatsAppInboundMessage(
            message_id="wamid-2",
            from_wa_id="93700888999",
            timestamp=1711000010,
            message_type="image",
            image=WhatsAppImagePayload(media_id="media-1", mime_type="image/jpeg"),
        )
    )

    asyncio.run(processor.process(job))

    assert len(messenger.sent_messages) == 1
    assert "derivative" in messenger.sent_messages[0].text
    history = store.get_history("93700888999")
    assert len(history) == 2
    assert "derivative" in history[0].content
