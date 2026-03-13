from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from rag_core.types import ChatTurn

if TYPE_CHECKING:
    from .events import WhatsAppInboundMessage


@dataclass(slots=True, frozen=True)
class WhatsAppJob:
    message: WhatsAppInboundMessage
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True, frozen=True)
class MediaBlob:
    content: bytes
    mime_type: str | None = None
    sha256: str | None = None


@dataclass(slots=True, frozen=True)
class OCRResult:
    text: str
    confidence: float | None = None
    engine: str = "unknown"


class JobQueue(ABC):
    @abstractmethod
    async def enqueue(self, job: WhatsAppJob) -> None:
        raise NotImplementedError

    @abstractmethod
    async def dequeue(self, *, timeout_seconds: float = 1.0) -> WhatsAppJob | None:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class ProcessedMessageStore(ABC):
    @abstractmethod
    def mark_if_new(self, message_id: str) -> bool:
        raise NotImplementedError


class ConversationStore(ABC):
    @abstractmethod
    def get_history(self, wa_id: str) -> Sequence[ChatTurn]:
        raise NotImplementedError

    @abstractmethod
    def append_turn(self, wa_id: str, *, user_content: str, assistant_content: str) -> None:
        raise NotImplementedError


class OutboundMessenger(ABC):
    @abstractmethod
    async def send_text(
        self,
        *,
        to: str,
        text: str,
        in_reply_to_message_id: str | None = None,
    ) -> None:
        raise NotImplementedError


class MediaProvider(ABC):
    @abstractmethod
    async def fetch(self, *, media_id: str, hinted_mime_type: str | None = None) -> MediaBlob:
        raise NotImplementedError


class OCRProvider(ABC):
    @abstractmethod
    async def extract_text(self, media: MediaBlob) -> OCRResult:
        raise NotImplementedError
