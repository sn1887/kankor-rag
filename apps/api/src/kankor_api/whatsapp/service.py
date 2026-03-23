from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import JobQueue, ProcessedMessageStore, WhatsAppJob
from .events import parse_inbound_messages


@dataclass(slots=True, frozen=True)
class WebhookIngestResult:
    received_messages: int
    accepted_messages: int
    duplicate_messages: int
    unsupported_messages: int

    def as_dict(self) -> dict[str, int]:
        return {
            "received_messages": self.received_messages,
            "accepted_messages": self.accepted_messages,
            "duplicate_messages": self.duplicate_messages,
            "unsupported_messages": self.unsupported_messages,
        }


class WhatsAppWebhookService:
    def __init__(
        self,
        *,
        queue: JobQueue,
        processed_store: ProcessedMessageStore,
        signature_secret: str | None = None,
    ) -> None:
        self.queue = queue
        self.processed_store = processed_store
        self.signature_secret = (signature_secret or "").strip() or None

    def verify_signature(self, *, payload_bytes: bytes, header_signature: str | None) -> bool:
        if not self.signature_secret:
            return True
        if not header_signature:
            return False

        provided = header_signature.strip()
        if not provided.startswith("sha256="):
            return False

        expected_hash = hmac.new(
            self.signature_secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        expected_signature = f"sha256={expected_hash}"
        return hmac.compare_digest(provided, expected_signature)

    async def ingest_payload(self, payload: Mapping[str, Any]) -> WebhookIngestResult:
        parsed_messages = parse_inbound_messages(payload)
        accepted = 0
        duplicates = 0
        unsupported = 0

        for message in parsed_messages:
            if not message.is_supported:
                unsupported += 1
            if not self.processed_store.mark_if_new(message.message_id):
                duplicates += 1
                continue
            await self.queue.enqueue(WhatsAppJob(message=message))
            accepted += 1

        return WebhookIngestResult(
            received_messages=len(parsed_messages),
            accepted_messages=accepted,
            duplicate_messages=duplicates,
            unsupported_messages=unsupported,
        )
