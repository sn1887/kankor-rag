from __future__ import annotations

import asyncio
import hashlib
import hmac

from kankor_api.whatsapp.queue import InMemoryJobQueue
from kankor_api.whatsapp.service import WhatsAppWebhookService
from kankor_api.whatsapp.stores import InMemoryProcessedMessageStore


def _build_text_payload(message_id: str) -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": message_id,
                                    "from": "93700000000",
                                    "timestamp": "1711000000",
                                    "type": "text",
                                    "text": {"body": "What is Kankor?"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_whatsapp_ingest_is_idempotent() -> None:
    queue = InMemoryJobQueue()
    service = WhatsAppWebhookService(
        queue=queue,
        processed_store=InMemoryProcessedMessageStore(),
    )

    first = asyncio.run(service.ingest_payload(_build_text_payload("wamid-1")))
    second = asyncio.run(service.ingest_payload(_build_text_payload("wamid-1")))

    assert first.accepted_messages == 1
    assert second.accepted_messages == 0
    assert second.duplicate_messages == 1
    assert queue.size() == 1


def test_whatsapp_signature_validation_uses_hmac_sha256() -> None:
    body = b'{"ok":true}'
    secret = "webhook-secret"
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    service = WhatsAppWebhookService(
        queue=InMemoryJobQueue(),
        processed_store=InMemoryProcessedMessageStore(),
        signature_secret=secret,
    )

    assert service.verify_signature(payload_bytes=body, header_signature=expected) is True
    assert service.verify_signature(payload_bytes=body, header_signature="sha256=bad") is False
    assert service.verify_signature(payload_bytes=body, header_signature=None) is False
