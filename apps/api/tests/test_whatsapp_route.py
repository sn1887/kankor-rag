from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orjson
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from kankor_api.routes import whatsapp
from kankor_api.whatsapp.service import WebhookIngestResult


class _FakeQueue:
    def size(self) -> int:
        return 2


class _FakeService:
    def __init__(self) -> None:
        self.queue = _FakeQueue()
        self.received_payloads: list[dict] = []

    def verify_signature(self, *, payload_bytes: bytes, header_signature: str | None) -> bool:
        _ = payload_bytes
        return header_signature == "sha256=ok"

    async def ingest_payload(self, payload: dict) -> WebhookIngestResult:
        self.received_payloads.append(payload)
        return WebhookIngestResult(
            received_messages=1,
            accepted_messages=1,
            duplicate_messages=0,
            unsupported_messages=0,
        )


class _FakeSettings:
    rag_whatsapp_verify_token = "verify-token"


def _patch_state(fake_runtime) -> _FakeService:
    fake_state = SimpleNamespace(
        whatsapp=fake_runtime,
        settings=_FakeSettings(),
    )
    whatsapp.get_app_state = lambda: fake_state  # type: ignore[assignment]
    return fake_runtime.service


def _build_request(payload: dict) -> Request:
    body = orjson.dumps(payload)

    async def receive() -> dict:
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/whatsapp/webhook",
        "headers": [],
    }
    return Request(scope=scope, receive=receive)


def test_whatsapp_verify_route_accepts_correct_challenge() -> None:
    _patch_state(SimpleNamespace(service=_FakeService()))

    response = asyncio.run(
        whatsapp.verify_whatsapp_webhook(
            hub_mode="subscribe",
            hub_verify_token="verify-token",
            hub_challenge="123456",
        )
    )

    assert response.status_code == 200
    assert response.body == b"123456"


def test_whatsapp_ingest_rejects_invalid_signature() -> None:
    _patch_state(SimpleNamespace(service=_FakeService()))

    with pytest.raises(HTTPException, match="Invalid webhook signature"):
        asyncio.run(
            whatsapp.ingest_whatsapp_webhook(
                request=_build_request({"entry": []}),
                x_hub_signature_256="sha256=invalid",
            )
        )


def test_whatsapp_ingest_accepts_valid_payload() -> None:
    fake_service = _patch_state(SimpleNamespace(service=_FakeService()))

    response = asyncio.run(
        whatsapp.ingest_whatsapp_webhook(
            request=_build_request({"entry": []}),
            x_hub_signature_256="sha256=ok",
        )
    )

    assert response["status"] == "accepted"
    assert response["queue_size"] == 2
    assert len(fake_service.received_payloads) == 1
