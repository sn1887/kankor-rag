from __future__ import annotations

import orjson
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ..wiring import get_app_state

router = APIRouter(prefix='/v1/whatsapp', tags=['whatsapp'])


def _whatsapp_runtime_or_404():
    runtime = get_app_state().whatsapp
    if runtime is None:
        raise HTTPException(status_code=404, detail='WhatsApp channel is not enabled')
    return runtime


@router.get('/webhook')
async def verify_whatsapp_webhook(
    hub_mode: str | None = Query(default=None, alias='hub.mode'),
    hub_verify_token: str | None = Query(default=None, alias='hub.verify_token'),
    hub_challenge: str | None = Query(default=None, alias='hub.challenge'),
):
    _whatsapp_runtime_or_404()
    expected_token = (get_app_state().settings.rag_whatsapp_verify_token or '').strip()
    if (
        hub_mode == 'subscribe'
        and hub_challenge is not None
        and expected_token
        and hub_verify_token == expected_token
    ):
        return PlainTextResponse(content=hub_challenge)
    raise HTTPException(status_code=403, detail='Webhook verification failed')


@router.post('/webhook')
async def ingest_whatsapp_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None, alias='X-Hub-Signature-256'),
):
    runtime = _whatsapp_runtime_or_404()
    body = await request.body()
    if not runtime.service.verify_signature(
        payload_bytes=body,
        header_signature=x_hub_signature_256,
    ):
        raise HTTPException(status_code=401, detail='Invalid webhook signature')

    try:
        parsed = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail='Invalid JSON payload') from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail='Webhook payload must be a JSON object')

    result = await runtime.service.ingest_payload(parsed)
    return {
        'status': 'accepted',
        **result.as_dict(),
        'queue_size': runtime.service.queue.size(),
    }
