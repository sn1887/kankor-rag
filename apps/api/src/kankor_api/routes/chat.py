from __future__ import annotations
import json
from typing import List, Literal
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag_core.contracts.progress import NoopProgressSink
from rag_core.rag.citations import render_references_markdown_from_sources
from ..chat_parsing import MessageCandidate, extract_question_and_history
from ..auth import require_bearer_api_key
from ..stream_progress import PendingProgressSink, parse_truthy_header, render_progress_sse
from ..wiring import get_app_state

router = APIRouter(prefix='/v1/chat', tags=['chat'])

class MessageIn(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    messages: List[MessageIn]

@router.post('/stream')
def stream_chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None, alias='Authorization'),
    x_kankor_progress_events: str | None = Header(default=None, alias='X-Kankor-Progress-Events'),
) -> StreamingResponse:
    parsed = extract_question_and_history(
        [MessageCandidate(role=message.role, content=message.content) for message in request.messages]
    )
    question = parsed.question
    history = parsed.history
    state = get_app_state()
    require_bearer_api_key(
        authorization=authorization,
        expected_token=state.settings.resolved_chat_api_key,
        endpoint_label="chat stream endpoint",
    )
    pipeline = state.pipeline

    def event_stream():
        progress_enabled = parse_truthy_header(x_kankor_progress_events)
        progress_sink = PendingProgressSink() if progress_enabled else NoopProgressSink()
        try:
            for event in pipeline.stream_answer(question=question, history=history, progress_sink=progress_sink):
                if progress_enabled:
                    for progress_event in progress_sink.drain():
                        for chunk in render_progress_sse(progress_event):
                            yield chunk
                if event["type"] == "references":
                    data = dict(event.get("data") or {})
                    if data.get("answer_has_references_heading"):
                        continue
                    sources = list(data.get("sources") or [])
                    rendered = render_references_markdown_from_sources(sources=sources, heading="### منابع")
                    if rendered.strip():
                        payload = json.dumps({"text": f"\n\n{rendered}"}, ensure_ascii=False)
                        yield b"event: delta\n"
                        yield f"data: {payload}\n\n".encode("utf-8")
                    continue
                payload = json.dumps(event['data'], ensure_ascii=False)
                yield f"event: {event['type']}\n".encode('utf-8')
                yield f"data: {payload}\n\n".encode('utf-8')
        except Exception as exc:
            if progress_enabled:
                for progress_event in progress_sink.drain():
                    for chunk in render_progress_sse(progress_event):
                        yield chunk
            error_payload = json.dumps({'message': str(exc)}, ensure_ascii=False)
            yield b'event: error\n'
            yield f'data: {error_payload}\n\n'.encode('utf-8')
        finally:
            if progress_enabled:
                for progress_event in progress_sink.drain():
                    for chunk in render_progress_sse(progress_event):
                        yield chunk
            yield b'event: done\n'
            yield b'data: {}\n\n'

    return StreamingResponse(event_stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache, no-transform', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'})
