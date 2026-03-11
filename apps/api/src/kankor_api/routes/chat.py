from __future__ import annotations
import json
from typing import List, Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag_core.types import ChatTurn
from ..wiring import get_app_state

router = APIRouter(prefix='/v1/chat', tags=['chat'])

class MessageIn(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    messages: List[MessageIn]

@router.post('/stream')
def stream_chat(request: ChatRequest) -> StreamingResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail='messages must not be empty')
    user_messages = [message for message in request.messages if message.role == 'user']
    if not user_messages:
        raise HTTPException(status_code=400, detail='at least one user message is required')
    question = user_messages[-1].content
    history = [ChatTurn(role=message.role, content=message.content) for message in request.messages[:-1]]
    pipeline = get_app_state().pipeline

    def event_stream():
        try:
            for event in pipeline.stream_answer(question=question, history=history):
                payload = json.dumps(event['data'], ensure_ascii=False)
                yield f"event: {event['type']}\n".encode('utf-8')
                yield f"data: {payload}\n\n".encode('utf-8')
        except Exception as exc:
            error_payload = json.dumps({'message': str(exc)}, ensure_ascii=False)
            yield b'event: error\n'
            yield f'data: {error_payload}\n\n'.encode('utf-8')
        finally:
            yield b'event: done\n'
            yield b'data: {}\n\n'

    return StreamingResponse(event_stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache, no-transform', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'})
