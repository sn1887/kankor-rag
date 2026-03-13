from __future__ import annotations

import json
import time
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..chat_parsing import MessageCandidate, extract_question_and_history
from ..auth import require_bearer_api_key
from ..wiring import get_app_state

router = APIRouter(prefix='/v1', tags=['openai-compat'])


class ContentPart(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: str
    text: str | None = None


class OpenAIMessageIn(BaseModel):
    model_config = ConfigDict(extra='ignore')
    role: str
    content: str | list[ContentPart] | None = None


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    model: str | None = None
    messages: list[OpenAIMessageIn]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)


def _resolve_message_content(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    chunks = [part.text for part in content if part.type == 'text' and part.text]
    return '\n'.join(chunks).strip()


def _check_openai_compat_key(authorization: str | None) -> None:
    settings = get_app_state().settings
    require_bearer_api_key(
        authorization=authorization,
        expected_token=settings.rag_openai_compat_api_key,
        endpoint_label="OpenAI-compatible endpoint",
    )


def _active_model_id() -> str:
    return get_app_state().settings.active_llm_model_id


def _resolve_response_model(requested_model: str | None) -> str:
    active_model = _active_model_id()
    candidate = (requested_model or '').strip()
    if candidate and candidate != active_model:
        raise HTTPException(
            status_code=400,
            detail=f'model "{candidate}" is not available. Use "{active_model}".',
        )
    return active_model


def _resolve_requested_max_tokens(request: ChatCompletionsRequest) -> int | None:
    if request.max_tokens is not None and request.max_completion_tokens is not None and request.max_tokens != request.max_completion_tokens:
        raise HTTPException(
            status_code=400,
            detail='max_tokens and max_completion_tokens must match when both are provided.',
        )
    return request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens


@router.get('/models')
def list_models(authorization: str | None = Header(default=None, alias='Authorization')) -> dict:
    _check_openai_compat_key(authorization)
    created = int(time.time())
    model_id = _active_model_id()
    return {
        'object': 'list',
        'data': [
            {
                'id': model_id,
                'object': 'model',
                'created': created,
                'owned_by': 'kankor-rag',
            }
        ],
    }


@router.post('/chat/completions')
def chat_completions(
    request: ChatCompletionsRequest,
    authorization: str | None = Header(default=None, alias='Authorization'),
):
    _check_openai_compat_key(authorization)
    parsed = extract_question_and_history(
        [
            MessageCandidate(role=message.role, content=_resolve_message_content(message.content))
            for message in request.messages
        ]
    )
    question = parsed.question
    history = parsed.history
    state = get_app_state()
    pipeline = state.pipeline

    created = int(time.time())
    completion_id = f'chatcmpl-{uuid4().hex}'
    model_name = _resolve_response_model(request.model)
    requested_max_tokens = _resolve_requested_max_tokens(request)
    max_new_tokens = requested_max_tokens if requested_max_tokens is not None else pipeline.max_new_tokens
    temperature = request.temperature if request.temperature is not None else pipeline.temperature
    if requested_max_tokens is not None and requested_max_tokens > pipeline.max_new_tokens_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_tokens exceeds limit ({pipeline.max_new_tokens_limit}). "
                "Reduce max_tokens or increase RAG_MAX_NEW_TOKENS_HARD_LIMIT."
            ),
        )
    if request.temperature is not None and not (
        pipeline.temperature_min <= request.temperature <= pipeline.temperature_max
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"temperature must be between {pipeline.temperature_min} "
                f"and {pipeline.temperature_max}."
            ),
        )
    max_new_tokens, temperature = pipeline.resolve_generation_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if request.stream:
        def event_stream():
            try:
                role_chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model_name,
                    'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}],
                }
                yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode('utf-8')

                for event in pipeline.stream_answer(
                    question=question,
                    history=history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    if event['type'] != 'delta':
                        continue
                    delta_text = str(event['data'].get('text', ''))
                    if not delta_text:
                        continue
                    chunk = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model_name,
                        'choices': [{'index': 0, 'delta': {'content': delta_text}, 'finish_reason': None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')

                final_chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model_name,
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
            except Exception as exc:
                error_payload = {'error': {'message': str(exc), 'type': 'server_error'}}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode('utf-8')
            finally:
                yield b'data: [DONE]\n\n'

        return StreamingResponse(
            event_stream(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache, no-transform', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'},
        )

    chunks: list[str] = []
    for event in pipeline.stream_answer(
        question=question,
        history=history,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    ):
        if event['type'] == 'delta':
            chunks.append(str(event['data'].get('text', '')))
    answer = ''.join(chunks)
    response = {
        'id': completion_id,
        'object': 'chat.completion',
        'created': created,
        'model': model_name,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': answer},
                'finish_reason': 'stop',
            }
        ],
    }
    return JSONResponse(response)
