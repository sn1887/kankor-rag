from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from rag_core.types import ChatTurn

from .contracts import ConversationStore, JobQueue, ProcessedMessageStore, WhatsAppJob
from .events import WhatsAppImagePayload, WhatsAppInboundMessage, WhatsAppTextPayload


def _load_async_redis_module():
    try:
        import redis.asyncio as redis_async
    except ImportError as exc:
        raise RuntimeError(
            "redis package is required for Redis WhatsApp backends. Install it with `pip install redis`."
        ) from exc
    return redis_async


def _load_sync_redis_module():
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "redis package is required for Redis WhatsApp backends. Install it with `pip install redis`."
        ) from exc
    return redis


class RedisJobQueue(JobQueue):
    def __init__(
        self,
        *,
        redis_url: str,
        queue_key: str,
        async_client: Any | None = None,
        sync_client: Any | None = None,
    ) -> None:
        self.redis_url = redis_url.strip()
        self.queue_key = queue_key.strip()

        if async_client is None:
            redis_async = _load_async_redis_module()
            async_client = redis_async.Redis.from_url(self.redis_url)
        if sync_client is None:
            redis_sync = _load_sync_redis_module()
            sync_client = redis_sync.Redis.from_url(self.redis_url)

        self._async_client = async_client
        self._sync_client = sync_client

    async def enqueue(self, job: WhatsAppJob) -> None:
        payload = json.dumps(_job_to_dict(job), ensure_ascii=False)
        await self._async_client.lpush(self.queue_key, payload)

    async def dequeue(self, *, timeout_seconds: float = 1.0) -> WhatsAppJob | None:
        timeout = 0 if timeout_seconds <= 0 else max(1, int(math.ceil(timeout_seconds)))
        entry = await self._async_client.brpop(self.queue_key, timeout=timeout)
        if entry is None:
            return None

        raw_payload: Any
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            raw_payload = entry[1]
        else:
            raw_payload = entry

        payload_dict = json.loads(_decode_redis_value(raw_payload))
        if not isinstance(payload_dict, dict):
            raise ValueError("Redis queue payload must decode to a JSON object")
        return _job_from_dict(payload_dict)

    def size(self) -> int:
        try:
            return int(self._sync_client.llen(self.queue_key))
        except Exception:
            return -1

    async def close(self) -> None:
        await self._async_client.aclose()
        close_fn = getattr(self._sync_client, "close", None)
        if callable(close_fn):
            close_fn()


class RedisProcessedMessageStore(ProcessedMessageStore):
    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        ttl_seconds: int = 172800,
        client: Any | None = None,
    ) -> None:
        self.redis_url = redis_url.strip()
        self.key_prefix = key_prefix.strip()
        self.ttl_seconds = max(1, int(ttl_seconds))

        if client is None:
            redis_sync = _load_sync_redis_module()
            client = redis_sync.Redis.from_url(self.redis_url)
        self._client = client

    def mark_if_new(self, message_id: str) -> bool:
        normalized = message_id.strip()
        if not normalized:
            return False
        result = self._client.set(
            self._processed_key(normalized),
            "1",
            nx=True,
            ex=self.ttl_seconds,
        )
        return bool(result)

    def _processed_key(self, message_id: str) -> str:
        return f"{self.key_prefix}:processed:{message_id}"

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


class RedisConversationStore(ConversationStore):
    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        max_turns: int = 6,
        ttl_seconds: int = 604800,
        client: Any | None = None,
    ) -> None:
        self.redis_url = redis_url.strip()
        self.key_prefix = key_prefix.strip()
        self.max_turns = max(1, int(max_turns))
        self.ttl_seconds = max(1, int(ttl_seconds))

        if client is None:
            redis_sync = _load_sync_redis_module()
            client = redis_sync.Redis.from_url(self.redis_url)
        self._client = client

    def get_history(self, wa_id: str) -> list[ChatTurn]:
        key = self._history_key(wa_id)
        raw_entries = self._client.lrange(key, 0, -1)
        history: list[ChatTurn] = []
        for raw in raw_entries:
            try:
                payload = json.loads(_decode_redis_value(raw))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            role = str(payload.get("role") or "").strip()
            content = str(payload.get("content") or "").strip()
            if role not in {"user", "assistant", "system"}:
                continue
            if not content:
                continue
            history.append(ChatTurn(role=role, content=content))
        return history

    def append_turn(self, wa_id: str, *, user_content: str, assistant_content: str) -> None:
        normalized_user = user_content.strip()
        normalized_assistant = assistant_content.strip()
        if not normalized_user and not normalized_assistant:
            return

        key = self._history_key(wa_id)
        pipeline = self._client.pipeline()

        if normalized_user:
            pipeline.rpush(key, json.dumps({"role": "user", "content": normalized_user}, ensure_ascii=False))
        if normalized_assistant:
            pipeline.rpush(
                key,
                json.dumps({"role": "assistant", "content": normalized_assistant}, ensure_ascii=False),
            )

        max_entries = self.max_turns * 2
        pipeline.ltrim(key, -max_entries, -1)
        pipeline.expire(key, self.ttl_seconds)
        pipeline.execute()

    def _history_key(self, wa_id: str) -> str:
        return f"{self.key_prefix}:history:{wa_id.strip()}"

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


def _job_to_dict(job: WhatsAppJob) -> dict[str, Any]:
    message = job.message
    return {
        "message": {
            "message_id": message.message_id,
            "from_wa_id": message.from_wa_id,
            "timestamp": message.timestamp,
            "message_type": message.message_type,
            "contact_name": message.contact_name,
            "phone_number_id": message.phone_number_id,
            "text": (
                {
                    "body": message.text.body,
                }
                if message.text is not None
                else None
            ),
            "image": (
                {
                    "media_id": message.image.media_id,
                    "mime_type": message.image.mime_type,
                    "sha256": message.image.sha256,
                    "caption": message.image.caption,
                }
                if message.image is not None
                else None
            ),
        },
        "received_at": job.received_at.astimezone(timezone.utc).isoformat(),
    }


def _job_from_dict(payload: dict[str, Any]) -> WhatsAppJob:
    message_payload = payload.get("message")
    if not isinstance(message_payload, dict):
        raise ValueError("Invalid Redis queue payload: missing message object")

    message_id = str(message_payload.get("message_id") or "").strip()
    from_wa_id = str(message_payload.get("from_wa_id") or "").strip()
    message_type = str(message_payload.get("message_type") or "").strip()
    if not message_id or not from_wa_id or not message_type:
        raise ValueError("Invalid Redis queue payload: missing required message fields")

    raw_timestamp = message_payload.get("timestamp")
    timestamp: int | None
    if raw_timestamp is None or raw_timestamp == "":
        timestamp = None
    else:
        try:
            timestamp = int(raw_timestamp)
        except (TypeError, ValueError):
            timestamp = None

    text_payload = message_payload.get("text")
    text: WhatsAppTextPayload | None = None
    if isinstance(text_payload, dict):
        text = WhatsAppTextPayload(body=str(text_payload.get("body") or "").strip())

    image_payload = message_payload.get("image")
    image: WhatsAppImagePayload | None = None
    if isinstance(image_payload, dict):
        media_id = str(image_payload.get("media_id") or "").strip()
        if media_id:
            image = WhatsAppImagePayload(
                media_id=media_id,
                mime_type=_optional_str(image_payload.get("mime_type")),
                sha256=_optional_str(image_payload.get("sha256")),
                caption=_optional_str(image_payload.get("caption")),
            )

    message = WhatsAppInboundMessage(
        message_id=message_id,
        from_wa_id=from_wa_id,
        timestamp=timestamp,
        message_type=message_type,
        contact_name=_optional_str(message_payload.get("contact_name")),
        phone_number_id=_optional_str(message_payload.get("phone_number_id")),
        text=text,
        image=image,
    )
    return WhatsAppJob(message=message, received_at=_parse_received_at(payload.get("received_at")))


def _parse_received_at(raw: Any) -> datetime:
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized:
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
    return datetime.now(timezone.utc)


def _decode_redis_value(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, memoryview):
        return raw.tobytes().decode("utf-8", errors="ignore")
    return str(raw)


def _optional_str(raw: Any) -> str | None:
    value = str(raw or "").strip()
    return value or None
