from __future__ import annotations

import asyncio

from kankor_api.whatsapp.contracts import WhatsAppJob
from kankor_api.whatsapp.events import WhatsAppInboundMessage, WhatsAppTextPayload
from kankor_api.whatsapp.redis_backends import (
    RedisConversationStore,
    RedisJobQueue,
    RedisProcessedMessageStore,
)


class _FakeRedisState:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}


class _FakeAsyncRedisClient:
    def __init__(self, state: _FakeRedisState) -> None:
        self._state = state
        self.closed = False

    async def lpush(self, key: str, value: str) -> None:
        self._state.lists.setdefault(key, []).insert(0, value)

    async def brpop(self, key: str, timeout: int = 0):
        _ = timeout
        queue = self._state.lists.setdefault(key, [])
        if not queue:
            return None
        return key, queue.pop()

    async def aclose(self) -> None:
        self.closed = True


class _FakeSyncPipeline:
    def __init__(self, client: _FakeSyncRedisClient) -> None:
        self._client = client
        self._commands: list[tuple] = []

    def rpush(self, key: str, value: str) -> _FakeSyncPipeline:
        self._commands.append(("rpush", key, value))
        return self

    def ltrim(self, key: str, start: int, end: int) -> _FakeSyncPipeline:
        self._commands.append(("ltrim", key, start, end))
        return self

    def expire(self, key: str, ttl: int) -> _FakeSyncPipeline:
        self._commands.append(("expire", key, ttl))
        return self

    def execute(self) -> None:
        for command in self._commands:
            op = command[0]
            if op == "rpush":
                _, key, value = command
                self._client.rpush(key, value)
            elif op == "ltrim":
                _, key, start, end = command
                self._client.ltrim(key, start, end)
            elif op == "expire":
                pass


class _FakeSyncRedisClient:
    def __init__(self, state: _FakeRedisState) -> None:
        self._state = state
        self.closed = False

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        _ = ex
        if nx and key in self._state.kv:
            return False
        self._state.kv[key] = value
        return True

    def llen(self, key: str) -> int:
        return len(self._state.lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._state.lists.get(key, [])
        return _slice(values, start, end)

    def rpush(self, key: str, value: str) -> None:
        self._state.lists.setdefault(key, []).append(value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        values = self._state.lists.get(key, [])
        self._state.lists[key] = _slice(values, start, end)

    def expire(self, key: str, ttl: int) -> bool:
        _ = key, ttl
        return True

    def pipeline(self) -> _FakeSyncPipeline:
        return _FakeSyncPipeline(self)

    def close(self) -> None:
        self.closed = True


class _FakeAsyncRedisModule:
    def __init__(self, state: _FakeRedisState) -> None:
        self._state = state

    class Redis:
        _state: _FakeRedisState | None = None

        @classmethod
        def from_url(cls, redis_url: str):
            _ = redis_url
            assert cls._state is not None
            return _FakeAsyncRedisClient(cls._state)


class _FakeSyncRedisModule:
    def __init__(self, state: _FakeRedisState) -> None:
        self._state = state

    class Redis:
        _state: _FakeRedisState | None = None

        @classmethod
        def from_url(cls, redis_url: str):
            _ = redis_url
            assert cls._state is not None
            return _FakeSyncRedisClient(cls._state)


def _slice(values: list[str], start: int, end: int) -> list[str]:
    length = len(values)
    if length == 0:
        return []

    resolved_start = start if start >= 0 else max(0, length + start)
    if end == -1:
        resolved_end = length - 1
    else:
        resolved_end = end if end >= 0 else length + end
    if resolved_end < resolved_start:
        return []
    return values[resolved_start:resolved_end + 1]


def _patch_redis_modules(monkeypatch):
    from kankor_api.whatsapp import redis_backends

    state = _FakeRedisState()
    async_module = _FakeAsyncRedisModule(state)
    sync_module = _FakeSyncRedisModule(state)
    _FakeAsyncRedisModule.Redis._state = state
    _FakeSyncRedisModule.Redis._state = state

    monkeypatch.setattr(redis_backends, "_load_async_redis_module", lambda: async_module)
    monkeypatch.setattr(redis_backends, "_load_sync_redis_module", lambda: sync_module)


def test_redis_job_queue_roundtrip(monkeypatch) -> None:
    _patch_redis_modules(monkeypatch)

    queue = RedisJobQueue(redis_url="redis://fake", queue_key="kankor:wa:queue")
    job = WhatsAppJob(
        message=WhatsAppInboundMessage(
            message_id="wamid-1",
            from_wa_id="93700111222",
            timestamp=1711000000,
            message_type="text",
            text=WhatsAppTextPayload(body="Explain momentum"),
        )
    )

    asyncio.run(queue.enqueue(job))
    assert queue.size() == 1

    popped = asyncio.run(queue.dequeue(timeout_seconds=1))
    assert popped is not None
    assert popped.message.message_id == "wamid-1"
    assert popped.message.text is not None
    assert popped.message.text.body == "Explain momentum"
    assert queue.size() == 0

    asyncio.run(queue.close())


def test_redis_processed_store_idempotency(monkeypatch) -> None:
    _patch_redis_modules(monkeypatch)

    store = RedisProcessedMessageStore(
        redis_url="redis://fake",
        key_prefix="kankor:wa",
        ttl_seconds=120,
    )

    assert store.mark_if_new("wamid-1") is True
    assert store.mark_if_new("wamid-1") is False


def test_redis_conversation_store_trim_and_read(monkeypatch) -> None:
    _patch_redis_modules(monkeypatch)

    store = RedisConversationStore(
        redis_url="redis://fake",
        key_prefix="kankor:wa",
        max_turns=2,
        ttl_seconds=120,
    )

    store.append_turn("93700111222", user_content="q1", assistant_content="a1")
    store.append_turn("93700111222", user_content="q2", assistant_content="a2")
    store.append_turn("93700111222", user_content="q3", assistant_content="a3")

    history = store.get_history("93700111222")
    assert len(history) == 4
    assert history[0].content == "q2"
    assert history[-1].content == "a3"
