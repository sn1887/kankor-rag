from __future__ import annotations

import asyncio

from .contracts import JobQueue, WhatsAppJob


class InMemoryJobQueue(JobQueue):
    def __init__(self, *, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[WhatsAppJob] = asyncio.Queue(maxsize=maxsize)

    async def enqueue(self, job: WhatsAppJob) -> None:
        await self._queue.put(job)

    async def dequeue(self, *, timeout_seconds: float = 1.0) -> WhatsAppJob | None:
        if timeout_seconds <= 0:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except TimeoutError:
            return None

    def size(self) -> int:
        return self._queue.qsize()
