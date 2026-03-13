from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from typing import Protocol

from .contracts import JobQueue
from .processor import WhatsAppMessageProcessor
from .service import WhatsAppWebhookService

logger = logging.getLogger(__name__)


class Closable(Protocol):
    def close(self):
        ...


class WhatsAppWorker:
    def __init__(
        self,
        *,
        queue: JobQueue,
        processor: WhatsAppMessageProcessor,
        poll_timeout_seconds: float = 1.0,
    ) -> None:
        self.queue = queue
        self.processor = processor
        self.poll_timeout_seconds = max(0.1, float(poll_timeout_seconds))

    async def run(self, *, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            job = await self.queue.dequeue(timeout_seconds=self.poll_timeout_seconds)
            if job is None:
                continue
            try:
                await self.processor.process(job)
            except Exception:
                logger.exception("Unhandled WhatsApp worker error")


class WhatsAppRuntime:
    def __init__(
        self,
        *,
        service: WhatsAppWebhookService,
        worker: WhatsAppWorker,
        concurrency: int = 2,
        closables: Sequence[Closable] | None = None,
    ) -> None:
        self.service = service
        self.worker = worker
        self.concurrency = max(1, int(concurrency))
        self._closables = list(closables or [])
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event.clear()
        self._tasks = [
            asyncio.create_task(
                self.worker.run(stop_event=self._stop_event),
                name=f"whatsapp-worker-{index + 1}",
            )
            for index in range(self.concurrency)
        ]

    async def stop(self) -> None:
        self._stop_event.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        for closable in self._closables:
            try:
                result = closable.close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Failed to close WhatsApp dependency %s", type(closable).__name__)
