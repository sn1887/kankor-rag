from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_coro_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync code.

    This is used by sync generators like `RAGPipeline.stream_answer()` which can be
    called both from sync HTTP handlers and from async worker code (e.g. WhatsApp).
    If an event loop is already running in the current thread, we run the coroutine
    in a dedicated thread with its own loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - defensive path
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not result:  # pragma: no cover - defensive path
        raise RuntimeError("run_coro_sync returned no result.")
    return result[0]

