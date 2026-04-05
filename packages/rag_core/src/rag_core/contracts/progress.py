from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class ProgressStage(str, Enum):
    THINKING = "thinking"
    RETRIEVING = "retrieving"
    READING = "reading"
    WRITING = "writing"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class ProgressEvent:
    stage: ProgressStage
    message: str
    done: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class ProgressSink(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...


class NoopProgressSink:
    def emit(self, event: ProgressEvent) -> None:
        _ = event

    def close(self) -> None:
        return None
