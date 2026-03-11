from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal
Metadata = Dict[str, Any]

@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

@dataclass(slots=True)
class Hit:
    document: Document
    score: float

@dataclass(slots=True)
class ChatTurn:
    role: Literal['user', 'assistant', 'system']
    content: str
