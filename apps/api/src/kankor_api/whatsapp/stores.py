from __future__ import annotations

from collections import deque
from threading import Lock

from rag_core.types import ChatTurn

from .contracts import ConversationStore, ProcessedMessageStore


class InMemoryProcessedMessageStore(ProcessedMessageStore):
    def __init__(self) -> None:
        self._seen_message_ids: set[str] = set()
        self._lock = Lock()

    def mark_if_new(self, message_id: str) -> bool:
        normalized = message_id.strip()
        if not normalized:
            return False
        with self._lock:
            if normalized in self._seen_message_ids:
                return False
            self._seen_message_ids.add(normalized)
            return True


class InMemoryConversationStore(ConversationStore):
    def __init__(self, *, max_turns: int = 6) -> None:
        self.max_turns = max(1, int(max_turns))
        self._history_by_user: dict[str, deque[ChatTurn]] = {}
        self._lock = Lock()

    def get_history(self, wa_id: str) -> list[ChatTurn]:
        with self._lock:
            history = self._history_by_user.get(wa_id)
            if history is None:
                return []
            return list(history)

    def append_turn(self, wa_id: str, *, user_content: str, assistant_content: str) -> None:
        normalized_user = user_content.strip()
        normalized_assistant = assistant_content.strip()
        if not normalized_user and not normalized_assistant:
            return

        with self._lock:
            history = self._history_by_user.setdefault(wa_id, deque(maxlen=self.max_turns * 2))
            if normalized_user:
                history.append(ChatTurn(role='user', content=normalized_user))
            if normalized_assistant:
                history.append(ChatTurn(role='assistant', content=normalized_assistant))
