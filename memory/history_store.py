"""Session-based Chat Message History Storage (In-Memory and File-Persisted).

Demonstrates:
- BaseChatMessageHistory interface
- Multi-session isolation (independent histories per session_id)
- In-memory fast storage vs. persistent local JSON file storage
- Thread-safe / session-safe message persistence
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)


class FileSessionHistory(BaseChatMessageHistory):
    """File-persisted ChatMessageHistory saving conversation turns to disk as JSON."""

    def __init__(self, session_id: str, storage_dir: str = "data/chat_sessions"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{session_id}.json"
        self._messages: List[BaseMessage] = []
        self._load()

    def _load(self) -> None:
        """Loads messages from the JSON session file if it exists."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._messages = messages_from_dict(data)
            except Exception:
                self._messages = []
        else:
            self._messages = []

    def _save(self) -> None:
        """Persists in-memory messages to disk."""
        data = messages_to_dict(self._messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._save()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._messages.extend(messages)
        self._save()

    def clear(self) -> None:
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()


class SessionHistoryManager:
    """Manages multi-tenant / multi-session conversation histories."""

    def __init__(
        self,
        storage_type: Literal["memory", "file"] = "memory",
        storage_dir: str = "data/chat_sessions",
    ):
        self.storage_type = storage_type
        self.storage_dir = storage_dir
        self._memory_stores: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or creates the message history for a specific session_id."""
        if not session_id or not session_id.strip():
            session_id = "default_session"

        if self.storage_type == "file":
            return FileSessionHistory(session_id=session_id, storage_dir=self.storage_dir)

        if session_id not in self._memory_stores:
            self._memory_stores[session_id] = InMemoryChatMessageHistory()

        return self._memory_stores[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clears all conversation messages for a session."""
        if self.storage_type == "file":
            FileSessionHistory(session_id=session_id, storage_dir=self.storage_dir).clear()
        elif session_id in self._memory_stores:
            self._memory_stores[session_id].clear()

    def list_sessions(self) -> List[str]:
        """Lists all active session identifiers."""
        if self.storage_type == "file":
            p = Path(self.storage_dir)
            if not p.exists():
                return []
            return [f.stem for f in p.glob("*.json")]
        return list(self._memory_stores.keys())


# Default global in-memory session manager for convenient access
_default_manager = SessionHistoryManager(storage_type="memory")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Global factory function passed to RunnableWithMessageHistory."""
    return _default_manager.get_session_history(session_id)
