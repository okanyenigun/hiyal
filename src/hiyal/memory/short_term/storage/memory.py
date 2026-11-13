"""Storage that keeps entries in Python lists; lost when the process ends."""

from .base import BaseStorage
from ..entry import Entry
from ....utility import Checks


class InMemoryStorage(BaseStorage):
    """Keeps each session's entries in a Python list, in the order they were added."""

    def __init__(self) -> None:
        """Start with no sessions."""
        # One list per session; a session appears with its first entry.
        self._sessions: dict[str, list[Entry]] = {}

    def add(self, session_id: str, entry: Entry) -> None:
        self._sessions.setdefault(
            Checks.non_empty_string(session_id, "session_id"), []
        ).append(entry)

    def remove(self, session_id: str, entry_id: str) -> None:
        target = Checks.non_empty_string(entry_id, "entry_id")
        entries = self._sessions.get(
            Checks.non_empty_string(session_id, "session_id"), []
        )
        for index, stored in enumerate(entries):
            if stored.id_ == target:
                del entries[index]
                return
        raise ValueError(f"entry {target!r} is not stored in session {session_id!r}")

    def entries(self, session_id: str) -> list[Entry]:
        # A copy, so callers cannot change the storage by changing the list.
        return list(
            self._sessions.get(Checks.non_empty_string(session_id, "session_id"), [])
        )

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        wanted = [
            Checks.non_empty_string(entry_id, "entry_id") for entry_id in entry_ids
        ]
        return self._pick(session_id, wanted, self.entries(session_id))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(Checks.non_empty_string(session_id, "session_id"), None)

    # No I/O, so the async twins run directly, without a worker thread.

    async def add_async(self, session_id: str, entry: Entry) -> None:
        self.add(session_id, entry)

    async def remove_async(self, session_id: str, entry_id: str) -> None:
        self.remove(session_id, entry_id)

    async def entries_async(self, session_id: str) -> list[Entry]:
        return self.entries(session_id)

    async def get_async(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        return self.get(session_id, entry_ids)

    async def clear_async(self, session_id: str) -> None:
        self.clear(session_id)
