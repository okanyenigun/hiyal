"""Storage that keeps entries in memory; lost when the process ends."""

from .base import BaseStorage
from ..schemas import Entry, EntryKind
from ....utility import Checks


class InMemoryStorage(BaseStorage):
    """Keeps each session's entries in a dict by ID, which also keeps them in the order they were added."""

    def __init__(self) -> None:
        """Start with no sessions."""
        # One dict per session, ID -> entry; a session appears with its first entry.
        self._sessions: dict[str, dict[str, Entry]] = {}

    def add(self, session_id: str, entry: Entry) -> bool:
        session = self._sessions.setdefault(
            Checks.non_empty_string(session_id, "session_id"), {}
        )
        if entry.id_ in session:
            return self._repeat(session_id, entry, session[entry.id_])
        session[entry.id_] = entry
        return True

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        session = self._session(session_id)
        self._missing(session_id, entry_ids, set(session))
        return [session[entry_id] for entry_id in entry_ids]

    def entries(self, session_id: str) -> list[Entry]:
        return list(self._session(session_id).values())

    def remove(self, session_id: str, entry_id: str) -> None:
        session = self._session(session_id)
        if session.pop(Checks.non_empty_string(entry_id, "entry_id"), None) is None:
            raise ValueError(
                f"entry {entry_id!r} is not stored in session {session_id!r}"
            )

    def clear(self, session_id: str) -> None:
        self._sessions.pop(Checks.non_empty_string(session_id, "session_id"), None)

    # No I/O, so the async twins run directly, without a worker thread.

    async def add_async(self, session_id: str, entry: Entry) -> bool:
        return self.add(session_id, entry)

    async def get_async(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        return self.get(session_id, entry_ids)

    async def index_async(self, session_id: str) -> list[tuple[str, EntryKind]]:
        return self.index(session_id)

    async def entries_async(self, session_id: str) -> list[Entry]:
        return self.entries(session_id)

    async def remove_async(self, session_id: str, entry_id: str) -> None:
        self.remove(session_id, entry_id)

    async def clear_async(self, session_id: str) -> None:
        self.clear(session_id)

    def _session(self, session_id: str) -> dict[str, Entry]:
        """The session's entries by ID; empty for a session nothing was added to."""
        return self._sessions.get(Checks.non_empty_string(session_id, "session_id"), {})
