"""The storage interface every short-term memory storage implements."""

import asyncio
from abc import ABC, abstractmethod
from ..entry import Entry


class BaseStorage(ABC):
    """Keeps entries in order, separately for each session.

    One storage can serve many sessions at the same time: every call names
    the session it acts on, and sees only that session's entries. A
    ``ShortTermMemoryManager`` manages one session; it passes its
    ``session_id`` to every call, both to its primary storage and to its
    dump storage.

    Every method has an ``_async`` twin (``add_async``, ``entries_async``,
    ...) for async code. By default it runs the sync method in a worker
    thread, so file or database work never blocks the event loop; a storage
    with no I/O can override it to run directly.
    """

    @abstractmethod
    def add(self, session_id: str, entry: Entry) -> None:
        """Append an entry to the end of the session.

        Does not check for duplicates; preventing them is the manager's job.
        """

    @abstractmethod
    def remove(self, session_id: str, entry_id: str) -> None:
        """Remove the session's entry whose ``id_`` is ``entry_id``.

        If the ID was stored more than once in the session, only the oldest
        copy is removed. Raises ``ValueError`` if the session has no entry
        with that ID, and ``TypeError`` if ``entry_id`` is not a string.
        """

    @abstractmethod
    def entries(self, session_id: str) -> list[Entry]:
        """The session's entries, oldest first, as a new list the caller may change.

        A session nothing was added to has no entries.
        """

    @abstractmethod
    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        """The session's entries with these IDs, in the order of ``entry_ids``.

        Reads only what it needs where the storage allows it (SQLite). If an
        ID was stored more than once in the session, the oldest copy is
        returned. Raises ``ValueError`` if the session has no entry with one
        of the IDs, and ``TypeError`` if an ID is not a string.
        """

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Remove every entry of the session at once; other sessions keep theirs."""

    async def add_async(self, session_id: str, entry: Entry) -> None:
        """``add``, in a worker thread."""
        await asyncio.to_thread(self.add, session_id, entry)

    async def remove_async(self, session_id: str, entry_id: str) -> None:
        """``remove``, in a worker thread."""
        await asyncio.to_thread(self.remove, session_id, entry_id)

    async def entries_async(self, session_id: str) -> list[Entry]:
        """``entries``, in a worker thread."""
        return await asyncio.to_thread(self.entries, session_id)

    async def get_async(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        """``get``, in a worker thread."""
        return await asyncio.to_thread(self.get, session_id, entry_ids)

    async def clear_async(self, session_id: str) -> None:
        """``clear``, in a worker thread."""
        await asyncio.to_thread(self.clear, session_id)

    @staticmethod
    def _pick(
        session_id: str, entry_ids: list[str], stored: list[Entry]
    ) -> list[Entry]:
        """The entries of ``stored`` with ``entry_ids``, in that order; the oldest copy of a repeated ID.

        For ``get``. Raises ``ValueError`` for an ID ``stored`` does not hold.
        """
        by_id: dict[str, Entry] = {}
        for entry in stored:
            by_id.setdefault(entry.id_, entry)
        missing = [entry_id for entry_id in entry_ids if entry_id not in by_id]
        if missing:
            raise ValueError(
                f"entries {missing!r} are not stored in session {session_id!r}"
            )
        return [by_id[entry_id] for entry_id in entry_ids]
