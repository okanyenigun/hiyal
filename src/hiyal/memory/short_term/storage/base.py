"""The storage interface every short-term memory storage implements."""

import asyncio
from abc import ABC, abstractmethod
from ..schemas import Entry, EntryKind


class BaseStorage(ABC):
    """Keeps entries in order, separately for each session, each ID once.

    One storage can hold many sessions: every call names the session it
    acts on, and sees only that session's entries.

    IDs are unique within a session, and the storage enforces it: ``add``
    refuses another entry with an ID already stored, and does not store a
    **repeat** (the same ID with the same content, e.g. a retried add)
    again. Content is compared as stored, as JSON, so ``LLMResponse.raw``
    (never stored) does not count.

    Every method has an ``_async`` twin (``add_async``, ``get_async``, ...)
    for async code, with the same arguments and result. By default it runs
    the sync method in a worker thread, so file or database work never
    blocks the event loop; a storage with no I/O overrides it to run
    directly, and one with an async driver to use it.
    """

    @abstractmethod
    def add(self, session_id: str, entry: Entry) -> bool:
        """Append ``entry`` to the end of the session; ``True`` when stored, ``False`` for a repeat.

        Raises ``ValueError`` if the session already holds another entry
        with the same ID.
        """

    @abstractmethod
    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        """The session's entries with these IDs, in the order of ``entry_ids``.

        Raises ``ValueError`` if the session has no entry with one of them.
        """

    def index(self, session_id: str) -> list[tuple[str, EntryKind]]:
        """The session's ``(id, kind)`` pairs, oldest first: what is stored, without the entries.

        Lets a reader find what it needs before fetching it with ``get``.
        By default it is made from ``entries()``; a storage that can read
        less overrides it.
        """
        return [(entry.id_, entry.kind) for entry in self.entries(session_id)]

    @abstractmethod
    def entries(self, session_id: str) -> list[Entry]:
        """Every entry of the session, oldest first; none for a session nothing was added to."""

    @abstractmethod
    def remove(self, session_id: str, entry_id: str) -> None:
        """Remove the session's entry with ``entry_id``; raises ``ValueError`` if there is none."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Remove every entry of the session; other sessions keep theirs."""

    # ------------------------------------------------------------------ async

    async def add_async(self, session_id: str, entry: Entry) -> bool:
        """``add``, in a worker thread by default."""
        return await asyncio.to_thread(self.add, session_id, entry)

    async def get_async(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        """``get``, in a worker thread by default."""
        return await asyncio.to_thread(self.get, session_id, entry_ids)

    async def index_async(self, session_id: str) -> list[tuple[str, EntryKind]]:
        """``index``, in a worker thread by default."""
        return await asyncio.to_thread(self.index, session_id)

    async def entries_async(self, session_id: str) -> list[Entry]:
        """``entries``, in a worker thread by default."""
        return await asyncio.to_thread(self.entries, session_id)

    async def remove_async(self, session_id: str, entry_id: str) -> None:
        """``remove``, in a worker thread by default."""
        await asyncio.to_thread(self.remove, session_id, entry_id)

    async def clear_async(self, session_id: str) -> None:
        """``clear``, in a worker thread by default."""
        await asyncio.to_thread(self.clear, session_id)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _repeat(session_id: str, entry: Entry, stored: Entry) -> bool:
        """Decide what to do with ``entry`` when ``stored`` already has its ID: returns ``False`` or raises ``ValueError``.

        - Same content (e.g. a retried add): it is a repeat, so returns
          ``False`` and nothing is stored again.
        - Different content: two different entries cannot share an ID, so
          raises ``ValueError``.

        Content is compared as stored, as JSON.
        """
        if stored.model_dump_json() == entry.model_dump_json():
            return False
        raise ValueError(
            f"another entry with id_ {entry.id_!r} is already in session {session_id!r}"
        )

    @staticmethod
    def _missing(session_id: str, entry_ids: list[str], found: set[str]) -> None:
        """Raise ``ValueError`` naming the ``entry_ids`` not ``found``, if any."""
        missing = [entry_id for entry_id in entry_ids if entry_id not in found]
        if missing:
            raise ValueError(
                f"entries {missing!r} are not stored in session {session_id!r}"
            )
