"""The manager's pins: the entries kept in what goes to the model, whatever the policy chooses."""

from ..schemas import Entry, EntryKind, Pin
from ..storage import BaseStorage


class PinKeeper:
    """Keeps a session's pins for the ``ShortTermMemoryManager``: which entries are pinned, and their records.

    A pin is kept twice: here, in memory, for the manager's every read; and
    in the session, as a pin record stored right after the entry, so a
    manager opened on the session again knows it. This class keeps the two
    in step. The manager decides *when* to pin or unpin; this class does it.

    For the manager's async side, each method that writes or reads entries
    has an ``_async`` twin, using the storage's async twins.
    """

    def __init__(self, storage: BaseStorage, session_id: str) -> None:
        """Keep the pins of ``session_id`` in ``storage``; none until ``read_back`` or ``pin``."""
        self.storage = storage
        self.session_id = session_id
        # Pinned entry ID -> its pin record's ID, oldest first.
        self._records: dict[str, str] = {}

    def read_back(self, index: list[tuple[str, EntryKind]]) -> None:
        """Rebuild the pins from the pin records in the session's ``index``, oldest first; only the records are read.

        For a session opened again, or rewound. The records are stored in
        the order of their entries, so the pins keep it.
        """
        record_ids = [entry_id for entry_id, kind in index if kind is EntryKind.PIN]
        self._records = {
            record.payload.entry_id: record.id_
            for record in self.storage.get(self.session_id, record_ids)
            if isinstance(record.payload, Pin)
        }

    def pin(self, entry_id: str) -> None:
        """Pin the stored entry ``entry_id``: store its pin record, right after it when it was just added."""
        record = Entry.of(Pin(entry_id=entry_id))
        self.storage.add(self.session_id, record)
        self._records[entry_id] = record.id_

    async def pin_async(self, entry_id: str) -> None:
        """``pin``, with the storage's async twins."""
        record = Entry.of(Pin(entry_id=entry_id))
        await self.storage.add_async(self.session_id, record)
        self._records[entry_id] = record.id_

    def unpin(self, entry_id: str) -> str:
        """Unpin ``entry_id``: remove its pin record from the session; returns the record's ID.

        Raises ``ValueError`` if the entry is not pinned.
        """
        record_id = self._record_id(entry_id)
        self.storage.remove(self.session_id, record_id)
        del self._records[entry_id]
        return record_id

    async def unpin_async(self, entry_id: str) -> str:
        """``unpin``, with the storage's async twins."""
        record_id = self._record_id(entry_id)
        await self.storage.remove_async(self.session_id, record_id)
        del self._records[entry_id]
        return record_id

    def is_pinned(self, entry_id: str) -> bool:
        """Whether ``entry_id`` is pinned."""
        return entry_id in self._records

    @property
    def ids(self) -> list[str]:
        """The pinned entries' IDs, oldest first."""
        return list(self._records)

    def entries(self) -> list[Entry]:
        """The pinned entries, oldest first; only those are read."""
        return self.storage.get(self.session_id, self.ids)

    async def entries_async(self) -> list[Entry]:
        """``entries``, with the storage's async twins."""
        return await self.storage.get_async(self.session_id, self.ids)

    def _record_id(self, entry_id: str) -> str:
        """The ID of the pin record of ``entry_id``; raises ``ValueError`` if the entry is not pinned."""
        record_id = self._records.get(entry_id)
        if record_id is None:
            raise ValueError(f"entry {entry_id!r} is not pinned")
        return record_id
