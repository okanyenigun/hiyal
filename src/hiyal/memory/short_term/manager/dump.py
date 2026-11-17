"""The manager's dump: copying the session's entries to an archive."""

from ..schemas import MANAGED_KINDS, Entry, EntryKind
from ..storage import BaseStorage


class DumpKeeper:
    """Keeps a session's dump for the ``ShortTermMemoryManager``: how far the archive goes, and the adds since the last dump.

    The archive (``dump_storage``) holds a copy of the session's first
    stored entries, under the same session ID; a dump copies the ones after
    them, in order. Removing entries from the session leaves the archive's
    copies, so the archive keeps everything that was ever stored. The
    manager decides *when* to dump (``dump_every`` adds, a ``dump()``); this
    class does it.

    For the manager's async side, ``dump`` has an ``_async`` twin, using the
    storages' async twins.
    """

    def __init__(
        self,
        storage: BaseStorage,
        session_id: str,
        dump_storage: BaseStorage | None,
        every: int | None,
    ) -> None:
        """Copy the entries of ``session_id`` from ``storage`` to ``dump_storage``; ``every``: a dump is due after this many adds.

        Without a ``dump_storage`` there is no dump; with ``every=None``,
        only on demand.
        """
        self.storage = storage
        self.session_id = session_id
        self.dump_storage = dump_storage
        self.every = every
        self._dumped = (
            0  # how many stored entries the archive holds: always the first ones
        )
        self._adds = 0  # the app's adds since the last dump

    def read_back(self, index: list[tuple[str, EntryKind]]) -> None:
        """Rebuild from the session's ``index`` and the archive's: how many of the first stored entries it holds.

        For a session opened again, or rewound. The app's entries after
        those count as adds since the last dump.
        """
        archived = (
            {entry_id for entry_id, _ in self.dump_storage.index(self.session_id)}
            if self.dump_storage is not None
            else set()
        )
        self._dumped = next(
            (i for i, (entry_id, _) in enumerate(index) if entry_id not in archived),
            len(index),
        )
        self._adds = sum(kind not in MANAGED_KINDS for _, kind in index[self._dumped :])

    def added(self) -> None:
        """Count an entry the app added toward ``every``."""
        self._adds += 1

    @property
    def due(self) -> bool:
        """Whether a dump is due: ``every`` adds since the last one."""
        return self.every is not None and self._adds >= self.every

    def dump(self) -> list[Entry]:
        """Copy the entries not yet dumped to ``dump_storage``, in order; returns them.

        Everything stored in the session is copied, summaries, notes and pin
        records too; the session's storage is left as it is. The count
        toward ``every`` starts again. Raises ``ValueError`` without a
        ``dump_storage``.
        """
        dump_storage = self._dump_storage()
        index = self.storage.index(self.session_id)
        new = self.storage.get(self.session_id, self._new_ids(index))
        for entry in new:
            # The archive skips an entry it already holds (a repeat), so copying one again is safe.
            dump_storage.add(self.session_id, entry)
            self._dumped += 1  # one by one: if a copy fails, the next dump starts there
        self._adds = 0
        return new

    async def dump_async(self) -> list[Entry]:
        """``dump``, with the storages' async twins."""
        dump_storage = self._dump_storage()
        index = await self.storage.index_async(self.session_id)
        new = await self.storage.get_async(self.session_id, self._new_ids(index))
        for entry in new:
            await dump_storage.add_async(self.session_id, entry)
            self._dumped += 1
        self._adds = 0
        return new

    def forget(self, positions: list[int]) -> None:
        """Forget stored entries removed from the session, at ``positions`` in its index before they went.

        The archive keeps its copies, and still holds the first stored
        entries: fewer of them, if these were among them.
        """
        self._dumped -= sum(position < self._dumped for position in positions)

    def _dump_storage(self) -> BaseStorage:
        """The archive; raises ``ValueError`` without one."""
        if self.dump_storage is None:
            raise ValueError("this manager has no dump_storage")
        return self.dump_storage

    def _new_ids(self, index: list[tuple[str, EntryKind]]) -> list[str]:
        """The IDs in the session's ``index`` not yet dumped, in order."""
        return [entry_id for entry_id, _ in index[self._dumped :]]
