"""The manager's summary: what stands in for the entries that left the window as too old."""

from ..schemas import Entry, EntryKind, Summary
from ..storage import BaseStorage
from ..summarizers import SummaryWorker


class SummaryKeeper:
    """Keeps a session's summary for the ``ShortTermMemoryManager``: the current version, and the entries it covers.

    Each version is a summary entry in the session, made from the previous
    version and the entries that left since; the latest is the current
    one. Each version lists only the entries it added (``newly_covered``),
    so what the summary covers is what all the versions list. A covered
    entry stays in the history but leaves the pool for good. The manager
    decides *when* to summarize (entries left the window, a ``compact()``);
    this class does it.

    For the manager's async side, each method that writes or reads entries
    has an ``_async`` twin, using the async twins of the storage and the
    summarizer.
    """

    def __init__(
        self, storage: BaseStorage, session_id: str, summarizer: SummaryWorker | None
    ) -> None:
        """Keep the summary of ``session_id`` in ``storage``, made by ``summarizer``; none until ``read_back`` or ``summarize``."""
        self.storage = storage
        self.session_id = session_id
        self.summarizer = summarizer
        self._summary_id: str | None = None  # the current version
        self._covered: set[str] = set()  # the IDs all the versions cover

    def read_back(self, index: list[tuple[str, EntryKind]]) -> None:
        """Rebuild the summary from the session's ``index``: the latest version, and what all the versions cover.

        For a session opened again, or rewound. Only the summary versions are
        read.
        """
        ids = [entry_id for entry_id, kind in index if kind is EntryKind.SUMMARY]
        versions = self.storage.get(self.session_id, ids)
        self._summary_id = versions[-1].id_ if versions else None
        self._covered = {
            entry_id
            for version in versions
            if isinstance(version.payload, Summary)
            for entry_id in version.payload.newly_covered
        }

    def summarize(self, leaving: list[Entry]) -> Entry:
        """Summarize ``leaving`` after the current version, and store the result as the new one; returns it.

        The summarizer gets one text: the current version (if any), then the
        entries, each as it reads to a model. The entries are then covered.
        Raises ``ValueError`` without a ``summarizer``; when the summarizer
        fails, nothing changes, so the entries are still there for the next
        try.
        """
        summarizer = self._summarizer()
        text = summarizer.summarize(self._text(self.current, leaving))
        entry = self._version(text, leaving)
        self.storage.add(self.session_id, entry)
        self._made(entry, leaving)
        return entry

    async def summarize_async(self, leaving: list[Entry]) -> Entry:
        """``summarize``, with the async twins of the summarizer and the storage."""
        summarizer = self._summarizer()
        text = await summarizer.summarize_async(
            self._text(await self.current_async(), leaving)
        )
        entry = self._version(text, leaving)
        await self.storage.add_async(self.session_id, entry)
        self._made(entry, leaving)
        return entry

    def covers(self, entry_id: str) -> bool:
        """Whether the summary covers ``entry_id``, so it stays out of the pool."""
        return entry_id in self._covered

    def forget(self, entry_id: str) -> None:
        """Forget the removed entry ``entry_id``; a version already made from it keeps what it says."""
        self._covered.discard(entry_id)

    @property
    def current(self) -> Entry | None:
        """The current version, or ``None`` before the first; only that entry is read."""
        if self._summary_id is None:
            return None
        return self.storage.get(self.session_id, [self._summary_id])[0]

    async def current_async(self) -> Entry | None:
        """``current``, with the storage's async twins."""
        if self._summary_id is None:
            return None
        return (await self.storage.get_async(self.session_id, [self._summary_id]))[0]

    def _summarizer(self) -> SummaryWorker:
        """The summarizer; raises ``ValueError`` without one."""
        if self.summarizer is None:
            raise ValueError("this manager has no summarizer")
        return self.summarizer

    @staticmethod
    def _text(previous: Entry | None, leaving: list[Entry]) -> str:
        """The text to summarize: the ``previous`` version (if any), then the ``leaving`` entries, each as it reads to a model."""
        texts = [previous.to_text()] if previous is not None else []
        texts += [entry.to_text() for entry in leaving]
        return "\n\n".join(texts)

    @staticmethod
    def _version(text: str, leaving: list[Entry]) -> Entry:
        """A new version: the summary ``text``, covering the ``leaving`` entries."""
        return Entry.of(
            Summary(text=text, newly_covered=[left.id_ for left in leaving])
        )

    def _made(self, version: Entry, leaving: list[Entry]) -> None:
        """Make the stored ``version`` the current one; the ``leaving`` entries are now covered."""
        self._summary_id = version.id_
        self._covered.update(left.id_ for left in leaving)
