"""Compaction that rebuilds the summary from the original entries."""

from .base import BaseCompaction, SummaryUpdate
from ..entry import Entry, SummaryEntry


class RebuildCompaction(BaseCompaction):
    """Sends all the original entries again: those the previous summary covers, then the new ones.

    Rebuilding from originals avoids summary drift, at the cost of sending
    more each time. Originals that are no longer available (e.g. removed,
    or moved to a dump) are replaced by the previous summary's text, sent
    as the previous summary.
    """

    def compact(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> SummaryUpdate:
        """Summarize ``covered`` and ``newly`` from scratch; fall back on ``previous`` for missing originals."""
        entries, fallback = self._inputs(newly, previous, covered)
        return self._update(
            self.summarizer.summarize(entries, fallback), previous, newly
        )

    async def compact_async(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> SummaryUpdate:
        """``compact``, with ``summarizer.summarize_async``."""
        entries, fallback = self._inputs(newly, previous, covered)
        return self._update(
            await self.summarizer.summarize_async(entries, fallback), previous, newly
        )

    def _inputs(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> tuple[list[Entry], SummaryEntry | None]:
        """What to summarize: every original, then ``previous`` only if some original is missing."""
        self._require_entries(newly)
        available = {entry.id_ for entry in covered}
        missing = previous is not None and any(
            entry_id not in available for entry_id in previous.covers
        )
        return [*covered, *newly], previous if missing else None
