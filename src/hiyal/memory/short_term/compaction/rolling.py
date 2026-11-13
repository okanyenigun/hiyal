"""Compaction that folds new entries into the previous summary."""

from .base import BaseCompaction, SummaryUpdate
from ..entry import Entry, SummaryEntry


class RollingCompaction(BaseCompaction):
    """Sends the previous summary and only the new entries.

    Each update is cheap, as it never resends what the summary already
    covers. The trade-off is drift: details the previous summary lost
    cannot come back.
    """

    uses_covered = False

    def compact(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> SummaryUpdate:
        """Fold ``newly`` into ``previous``; ``covered`` is not needed."""
        self._require_entries(newly)
        return self._update(self.summarizer.summarize(newly, previous), previous, newly)

    async def compact_async(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> SummaryUpdate:
        """``compact``, with ``summarizer.summarize_async``."""
        self._require_entries(newly)
        return self._update(
            await self.summarizer.summarize_async(newly, previous), previous, newly
        )
