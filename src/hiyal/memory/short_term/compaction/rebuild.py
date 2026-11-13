"""Compaction that rebuilds the summary from the original entries."""

from .base import BaseCompaction
from ..entry import Entry, SummaryEntry
from ....models.schemas import LLMResponse


class RebuildCompaction(BaseCompaction):
    """Sends all the original entries again: those the previous summary covers, then the new ones.

    Rebuilding from originals avoids summary drift, at the cost of sending
    more each time. When some originals are no longer available (e.g.
    removed, or moved to a dump), the previous summary's text is sent too,
    as the previous summary.
    """

    def compact(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """Summarize ``covered`` and ``newly`` from scratch; fall back on ``previous`` for missing originals."""
        entries, fallback = self._inputs(newly, previous, covered)
        return self.summarizer.summarize(entries, fallback)

    async def compact_async(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """``compact``, with ``summarizer.summarize_async``."""
        entries, fallback = self._inputs(newly, previous, covered)
        return await self.summarizer.summarize_async(entries, fallback)

    def _inputs(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> tuple[list[Entry], SummaryEntry | None]:
        """What to summarize: every original, then ``previous`` only if some original is missing."""
        self._require_entries(newly)
        missing = previous is not None and len(covered) < previous.covers_count
        return [*covered, *newly], previous if missing else None
