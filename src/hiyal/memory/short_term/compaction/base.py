"""The interface every compaction implements."""

import asyncio
from abc import ABC, abstractmethod
from typing import ClassVar
from ..summarizers import BaseSummarizer
from ..entry import Entry, SummaryEntry
from ....models.schemas import LLMResponse


class BaseCompaction(ABC):
    """Writes the text of the summary that replaces the current one, from the entries it is given.

    A worker: it gets plain data (the entries to add, the current summary,
    the original entries that summary covers) and returns the summarizer's
    response, whose text is the new summary. It knows no history and keeps
    no state. Deciding *when* to compact and *which* entries to add is the
    caller's job, and so is recording what the new summary covers, since
    that depends on the history; the ``ShortTermMemoryManager`` summarizes
    what leaves its window, and what ``compact()`` moves out of it.

    Compactions differ in what they send to the summarizer: the previous
    summary and only the new entries (``RollingCompaction``), or all the
    original entries again (``RebuildCompaction``).
    """

    # Whether ``compact`` reads ``covered``; when not, a caller can pass an
    # empty list instead of loading the previous summary's originals.
    uses_covered: ClassVar[bool] = True

    def __init__(self, summarizer: BaseSummarizer) -> None:
        """Use ``summarizer`` for the summary text."""
        self.summarizer = summarizer

    @abstractmethod
    def compact(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """The response whose text replaces ``previous``, now also covering ``newly``.

        - ``newly``: the entries to add to the summary, oldest first.
        - ``previous``: the current summary, or ``None`` for the first one.
        - ``covered``: the original entries ``previous`` covers that are
          still available, oldest first; empty for the first summary.

        Raises ``ValueError`` when ``newly`` is empty, or when the
        summarizer's response is not complete.
        """

    async def compact_async(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """``compact`` for async code; by default it runs ``compact`` in a worker thread.

        A compaction that can call its summarizer asynchronously overrides
        it (``RollingCompaction`` and ``RebuildCompaction`` do).
        """
        return await asyncio.to_thread(self.compact, newly, previous, covered)

    @staticmethod
    def _require_entries(newly: list[Entry]) -> None:
        """Raise ``ValueError`` when there is nothing new to summarize."""
        if not newly:
            raise ValueError("there is nothing to summarize yet")
