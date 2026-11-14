"""Compaction that rolls, and rebuilds from the original entries every few versions."""

from .base import BaseCompaction
from .rebuild import RebuildCompaction
from .rolling import RollingCompaction
from ..entry import Entry, SummaryEntry
from ..summarizers import BaseSummarizer
from ....models.schemas import LLMResponse
from ....utility import Checks


class HybridCompaction(BaseCompaction):
    """Rolls like ``RollingCompaction``, and rebuilds like ``RebuildCompaction`` every ``rebuild_every`` versions.

    Rolling is cheap but drifts: each summary is a summary of the previous
    one, so what one loses never comes back. Rebuilding has no drift, but
    resends every original, so each update costs more than the last. This
    one rolls, and makes every ``rebuild_every``-th summary from the
    originals: drift lasts at most ``rebuild_every - 1`` rolls, and the
    rebuild's cost comes once per ``rebuild_every`` summaries. A long
    rebuild relies on the summarizer's chunking to fit.

    With ``rebuild_every=3``: versions 1 and 2 roll, 3 is rebuilt, 4 and 5
    roll, 6 is rebuilt, and so on. Which round it is comes from the stored
    summary's ``version``, so a continued session keeps the rhythm.
    """

    def __init__(self, summarizer: BaseSummarizer, rebuild_every: int) -> None:
        """Use ``summarizer``, rebuilding every ``rebuild_every`` versions; raises ``ValueError`` unless it is a positive integer."""
        super().__init__(summarizer)
        self.rebuild_every = Checks.positive_int(rebuild_every, "rebuild_every")
        self._rolling = RollingCompaction(summarizer)
        self._rebuild = RebuildCompaction(summarizer)

    def rebuilds(self, previous: SummaryEntry | None) -> bool:
        """Whether the summary after ``previous`` is rebuilt: when its version is a multiple of ``rebuild_every``."""
        version = previous.version + 1 if previous is not None else 1
        return version % self.rebuild_every == 0

    def needs_covered(self, previous: SummaryEntry | None) -> bool:
        """Only on a rebuild round."""
        return self.rebuilds(previous)

    def compact(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """Rebuild from ``covered`` and ``newly`` on a rebuild round; otherwise fold ``newly`` into ``previous``."""
        worker = self._rebuild if self.rebuilds(previous) else self._rolling
        return worker.compact(newly, previous, covered)

    async def compact_async(
        self, newly: list[Entry], previous: SummaryEntry | None, covered: list[Entry]
    ) -> LLMResponse:
        """``compact``, with ``summarizer.summarize_async``."""
        worker = self._rebuild if self.rebuilds(previous) else self._rolling
        return await worker.compact_async(newly, previous, covered)
