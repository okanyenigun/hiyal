"""The interface every active-history policy implements."""

from abc import ABC, abstractmethod
from ..entry import Entry


class BasePolicy(ABC):
    """Decides which part of the full history is active.

    A policy only *selects*: it receives every entry in order and returns a
    subset, never new or changed entries. It holds no entries itself, so the
    active history is always recomputed from the full one and can never get
    out of sync with it.

    A policy does not need to keep pairs together (an input and its
    response, a tool call and its result): the history repairs broken pairs
    after the policy has chosen.
    """

    @property
    def is_filter(self) -> bool:
        """Whether this policy keeps or drops each entry on its own, whatever its position.

        A filter (``TypeFilterPolicy``, ``FullPolicy``) gives the same answer
        for an entry wherever it stands. A window (``LastNPolicy``,
        ``SlidingWindowPolicy``) depends on position and count. ``ChainPolicy``
        runs filters before windows, so a window only counts what the filters
        kept.
        """
        return False

    def filter(self, entries: list[Entry]) -> list[Entry]:
        """What this policy's filter part keeps of ``entries``: a filter's choice, or all of them for a window.

        Entries a filter drops are "not wanted"; entries a window drops are
        "too old". The ``ShortTermMemoryManager`` uses this to summarize only
        the latter.
        """
        return self.select(entries) if self.is_filter else list(entries)

    @abstractmethod
    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the active entries, oldest first.

        ``entries`` is the full history, oldest first. The result keeps
        their order and must not modify them.
        """
