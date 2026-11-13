"""Policy that keeps the most recent entries active."""

from .base import BasePolicy
from ..entry import Entry
from ....utility import Checks


class LastNPolicy(BasePolicy):
    """The last ``n`` entries are active.

    With 1000 entries and ``n=250``, entries 751-1000 are active. Each new
    entry pushes the oldest active one out, so the active history moves
    forward one entry at a time and always holds at most ``n`` entries.

    With a compaction, avoid it: once full, every add drops one entry, so
    every add is a compaction call (an LLM call) and a new summary that
    breaks the provider's prompt cache. ``SlidingWindowPolicy`` drops
    entries in batches, so it summarizes once every ``step`` adds.
    """

    def __init__(self, n: int) -> None:
        """Keep the last ``n`` entries; raises ``ValueError`` if ``n`` is not a positive integer."""
        self.n = Checks.positive_int(n, "n")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the last ``n`` entries, or all of them when there are fewer."""
        return list(entries[-self.n :])
