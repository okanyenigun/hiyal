"""Policy that keeps the most recent entries active."""

from .base import BasePolicy
from ..entry import Entry
from ....utility import Checks


class LastNPolicy(BasePolicy):
    """The last ``n`` entries are active.

    With 1000 entries and ``n=250``, entries 751-1000 are active. Each new
    entry pushes the oldest active one out, so the active history moves
    forward one entry at a time and always holds at most ``n`` entries.
    """

    def __init__(self, n: int) -> None:
        """Keep the last ``n`` entries; raises ``ValueError`` if ``n`` is not a positive integer."""
        self.n = Checks.positive_int(n, "n")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the last ``n`` entries, or all of them when there are fewer."""
        return list(entries[-self.n :])
