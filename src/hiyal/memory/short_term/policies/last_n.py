"""LastNPolicy: keeps the most recent entries."""

from .base import WindowPolicy
from ..schemas import Entry
from ....utility import Checks


class LastNPolicy(WindowPolicy):
    """Keeps the last ``n`` entries.

    With 1000 entries and ``n=250``, entries 751-1000 are kept. Once full,
    each new entry pushes the oldest one out, so with a summary every new
    entry means a summary call. ``SlidingWindowPolicy`` drops entries in
    batches instead.
    """

    def __init__(self, n: int) -> None:
        """Keep the last ``n`` entries; raises ``ValueError`` if ``n`` is not a positive integer."""
        self.n = Checks.positive_int(n, "n")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The last ``n`` entries, or all of them when there are fewer."""
        return entries[-self.n :]
