"""HeadTailPolicy: keeps the first entries and the last ones."""

from .base import WindowPolicy
from ..schemas import Entry
from ....utility import Checks


class HeadTailPolicy(WindowPolicy):
    """Keeps the first ``head`` entries and the last ``tail``; the middle is dropped.

    For a task stated at the start: the opening entries stay, while the rest
    moves forward like ``LastNPolicy(tail)``. With 1000 entries, ``head=2``
    and ``tail=250``, entries 1-2 and 751-1000 are kept.
    """

    def __init__(self, head: int, tail: int) -> None:
        """Keep the first ``head`` and the last ``tail`` entries; raises ``ValueError`` unless both are positive integers."""
        self.head = Checks.positive_int(head, "head")
        self.tail = Checks.positive_int(tail, "tail")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The first ``head`` entries and the last ``tail``, or all of them when there are no more."""
        if len(entries) <= self.head + self.tail:
            return list(entries)
        return [*entries[: self.head], *entries[-self.tail :]]
