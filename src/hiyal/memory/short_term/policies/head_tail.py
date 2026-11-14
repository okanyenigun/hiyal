"""Policy that keeps the first entries and the last ones active."""

from .base import BasePolicy
from ..entry import Entry
from ....utility import Checks


class HeadTailPolicy(BasePolicy):
    """The first ``head`` entries and the last ``tail`` ones are active; the middle leaves.

    For a task stated at the start: the opening entries stay, while the rest
    moves forward like ``LastNPolicy(tail)``. With 1000 entries, ``head=2``
    and ``tail=250``, entries 1-2 and 751-1000 are active.

    The head is the first entries the policy is given. In a
    ``ShortTermMemoryManager`` those are the oldest unpinned entries not yet
    summarized: the session's first ones, unless ``compact()`` summarized
    them or the session switched to this policy after they had left.

    Like ``LastNPolicy``, once full it drops one entry per add, so with a
    compaction every add is a compaction call.
    """

    def __init__(self, head: int, tail: int) -> None:
        """Keep the first ``head`` and the last ``tail`` entries; raises ``ValueError`` unless both are positive integers."""
        self.head = Checks.positive_int(head, "head")
        self.tail = Checks.positive_int(tail, "tail")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the first ``head`` entries and the last ``tail``, or all of them when there are no more."""
        if len(entries) <= self.head + self.tail:
            return list(entries)
        return [*entries[: self.head], *entries[-self.tail :]]
