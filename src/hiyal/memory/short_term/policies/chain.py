"""Policy that applies several policies one after another."""

from .base import BasePolicy
from ..entry import Entry


class ChainPolicy(BasePolicy):
    """Runs ``policies`` one after another, each selecting from the previous one's result.

    This is how policies combine, for hybrid strategies::

        ChainPolicy([TypeFilterPolicy(exclude=[ToolResultEntry]), LastNPolicy(50)])

    **Filters run first**, whatever order they are given in (see
    ``BasePolicy.is_filter``): the filters in their given order, then the
    other policies in theirs. So a window always counts only what the filters
    kept, and the example gives the last 50 entries that survived the
    filter, also when written the other way round. It also keeps the window
    a ``ShortTermMemoryManager`` updates one entry at a time equal to choosing from
    the whole history; a window that counted entries a later filter removes
    would drift from it.
    """

    def __init__(self, policies: list[BasePolicy]) -> None:
        """Apply ``policies``, filters first; raises ``ValueError`` if it is empty."""
        if not policies:
            raise ValueError("a chain needs at least one policy")
        # Stable: filters keep their order among themselves, and so do the others.
        self.policies: list[BasePolicy] = [p for p in policies if p.is_filter] + [
            p for p in policies if not p.is_filter
        ]

    @property
    def is_filter(self) -> bool:
        """A chain is a filter when every policy in it is one."""
        return all(policy.is_filter for policy in self.policies)

    def filter(self, entries: list[Entry]) -> list[Entry]:
        """What the chain's filters keep of ``entries``; its windows are left out."""
        for policy in self.policies:
            if policy.is_filter:
                entries = policy.select(entries)
        return list(entries)

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Pass ``entries`` through every policy, filters first, and return the final result."""
        for policy in self.policies:
            entries = policy.select(entries)
        return list(entries)
