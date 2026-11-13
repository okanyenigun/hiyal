"""Policy that keeps the whole history active."""

from .base import BasePolicy
from ..entry import Entry


class FullPolicy(BasePolicy):
    """Every entry is active.

    The simplest choice and the right one for short conversations: nothing
    is lost, but the active history grows with every call.
    """

    @property
    def is_filter(self) -> bool:
        """Always: every entry is kept, wherever it stands."""
        return True

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return all entries unchanged."""
        return list(entries)
