"""KindFilterPolicy: drops entries of chosen kinds."""

from collections.abc import Iterable
from .base import FilterPolicy
from ..schemas import Entry, EntryKind


class KindFilterPolicy(FilterPolicy):
    """Drops every entry whose kind is in ``exclude``; keeps the rest.

    For example ``KindFilterPolicy(exclude=[EntryKind.TOOL_RESULT])`` keeps
    verbose tool outputs out of the window.

    With a ``ShortTermMemoryManager``, system prompts and the manager's own
    entries (summaries, notes, pin records) never reach the policy, so
    excluding them has no effect.
    """

    def __init__(self, exclude: Iterable[EntryKind | str]) -> None:
        """Drop the entries of the kinds in ``exclude``, given as ``EntryKind`` or its value.

        Raises ``ValueError`` if ``exclude`` is empty or names an unknown kind.
        """
        self.exclude = frozenset(EntryKind(kind) for kind in exclude)
        if not self.exclude:
            raise ValueError("exclude needs at least one entry kind")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The entries whose kind is not excluded."""
        return [entry for entry in entries if entry.kind not in self.exclude]
