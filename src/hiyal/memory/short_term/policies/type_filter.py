"""Policy that removes entries of chosen types."""

from .base import BasePolicy
from ..entry import Entry


class TypeFilterPolicy(BasePolicy):
    """Removes every entry of the excluded types; keeps everything else.

    For example ``TypeFilterPolicy(exclude=[ToolResultEntry])`` keeps
    verbose tool outputs out of the active history. The history then also
    drops the responses whose tool calls they answered, so no call is left
    without its result.

    Removed entries stay in the full history; changing the policy brings
    them back. Combine it with a window through ``ChainPolicy``.
    """

    def __init__(self, exclude: list[type[Entry]]) -> None:
        """Remove entries that are instances of any type in ``exclude``.

        Raises ``ValueError`` if ``exclude`` is empty.
        """
        if not exclude:
            raise ValueError("exclude needs at least one entry type")
        self.exclude: tuple[type[Entry], ...] = tuple(exclude)

    @property
    def is_filter(self) -> bool:
        """Always: each entry is kept or dropped by its type alone."""
        return True

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the entries whose type is not excluded."""
        return [entry for entry in entries if not isinstance(entry, self.exclude)]
