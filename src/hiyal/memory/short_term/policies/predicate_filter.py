"""Policy that keeps the entries a function says yes to."""

from collections.abc import Callable
from .base import BasePolicy
from ..entry import Entry


class PredicateFilterPolicy(BasePolicy):
    """Keeps every entry ``keep`` returns true for; removes the rest.

    For any rule ``TypeFilterPolicy`` cannot express, e.g. leaving out the
    responses that failed::

        PredicateFilterPolicy(
            lambda e: not isinstance(e, LLMResponse) or e.status == "completed"
        )

    ``keep`` must decide by the entry alone, the same way every time: it is
    asked again whenever the window is chosen, and a continued session must
    get the same answers. Removed entries stay in the full history and are
    not summarized (a filter means "not wanted"); changing the policy brings
    them back. Combine it with a window through ``ChainPolicy``.
    """

    def __init__(self, keep: Callable[[Entry], bool]) -> None:
        """Keep the entries ``keep`` returns true for; raises ``TypeError`` if it is not callable."""
        if not callable(keep):
            raise TypeError(f"keep must be callable, got {keep!r}")
        self.keep = keep

    @property
    def is_filter(self) -> bool:
        """Always: ``keep`` decides each entry on its own."""
        return True

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the entries ``keep`` returns true for."""
        return [entry for entry in entries if self.keep(entry)]
