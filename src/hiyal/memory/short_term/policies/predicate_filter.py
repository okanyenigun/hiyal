"""PredicateFilterPolicy: keeps the entries a function says yes to."""

from collections.abc import Callable
from .base import FilterPolicy
from ..schemas import Entry


class PredicateFilterPolicy(FilterPolicy):
    """Keeps every entry ``keep`` returns true for; drops the rest.

    For any rule ``KindFilterPolicy`` cannot express, e.g. leaving out the
    responses that failed::

        PredicateFilterPolicy(
            lambda e: e.kind != EntryKind.LLM_RESPONSE or e.payload.status == "completed"
        )

    ``keep`` must decide by the entry alone, the same way every time: it is
    asked again whenever the window is chosen, and a resumed session must
    get the same answers.
    """

    def __init__(self, keep: Callable[[Entry], bool]) -> None:
        """Keep the entries ``keep`` returns true for; raises ``TypeError`` if it is not callable."""
        if not callable(keep):
            raise TypeError(f"keep must be callable, got {keep!r}")
        self.keep = keep

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The entries ``keep`` returns true for."""
        return [entry for entry in entries if self.keep(entry)]
