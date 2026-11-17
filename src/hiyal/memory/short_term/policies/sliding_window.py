"""SlidingWindowPolicy: keeps recent entries, moving forward in steps."""

from .base import WindowPolicy
from ..schemas import Entry
from ....utility import Checks


class SlidingWindowPolicy(WindowPolicy):
    """At most ``size`` entries, moving forward ``step`` entries at a time.

    The start stays put while new entries arrive; only when the window would
    pass ``size`` does it jump forward, dropping the oldest ``step`` entries
    at once. With ``size=250, step=50``: entries 1-250 are kept; entry 251
    moves the start to 51; it stays there until entry 301 arrives.

    Why steps: providers cache the start of a request, and a start that
    moves rarely keeps that cache matching; with a summary, it also means
    one summary call per ``step`` entries. ``step=1`` is ``LastNPolicy(size)``.
    """

    def __init__(self, size: int, step: int) -> None:
        """Keep at most ``size`` entries, jumping ``step`` at a time.

        Raises ``ValueError`` unless both are positive integers and ``step <= size``.
        """
        self.size = Checks.positive_int(size, "size")
        self.step = Checks.positive_int(step, "step")
        if step > size:
            raise ValueError(f"step ({step}) cannot be larger than size ({size})")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The entries from the window's start (see ``start``) on."""
        return entries[self.start(len(entries)) :]

    def start(self, count: int) -> int:
        """The index of the first kept entry, out of ``count``: 0 up to ``size``, then the smallest multiple of ``step`` that leaves at most ``size``."""
        excess = count - self.size
        if excess <= 0:
            return 0
        return -(-excess // self.step) * self.step  # ceiling division, in steps
