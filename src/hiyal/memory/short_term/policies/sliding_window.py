"""Policy that keeps a window of recent entries active, moving it in steps."""

from .base import BasePolicy
from ..entry import Entry
from ....utility import Checks


class SlidingWindowPolicy(BasePolicy):
    """A window of at most ``size`` entries that moves forward ``step`` entries at a time.

    Unlike ``LastNPolicy``, the start of the window stays put while new
    entries arrive. Only when the window would exceed ``size`` does it jump
    forward, dropping the oldest ``step`` entries at once.

    Example with ``size=250, step=50``: entries 1-250 are active; adding
    entry 251 moves the start to entry 51 (entries 51-251 are active); the
    start then stays at 51 until entry 301 arrives.

    Why steps: providers cache the beginning of a request. When the oldest
    active entry changes on every call (as with ``LastNPolicy``), that cache
    never matches. A start that moves rarely keeps it matching between
    jumps. ``step=1`` behaves like ``LastNPolicy(size)``.

    The window is computed from the number of entries alone, so the same
    history always gives the same window.
    """

    def __init__(self, size: int, step: int) -> None:
        """Set the window's maximum size and how far it jumps.

        Raises ``ValueError`` unless both are integers and ``0 < step <= size``.
        """
        self.size = Checks.positive_int(size, "size")
        self.step = Checks.positive_int(step, "step")
        if step > size:
            raise ValueError(f"step ({step}) cannot be larger than size ({size})")

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the entries inside the current window, oldest first."""
        return list(entries[self.start(len(entries)) :])

    def start(self, count: int) -> int:
        """Index of the window's first entry when the history holds ``count`` entries.

        Zero until the history exceeds ``size``; then the smallest multiple
        of ``step`` that leaves at most ``size`` entries.
        """
        excess = count - self.size
        if excess <= 0:
            return 0
        jumps = -(-excess // self.step)  # ceiling division
        return jumps * self.step
