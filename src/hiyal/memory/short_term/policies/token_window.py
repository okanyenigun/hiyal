"""TokenWindowPolicy: keeps the newest entries within a token limit, cutting well below it."""

from collections.abc import Callable
from .base import WindowPolicy
from ..schemas import Entry
from ....utility import Checks, Tokens


def text_tokens(entry: Entry) -> int:
    """The tokens in ``entry.to_text()``, in tiktoken's default encoding.

    Media count only as their placeholders' text; pass ``TokenWindowPolicy``
    a counter that adds an estimate for them when entries carry media.
    """
    return Tokens.count(entry.to_text())


class TokenWindowPolicy(WindowPolicy):
    """The newest entries within ``max_tokens``; once they pass it, the oldest are dropped until ``target_tokens``.

    ``SlidingWindowPolicy`` in tokens: the start stays put while new entries
    arrive, and jumps forward only when the window passes ``max_tokens``,
    far enough to get it back to ``target_tokens``. The gap between the two
    means one cut every few entries instead of one per entry. With
    ``max_tokens=100_000, target_tokens=50_000``: the window grows 40k, 70k,
    98k; the entry that takes it to 103k drops the oldest until it is at
    most 50k; then it grows again.

    The cuts are replayed from the oldest entry, so the same entries always
    give the same window. The newest entry always stays, even when it alone
    passes ``max_tokens``.

    Each entry's size comes from ``count``: ``text_tokens`` by default, or
    the same counter the rest of the memory uses.
    """

    def __init__(
        self,
        max_tokens: int,
        target_tokens: int | None = None,
        *,
        count: Callable[[Entry], int] = text_tokens,
    ) -> None:
        """Keep the window within ``max_tokens``, cutting it to ``target_tokens`` (half of ``max_tokens`` when ``None``).

        Raises ``ValueError`` unless ``max_tokens`` is a positive integer and
        ``target_tokens`` a positive integer no larger than it, and
        ``TypeError`` if ``count`` is not callable.
        """
        self.max_tokens = Checks.positive_int(max_tokens, "max_tokens")
        if target_tokens is None:
            target_tokens = max(1, max_tokens // 2)
        self.target_tokens = Checks.positive_int(target_tokens, "target_tokens")
        if target_tokens > max_tokens:
            raise ValueError(
                f"target_tokens ({target_tokens}) cannot be larger than max_tokens ({max_tokens})"
            )
        if not callable(count):
            raise TypeError(f"count must be callable, got {count!r}")
        self.count = count

    def select(self, entries: list[Entry]) -> list[Entry]:
        """The entries from the window's start (see ``start``) on."""
        return entries[self.start(entries) :]

    def start(self, entries: list[Entry]) -> int:
        """The index of the first kept entry: the cuts replayed from the oldest entry on."""
        sizes = [self.count(entry) for entry in entries]
        start, total = 0, 0
        for newest, size in enumerate(sizes):
            total += size
            if total > self.max_tokens:
                while total > self.target_tokens and start < newest:
                    total -= sizes[start]
                    start += 1
        return start
