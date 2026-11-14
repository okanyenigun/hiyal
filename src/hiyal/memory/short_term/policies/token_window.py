"""Policy that keeps the newest entries within a token limit, cutting well below it."""

from collections.abc import Mapping
from .base import BasePolicy
from ..entry import MEDIA_TOKENS, Entry, check_media_tokens, entry_tokens
from ....utility import Checks, Tokens
from ....utility.tokens import DEFAULT_ENCODING


class TokenWindowPolicy(BasePolicy):
    """The newest entries within ``max_tokens``; once they pass it, the oldest leave until ``target_tokens`` is reached.

    ``SlidingWindowPolicy`` in tokens: the start of the window stays put
    while new entries arrive, and jumps forward only when the window passes
    ``max_tokens``, far enough to get it back down to ``target_tokens``.
    The gap between the two means one cut (with a compaction, one summary)
    every few adds instead of on every add.

    Example with ``max_tokens=100_000, target_tokens=50_000``: the window
    grows 40k, 70k, 98k with nothing leaving; the add that takes it to
    103k drops the oldest entries until it is at most 50k; then it grows
    again until the next cut.

    The window is computed from the entries alone, by replaying the cuts
    from the oldest one, so the same entries always give the same window:
    one chosen add by add matches one chosen from the whole history (a
    continued session). The newest entry always stays, even when it alone
    passes ``max_tokens``; it leaves at the next cut.

    Tokens are counted as ``compact`` counts them: each entry's
    ``to_text()`` with ``encoding``, plus ``media_tokens`` for each attached
    image, audio clip and document. Entries the policy is not given
    (pinned ones, those a filter removed) do not count.
    """

    def __init__(
        self,
        max_tokens: int,
        target_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
        media_tokens: Mapping[str, int] = MEDIA_TOKENS,
    ) -> None:
        """Keep the window within ``max_tokens``, cutting it to ``target_tokens`` (half of ``max_tokens`` when ``None``).

        Raises ``ValueError`` unless ``max_tokens`` is a positive integer,
        ``target_tokens`` a positive integer no larger than it, ``encoding``
        known, and ``media_tokens`` valid (see ``compact``).
        """
        self.max_tokens = Checks.positive_int(max_tokens, "max_tokens")
        if target_tokens is None:
            target_tokens = max(1, max_tokens // 2)
        self.target_tokens = Checks.positive_int(target_tokens, "target_tokens")
        if target_tokens > max_tokens:
            raise ValueError(
                f"target_tokens ({target_tokens}) cannot be larger than max_tokens ({max_tokens})"
            )
        self.encoding = Tokens.require_encoding(encoding)
        self.media_tokens = check_media_tokens(media_tokens)

    def select(self, entries: list[Entry]) -> list[Entry]:
        """Return the entries inside the current window, oldest first."""
        return list(entries[self.start(entries) :])

    def start(self, entries: list[Entry]) -> int:
        """Index of the window's first entry: the cuts replayed from the oldest entry on."""
        start, total = 0, 0
        sizes = [entry_tokens(e, self.encoding, self.media_tokens) for e in entries]
        for newest, size in enumerate(sizes):
            total += size
            if total > self.max_tokens:
                while total > self.target_tokens and start < newest:
                    total -= sizes[start]
                    start += 1
        return start
