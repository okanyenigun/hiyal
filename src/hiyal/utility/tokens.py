import tiktoken
from functools import lru_cache


DEFAULT_ENCODING = "o200k_base"


class Tokens:
    """Helpers for counting tokens with tiktoken.

    ``o200k_base``, used by recent OpenAI models, is the default encoding.
    Other providers' models tokenize a little differently, so their counts
    are close, not exact.

    tiktoken downloads an encoding's file the first time it is used on a
    machine and caches it afterwards, so the first count needs network.
    """

    @staticmethod
    def count(text: str, encoding: str = DEFAULT_ENCODING) -> int:
        """The number of tokens in ``text``; raises ``ValueError`` for an unknown encoding."""
        return len(_load(Tokens.require_encoding(encoding)).encode(text))

    @staticmethod
    def to_words(tokens: float) -> int:
        """Roughly how many English words make ``tokens`` tokens (about 3 words per 4 tokens); at least 1."""
        return max(1, int(tokens * 3 / 4))

    @staticmethod
    def require_encoding(encoding: str) -> str:
        """``encoding`` when tiktoken knows it; raises ``ValueError`` otherwise. Needs no network."""
        names = tiktoken.list_encoding_names()
        if encoding not in names:
            raise ValueError(
                f"unknown encoding {encoding!r}; use one of {', '.join(names)}"
            )
        return encoding


@lru_cache
def _load(encoding: str) -> tiktoken.Encoding:
    """The encoding, loaded once per process."""
    return tiktoken.get_encoding(encoding)
