"""ChunkedSummarizer: shrinks a text too long for one call by summarizing it chunk by chunk."""

import asyncio
from .base import LLMSummaryWorker
from ....models.bases import BaseLLM
from ....prompts.hub import hub
from ....utility import Checks, Tokens
from ....utility.tokens import DEFAULT_ENCODING


class ChunkedSummarizer(LLMSummaryWorker):
    """Summarizes the text chunk by chunk, round after round, until it fits ``chunk_tokens``.

    One round: split the text into chunks, summarize each, join the
    summaries. While the joined summaries are still too big, it does
    another round on them::

        text (50k)   [8k][8k][8k][8k][8k][8k][2k]   split into 7 chunks
        round 1      [2k][2k][2k][2k][2k][2k][2k]   each summarized, joined: 14k > 8k
        round 2      [2k 2k 2k 2k][2k 2k 2k]        split again into 2 chunks
                     [2k][2k]                       each summarized, joined: 4k <= 8k, done

    It only shrinks; a ``SingleCallSummarizer`` after it writes the final
    summary.

    - **Split:** at the first of ``separators``; a piece too big for a
      chunk at the next one, and past the last one anywhere. By default:
      at paragraphs, a paragraph at its lines, a line at its words, and a
      word (a long run with no spaces) anywhere. Neighbouring pieces are packed, in order, into chunks of
      about ``chunk_tokens``.
    - **Length:** every call aims at ``chunk_aim`` of ``chunk_tokens``, so
      the summaries of several chunks fit one chunk in the next round.
    - **A short chunk is kept:** a chunk already within that aim is not
      summarized.
    - **Stall guard:** a round that shrinks the text by less than
      ``min_decrease`` is the last one: another round is unlikely to get
      further. The shortest text reached is returned, which may still be
      over ``chunk_tokens``.
    - **Async:** ``summarize_async`` summarizes the chunks of a round at the
      same time, and joins them in order.

    A text that already fits passes on unchanged, with no call.
    """

    def __init__(
        self,
        llm: BaseLLM,
        chunk_tokens: int,
        *,
        chunk_aim: float = 0.25,
        min_decrease: float = 0.1,
        prompt_name: str = "memory/summarize",
        prompt_version: str = "v1",
        separators: tuple[str, ...] | list[str] = ("\n\n", "\n", " "),
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """Shrink the text to fit ``chunk_tokens``, the most text one call may take.

        - ``chunk_aim``: each call aims at this share of ``chunk_tokens``.
        - ``min_decrease``: the least share a round must shrink the text by
          for another round to follow.
        - ``prompt_name``, ``prompt_version``: the hub prompt for each chunk,
          as for ``SingleCallSummarizer``.
        - ``separators``: where the text is split, the biggest boundary
          first (see the class).
        - ``encoding``: the tiktoken encoding for every count.

        Raises ``ValueError`` for a ``chunk_tokens`` that is not a positive
        integer, a ``chunk_aim`` outside ``(0, 1]``, a ``min_decrease``
        outside ``[0, 1)``, an empty separator, or an unknown encoding;
        ``TypeError`` for ``separators`` that is not a tuple or list of
        strings.
        """
        super().__init__(llm)
        self.chunk_tokens = Checks.positive_int(chunk_tokens, "chunk_tokens")
        self.chunk_aim = self._share(chunk_aim, "chunk_aim")
        self.min_decrease = self._decrease(min_decrease)
        self.instructions = hub.get(prompt_name, prompt_version)
        self.separators = self._separators(separators)
        self.encoding = Tokens.require_encoding(encoding)

    def _summarize(self, text: str) -> str:
        """``text`` shrunk round after round until it fits ``chunk_tokens``, or a round stalls (see the class)."""
        while self._count(text) > self.chunk_tokens:
            chunks = self._chunks(text)
            shrunk = "\n\n".join(
                self._summarize_chunk(chunk, number, len(chunks))
                for number, chunk in enumerate(chunks, 1)
            )
            text, stalled = self._after_round(text, shrunk)
            if stalled:
                break
        return text

    async def _summarize_async(self, text: str) -> str:
        """``_summarize``, on the LLM's async client: the chunks of a round are summarized at the same time."""
        while self._count(text) > self.chunk_tokens:
            chunks = self._chunks(text)
            summaries = await asyncio.gather(
                *(
                    self._summarize_chunk_async(chunk, number, len(chunks))
                    for number, chunk in enumerate(chunks, 1)
                )
            )
            text, stalled = self._after_round(text, "\n\n".join(summaries))
            if stalled:
                break
        return text

    def _summarize_chunk(self, chunk: str, number: int, total: int) -> str:
        """The summary of one chunk, or the chunk itself when it is already within the aim."""
        request = self._request(chunk, number, total)
        return chunk if request is None else self._ask(*request).strip()

    async def _summarize_chunk_async(self, chunk: str, number: int, total: int) -> str:
        """``_summarize_chunk``, on the LLM's async client."""
        request = self._request(chunk, number, total)
        return chunk if request is None else (await self._ask_async(*request)).strip()

    def _request(self, chunk: str, number: int, total: int) -> tuple[str, str] | None:
        """The ``(instructions, prompt)`` asking to summarize chunk ``number`` of ``total``; ``None`` when it is within the aim, so it is kept."""
        aim = self.chunk_aim * self.chunk_tokens
        if self._count(chunk) <= aim:
            return None
        header = (
            f"Part {number} of {total} of a longer conversation; the other parts are "
            "summarized separately. Conversation to summarize:"
            if total > 1
            else "Conversation to summarize:"
        )
        return (
            f"{self.instructions} Aim for about {Tokens.to_words(aim)} words.",
            f"{header}\n{chunk}",
        )

    def _chunks(self, text: str) -> list[str]:
        """``text`` as chunks of about ``chunk_tokens``: its pieces (see ``_split``), packed in order."""
        chunks: list[str] = []
        current, used = "", 0
        for piece in self._split(text):
            size = self._count(piece)
            if current and used + size > self.chunk_tokens:
                chunks.append(current)
                current, used = "", 0
            current += piece
            used += size
        chunks.append(current)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split(self, text: str, level: int = 0) -> list[str]:
        """``text`` in pieces of at most ``chunk_tokens``, each keeping the separator after it.

        Cut at ``separators[level]``; a piece still too big is cut at the
        next separator, and past the last one anywhere (see ``_cut``).
        Joined, the pieces are ``text`` again.
        """
        if self._count(text) <= self.chunk_tokens:
            return [text]
        if level == len(self.separators):
            return self._cut(text)
        separator = self.separators[level]
        parts = text.split(separator)
        pieces: list[str] = []
        for index, part in enumerate(parts):
            if index < len(parts) - 1:
                part += separator
            if part:
                pieces += self._split(part, level + 1)
        return pieces

    def _cut(self, text: str) -> list[str]:
        """``text``, which has no separator to cut at, in runs of at most ``chunk_tokens``."""
        pieces = []
        while self._count(text) > self.chunk_tokens:
            end = max(1, self._longest_start(text))
            pieces.append(text[:end])
            text = text[end:]
        return [*pieces, text] if text else pieces

    def _longest_start(self, text: str) -> int:
        """The length of the longest start of ``text`` within ``chunk_tokens``; a binary search on the length."""
        low, high = (
            0,
            len(text),
        )  # the start of length ``low`` fits; of length ``high`` does not
        while high - low > 1:
            middle = (low + high) // 2
            if self._count(text[:middle]) <= self.chunk_tokens:
                low = middle
            else:
                high = middle
        return low

    def _after_round(self, before: str, after: str) -> tuple[str, bool]:
        """What a round leaves: the shorter of ``before`` and ``after``, and whether it stalled (shrank by less than ``min_decrease``)."""
        old, new = self._count(before), self._count(after)
        shorter = after if new < old else before
        return shorter, new >= old or (old - new) / old < self.min_decrease

    @staticmethod
    def _separators(value: object) -> tuple[str, ...]:
        """``value`` as a tuple of non-empty strings; raises ``TypeError`` or ``ValueError`` otherwise.

        A string alone is refused: it would be read as separators of one
        character each.
        """
        if isinstance(value, str) or not isinstance(value, (tuple, list)):
            raise TypeError(
                f"separators must be a tuple or list of strings, got {type(value).__name__}"
            )
        return tuple(Checks.non_empty_string(separator, "a separator") for separator in value)

    def _count(self, text: str) -> int:
        """Tokens in ``text``, in this worker's encoding."""
        return Tokens.count(text, self.encoding)
