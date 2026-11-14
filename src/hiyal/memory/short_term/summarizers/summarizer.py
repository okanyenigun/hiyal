"""One summarizer whose behaviours are switched on by its parameters."""

import asyncio
from .base import BaseSummarizer
from ..entry import Entry, SummaryEntry
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt, UserPrompt
from ....utility import Checks, Tokens
from ....utility.tokens import DEFAULT_ENCODING
from ....prompts.hub import hub


class Summarizer(BaseSummarizer):
    """Summarizes entries as text; focus, chunking, a token budget and a cap per entry are switched on by parameters.

    ==========  ==================  ============================================
    Behaviour   Switched on by      What it adds
    ==========  ==================  ============================================
    text        (always)            each entry's ``to_text()``, in one call
    cap         ``max_entry_chars`` each entry's text cut to this length, so
                                    one huge entry cannot fill the call
    focus       ``focus``           what the summary is for, in every call's
                                    system prompt
    chunking    ``chunk_tokens``    map-reduce when the text is too long for
                                    one call
    budget      ``max_tokens``      a length target in every call, then
                                    shortening until the summary fits
    ==========  ==================  ============================================

    With none of them it is one call: the instructions as the system prompt,
    and the rendered entries (see ``render``). The pipeline, each step only
    when switched on::

        render text -> one call, or chunk + combine -> fit the budget

    Every call is text only. Media are in the text as their placeholders,
    with their ``description`` (and an audio's ``transcript``) when set;
    filling descriptions is not this class's job.

    Length targets:

    - With chunking, the intermediate calls (each chunk's summary, each
      combine that is not the last) aim at ``chunk_aim`` of ``chunk_tokens``,
      so several summaries fit one combine. Models overshoot it; when no two
      summaries fit one combine, the ones over the target are shortened
      first, until they fit or shortening stalls (see ``_combine``).
    - With a budget, every call aims at ``budget_aim`` of ``max_tokens`` (below it,
      as models overshoot); intermediate calls take the smaller of the two.
    - The final call has a target only with a budget.

    ``summarize`` returns one response: the last call's that counts (with a
    budget, the first that fits, or else the shortest).
    """

    def __init__(
        self,
        llm: BaseLLM,
        instructions: str | None = None,
        *,
        prompt_name: str = "memory/summarize",
        prompt_version: str = "v1",
        combine_version: str = "v1",
        shorten_version: str = "v1",
        focus: str | None = None,
        chunk_tokens: int | None = None,
        max_tokens: int | None = None,
        max_rounds: int = 3,
        min_decrease: float = 0.1,
        budget_aim: float = 0.75,
        chunk_aim: float = 0.25,
        max_entry_chars: int | None = None,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """Configure the behaviours; every keyword left out leaves its behaviour off.

        - ``llm``: makes every call; a text model is enough.
        - ``instructions``: replace the hub's prompt for summarizing entries;
          combining and shortening keep their hub prompts.
        - ``prompt_name``, ``prompt_version``: the hub prompt for
          summarizing entries, when no ``instructions`` are given.
          ``"memory/summarize"`` covers everything; with a note taker,
          ``"memory/summarize_with_notes"`` records what happened and
          leaves the current state to the notes, so the two do not repeat
          each other.
        - ``combine_version``, ``shorten_version``: the versions of the hub
          prompts ``memory/combine`` and ``memory/shorten``. Match them to
          the summary's shape: ``v1`` writes prose; ``v2`` keeps the
          headings of ``memory/summarize`` ``v2``, so use all three together.
        - ``focus``: what the summary is for; can be set again later.
        - ``chunk_tokens``: the most text one call may take; longer input is
          chunked and combined.
        - ``max_tokens``: the most the summary may be; ``max_rounds`` and
          ``min_decrease`` stop the shortening (see ``_fit_budget``).
        - ``budget_aim``: with ``max_tokens``, each call aims at this share
          of it, since models overshoot length targets.
        - ``chunk_aim``: intermediate calls (chunk summaries, partial
          combines) aim at this share of ``chunk_tokens``, so several of
          their summaries fit one combine call and each combine level
          shrinks the pieces.
        - ``max_entry_chars``: the most characters of one entry's text sent
          to the model; the rest is cut, with a note saying how much (only
          what is sent: the entry itself stays whole). For example ``2000``,
          so one huge tool output cannot fill a summary call or the summary.
        - ``encoding``: the tiktoken encoding for every count.

        Raises ``ValueError`` for a blank focus, a ``chunk_tokens``,
        ``max_tokens`` or ``max_entry_chars`` that is not a positive integer, a negative
        ``max_rounds``, a ``min_decrease`` outside ``[0, 1)``, a
        ``budget_aim`` or ``chunk_aim`` outside ``(0, 1]``, or an unknown
        encoding.
        """
        for name, value in (
            ("chunk_tokens", chunk_tokens),
            ("max_tokens", max_tokens),
            ("max_entry_chars", max_entry_chars),
        ):
            if value is not None:
                Checks.positive_int(value, name)
        if (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or max_rounds < 0
        ):
            raise ValueError(
                f"max_rounds must be a non-negative integer, got {max_rounds!r}"
            )
        if (
            isinstance(min_decrease, bool)
            or not isinstance(min_decrease, (int, float))
            or not 0 <= min_decrease < 1
        ):
            raise ValueError(
                f"min_decrease must be a number in [0, 1), got {min_decrease!r}"
            )
        for name, value in (("budget_aim", budget_aim), ("chunk_aim", chunk_aim)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be a number in (0, 1], got {value!r}")
        super().__init__(llm, instructions, prompt_name, prompt_version)
        self.combine_instructions = hub.get("memory/combine", combine_version)
        self.shorten_instructions = hub.get("memory/shorten", shorten_version)
        self.focus = focus
        self.chunk_tokens = chunk_tokens
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self.min_decrease = min_decrease
        self.budget_aim = budget_aim
        self.chunk_aim = chunk_aim
        self.max_entry_chars = max_entry_chars
        self.encoding = Tokens.require_encoding(encoding)

    @property
    def focus(self) -> str | None:
        """What the summaries are for, or ``None``; set it again when the task changes."""
        return self._focus

    @focus.setter
    def focus(self, focus: str | None) -> None:
        if focus is not None and (not isinstance(focus, str) or not focus.strip()):
            raise ValueError(f"focus must be None or a non-blank string, got {focus!r}")
        self._focus = focus.strip() if focus is not None else None

    # ------------------------------------------------------------------ the pipeline

    def summarize(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> LLMResponse:
        """Summarize ``entries`` (folding in ``previous``) with every behaviour that is on.

        Raises ``ValueError`` when there are no entries, a response is not
        complete (e.g. cut off by ``max_output_tokens``), or the chunk
        summaries cannot be shortened enough to combine within ``chunk_tokens``.
        """
        if not entries:
            raise ValueError("there are no entries to summarize")
        prompt = self.render(entries, previous)
        if self.chunk_tokens is None or self._count(prompt.text) <= self.chunk_tokens:
            response = self.query(
                self.system("summarize", self._aim(final=True)), prompt
            )
        else:
            response = self._map_reduce(entries, previous, prompt)
        if self.max_tokens is not None:
            response = self._fit_budget(response)
        return response

    async def summarize_async(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> LLMResponse:
        """``summarize``, with ``llm.query_async``.

        Calls that do not depend on each other run at the same time: the
        chunk summaries, the groups of one combine level, and the pieces
        shortened before a combine. Their results keep their order, so the
        summary is built exactly as ``summarize`` builds it.
        """
        if not entries:
            raise ValueError("there are no entries to summarize")
        prompt = self.render(entries, previous)
        if self.chunk_tokens is None or self._count(prompt.text) <= self.chunk_tokens:
            response = await self.query_async(
                self.system("summarize", self._aim(final=True)), prompt
            )
        else:
            response = await self._map_reduce_async(entries, previous, prompt)
        if self.max_tokens is not None:
            response = await self._fit_budget_async(response)
        return response

    def render(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> UserPrompt:
        """The previous summary, then each entry's text (see ``_text``); media are their placeholders.

        The prompt reads::

            Previous summary:
            <the previous summary, when rolling>

            Conversation to summarize:
            <each entry's text, separated by blank lines>
        """
        parts = [f"Previous summary:\n{previous.text}"] if previous is not None else []
        conversation = "\n\n".join(self._text(entry) for entry in entries)
        parts.append(f"Conversation to summarize:\n{conversation}")
        return UserPrompt(text="\n\n".join(parts))

    def _text(self, entry: Entry) -> str:
        """``entry.to_text()``, cut to ``max_entry_chars`` when it is set, saying how much was cut."""
        text = entry.to_text()
        if self.max_entry_chars is None or len(text) <= self.max_entry_chars:
            return text
        cut = len(text) - self.max_entry_chars
        return f"{text[: self.max_entry_chars]}... ({cut:,} more characters cut)"

    # ------------------------------------------------------------------ chunking

    def _map_reduce(
        self, entries: list[Entry], previous: SummaryEntry | None, whole: UserPrompt
    ) -> LLMResponse:
        """Summarize each chunk of ``entries``, then combine the chunk summaries (``previous`` first)."""
        spans = self._spans([self._count(self._text(entry)) for entry in entries])
        if len(spans) == 1 and previous is None:
            # The entries fit one chunk, only the prompt header tips it over: one call is still right.
            return self.query(self.system("summarize", self._aim(final=True)), whole)
        pieces = [previous.text] if previous is not None else []
        for number, (start, end) in enumerate(spans, 1):
            conversation = "\n\n".join(self._text(entry) for entry in entries[start:end])
            part = UserPrompt(
                text=f"Part {number} of {len(spans)} of a longer conversation; the other parts are "
                f"summarized separately. Conversation to summarize:\n{conversation}"
            )
            pieces.append(
                self.query(self.system("summarize", self._aim(final=False)), part).text
            )
        return self._combine(pieces)

    async def _map_reduce_async(
        self, entries: list[Entry], previous: SummaryEntry | None, whole: UserPrompt
    ) -> LLMResponse:
        """``_map_reduce``, with the chunks summarized at the same time."""
        spans = self._spans([self._count(self._text(entry)) for entry in entries])
        if len(spans) == 1 and previous is None:
            return await self.query_async(
                self.system("summarize", self._aim(final=True)), whole
            )
        system = self.system("summarize", self._aim(final=False))
        parts = []
        for number, (start, end) in enumerate(spans, 1):
            conversation = "\n\n".join(self._text(entry) for entry in entries[start:end])
            parts.append(
                UserPrompt(
                    text=f"Part {number} of {len(spans)} of a longer conversation; the other parts are "
                    f"summarized separately. Conversation to summarize:\n{conversation}"
                )
            )
        summaries = await asyncio.gather(
            *(self.query_async(system, part) for part in parts)
        )
        pieces = ([previous.text] if previous is not None else []) + [
            summary.text for summary in summaries
        ]
        return await self._combine_async(pieces)

    def _combine(self, pieces: list[str]) -> LLMResponse:
        """Combine summaries, oldest first, into one; in groups, level by level, when they do not fit one call.

        When no two neighbours fit one call, the pieces over the target are
        shortened first (``_shorten_pieces``) instead of giving up.
        """
        while True:
            spans = self._spans([self._count(piece) for piece in pieces])
            if len(spans) == 1:
                return self.query(
                    self.system("combine", self._aim(final=True)),
                    self._render_pieces(pieces),
                )
            if len(spans) == len(pieces):
                pieces = self._shorten_pieces(
                    pieces
                )  # no level could shrink them: shorten, then try again
                continue
            # One level: combine each group of neighbours; a group of one passes through unchanged.
            system = self.system("combine", self._aim(final=False))
            pieces = [
                self.query(system, self._render_pieces(pieces[start:end])).text
                if end - start > 1
                else pieces[start]
                for start, end in spans
            ]

    async def _combine_async(self, pieces: list[str]) -> LLMResponse:
        """``_combine``, with the groups of one level combined at the same time."""
        while True:
            spans = self._spans([self._count(piece) for piece in pieces])
            if len(spans) == 1:
                return await self.query_async(
                    self.system("combine", self._aim(final=True)),
                    self._render_pieces(pieces),
                )
            if len(spans) == len(pieces):
                pieces = await self._shorten_pieces_async(pieces)
                continue
            system = self.system("combine", self._aim(final=False))

            async def combined(start: int, end: int) -> str:
                if end - start == 1:
                    return pieces[start]
                return (
                    await self.query_async(
                        system, self._render_pieces(pieces[start:end])
                    )
                ).text

            pieces = list(
                await asyncio.gather(*(combined(start, end) for start, end in spans))
            )

    def _spans(self, sizes: list[int]) -> list[tuple[int, int]]:
        """``(start, end)`` runs of items, in order, of about ``chunk_tokens`` each; at least one item per run."""
        assert self.chunk_tokens is not None
        spans: list[tuple[int, int]] = []
        start, used = 0, 0
        for index, size in enumerate(sizes):
            if index > start and used + size > self.chunk_tokens:
                spans.append((start, index))
                start, used = index, 0
            used += size
        spans.append((start, len(sizes)))
        return spans

    def _shorten_pieces(self, pieces: list[str]) -> list[str]:
        """Shorten each piece over the intermediate target; the others pass through.

        Raises ``ValueError`` when the pieces together shrank by less than
        ``min_decrease`` (shortening has stalled), so ``_combine`` cannot loop.
        """
        target = self._aim(final=False)
        assert target is not None
        shortened = []
        for piece in pieces:
            tokens = self._count(piece)
            if tokens <= target:
                shortened.append(piece)
            else:
                # The piece overshot the target once: ask for proportionally less, as for the budget.
                shortened.append(
                    self._shorten(piece, tokens, target * target / tokens).text
                )
        return self._check_progress(pieces, shortened)

    async def _shorten_pieces_async(self, pieces: list[str]) -> list[str]:
        """``_shorten_pieces``, with the pieces shortened at the same time."""
        target = self._aim(final=False)
        assert target is not None

        async def shortened(piece: str) -> str:
            tokens = self._count(piece)
            if tokens <= target:
                return piece
            return (
                await self._shorten_async(piece, tokens, target * target / tokens)
            ).text

        return self._check_progress(
            pieces, list(await asyncio.gather(*(shortened(piece) for piece in pieces)))
        )

    def _check_progress(self, pieces: list[str], shortened: list[str]) -> list[str]:
        """``shortened``, unless together it shrank by less than ``min_decrease``: then raises ``ValueError``."""
        before, after = sum(map(self._count, pieces)), sum(map(self._count, shortened))
        if after >= before or (before - after) / before < self.min_decrease:
            raise ValueError(
                f"the chunk summaries are too long to combine within chunk_tokens={self.chunk_tokens}, "
                "and shortening them stalled; use a larger chunk_tokens"
            )
        return shortened

    @staticmethod
    def _render_pieces(pieces: list[str]) -> UserPrompt:
        """The prompt for combining: the summaries, numbered, oldest first."""
        parts = "\n\n".join(
            f"Part {number}:\n{piece}" for number, piece in enumerate(pieces, 1)
        )
        return UserPrompt(
            text=f"Summaries of consecutive parts of the conversation, oldest first:\n\n{parts}"
        )

    # ------------------------------------------------------------------ budget

    def _fit_budget(self, response: LLMResponse) -> LLMResponse:
        """Shorten ``response`` until it fits ``max_tokens``, or stop and return the shortest so far.

        Models do not count tokens, so the budget is enforced from outside:
        while the shortest summary so far is over ``max_tokens``, it is
        shortened again, and after each overshoot the aim drops by the same
        ratio (budget over actual). It stops when the summary fits, after
        ``max_rounds`` rounds, or when a round cuts less than ``min_decrease``
        of the length (the model has stopped making progress).
        """
        assert self.max_tokens is not None
        best, best_tokens = response, self._count(response.text)
        aim = self.budget_aim * self.max_tokens
        for _ in range(self.max_rounds):
            if best_tokens <= self.max_tokens:
                break
            aim = (
                aim * self.max_tokens / best_tokens
            )  # overshot: ask for proportionally less
            shortened = self._shorten(
                best.text, best_tokens, aim, limit=self.max_tokens
            )
            tokens = self._count(shortened.text)
            decrease = (best_tokens - tokens) / best_tokens
            if tokens < best_tokens:
                best, best_tokens = shortened, tokens
            if decrease < self.min_decrease:
                break  # stalled: another round is unlikely to help
        return best

    async def _fit_budget_async(self, response: LLMResponse) -> LLMResponse:
        """``_fit_budget``, with ``llm.query_async``; the rounds stay one after another."""
        assert self.max_tokens is not None
        best, best_tokens = response, self._count(response.text)
        aim = self.budget_aim * self.max_tokens
        for _ in range(self.max_rounds):
            if best_tokens <= self.max_tokens:
                break
            aim = aim * self.max_tokens / best_tokens
            shortened = await self._shorten_async(
                best.text, best_tokens, aim, limit=self.max_tokens
            )
            tokens = self._count(shortened.text)
            decrease = (best_tokens - tokens) / best_tokens
            if tokens < best_tokens:
                best, best_tokens = shortened, tokens
            if decrease < self.min_decrease:
                break
        return best

    def _shorten(
        self, text: str, tokens: int, aim: float, limit: int | None = None
    ) -> LLMResponse:
        """One call that shortens the summary ``text`` (``tokens`` long) to about ``aim`` tokens."""
        return self.query(
            self.system("shorten", aim), self._shorten_prompt(text, tokens, aim, limit)
        )

    async def _shorten_async(
        self, text: str, tokens: int, aim: float, limit: int | None = None
    ) -> LLMResponse:
        """``_shorten``, with ``llm.query_async``."""
        return await self.query_async(
            self.system("shorten", aim), self._shorten_prompt(text, tokens, aim, limit)
        )

    @staticmethod
    def _shorten_prompt(
        text: str, tokens: int, aim: float, limit: int | None
    ) -> UserPrompt:
        """The prompt asking to shorten ``text`` (``tokens`` long) to about ``aim`` tokens."""
        hard_limit = f" (hard limit: {limit} tokens)" if limit is not None else ""
        return UserPrompt(
            text=f"This summary is {tokens} tokens. Shorten it to about {Tokens.to_words(aim)} words{hard_limit}:\n\n{text}"
        )

    def _aim(self, final: bool) -> float | None:
        """The length, in tokens, a call aims for; ``None`` for no target.

        Intermediate calls (``final=False``) aim at ``chunk_aim`` of
        ``chunk_tokens``; with a budget, every call aims at ``budget_aim`` of
        ``max_tokens``; when both apply, the smaller.
        """
        aims = []
        if not final and self.chunk_tokens is not None:
            aims.append(self.chunk_aim * self.chunk_tokens)
        if self.max_tokens is not None:
            aims.append(self.budget_aim * self.max_tokens)
        return min(aims) if aims else None

    # ------------------------------------------------------------------ prompts

    def system(self, kind: str, aim: float | None = None) -> SystemPrompt:
        """The system prompt for one call: the instructions for ``kind``, then the focus and the length target.

        ``kind`` is ``"summarize"`` (``instructions``), ``"combine"``
        (``combine_instructions``) or ``"shorten"`` (``shorten_instructions``).
        The focus goes into every kind, so combining and shortening keep to it.
        """
        instructions = {
            "summarize": self.instructions,
            "combine": self.combine_instructions,
            "shorten": self.shorten_instructions,
        }
        if kind not in instructions:
            raise ValueError(
                f"kind must be 'summarize', 'combine' or 'shorten', got {kind!r}"
            )
        text = instructions[kind]
        if self.focus is not None:
            text += (
                f" Focus: {self.focus}. Keep what serves this focus, plus anything that limits it "
                "(for example a budget), and leave out the rest."
            )
        if aim is not None:
            text += f" Aim for about {Tokens.to_words(aim)} words"
            text += (
                f"; the summary must be at most {self.max_tokens} tokens."
                if self.max_tokens is not None
                else "."
            )
        return SystemPrompt(text=text)

    def _count(self, text: str) -> int:
        """Tokens in ``text``, in this summarizer's encoding."""
        return Tokens.count(text, self.encoding)
