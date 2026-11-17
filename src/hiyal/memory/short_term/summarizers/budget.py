"""BudgetSummarizer: shortens a text, round after round, until it fits a token budget."""

from collections.abc import Generator
from .base import LLMSummaryWorker
from ....models.bases import BaseLLM
from ....prompts.hub import hub
from ....utility import Checks, Tokens
from ....utility.tokens import DEFAULT_ENCODING


class BudgetSummarizer(LLMSummaryWorker):
    """Shortens the text to ``max_tokens``, round after round, until it fits or stops getting shorter.

    Models do not count tokens, so the budget is enforced from outside: a
    round asks for about ``aim`` of ``max_tokens``, and after an overshoot
    the next round asks for proportionally less. It stops when the text
    fits, after ``max_rounds`` rounds, or when a round shortens it by less
    than ``min_decrease`` (the model has stopped making progress). The
    shortest version so far is kept: this is the most it can do, so the
    result may still be over the budget.

    A text within the budget passes on unchanged, with no call. In async
    code the rounds are the same, one after another, since each shortens
    the last one's answer.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_tokens: int,
        *,
        max_rounds: int = 3,
        min_decrease: float = 0.1,
        aim: float = 0.75,
        prompt_version: str = "v1",
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """Shorten the text to at most ``max_tokens``.

        - ``max_rounds``: the most calls.
        - ``min_decrease``: the least share a round must shorten the text by.
        - ``aim``: the first round asks for this share of ``max_tokens``,
          since models overshoot length targets.
        - ``prompt_version``: the version of hub prompt ``memory/shorten``.
          ``v2`` keeps the headings of ``memory/summarize`` ``v2``.
        - ``encoding``: the tiktoken encoding for every count.

        Raises ``ValueError`` for a ``max_tokens`` or ``max_rounds`` that is
        not a positive integer, a ``min_decrease`` outside ``[0, 1)``, an
        ``aim`` outside ``(0, 1]``, or an unknown encoding.
        """
        super().__init__(llm)
        self.max_tokens = Checks.positive_int(max_tokens, "max_tokens")
        self.max_rounds = Checks.positive_int(max_rounds, "max_rounds")
        self.min_decrease = self._decrease(min_decrease)
        self.aim = self._share(aim, "aim")
        self.instructions = hub.get("memory/shorten", prompt_version)
        self.encoding = Tokens.require_encoding(encoding)

    def _summarize(self, text: str) -> str:
        """``text`` shortened round after round (see the class); the shortest version reached."""
        rounds = self._rounds(text)
        try:
            request = next(rounds)
            while True:
                request = rounds.send(self._ask(*request))
        except StopIteration as done:  # the rounds are over
            return done.value

    async def _summarize_async(self, text: str) -> str:
        """``_summarize``, on the LLM's async client."""
        rounds = self._rounds(text)
        try:
            request = next(rounds)
            while True:
                request = rounds.send(await self._ask_async(*request))
        except StopIteration as done:  # the rounds are over
            return done.value

    def _rounds(self, text: str) -> Generator[tuple[str, str], str, str]:
        """The rounds, written once for both sides: yields each request, is sent its answer, and returns the shortest version.

        A request is ``(instructions, prompt)``; ``_summarize`` answers it with
        ``_ask``, and ``_summarize_async`` with ``_ask_async``.
        """
        best, best_tokens = text, self._count(text)
        aim = self.aim * self.max_tokens
        for _ in range(self.max_rounds):
            if best_tokens <= self.max_tokens:
                break
            shortened = yield self._system(aim), self._prompt(best, best_tokens, aim)
            tokens = self._count(shortened)
            decrease = (best_tokens - tokens) / best_tokens
            if tokens < best_tokens:
                best, best_tokens = shortened, tokens
            if decrease < self.min_decrease:
                break  # stalled: another round is unlikely to help
            if tokens > self.max_tokens:
                aim = (
                    aim * self.max_tokens / tokens
                )  # overshot: ask for proportionally less
        return best

    def _system(self, aim: float) -> str:
        """The hub instructions, with the length target."""
        return (
            f"{self.instructions} Aim for about {Tokens.to_words(aim)} words; "
            f"the summary must be at most {self.max_tokens} tokens."
        )

    def _prompt(self, text: str, tokens: int, aim: float) -> str:
        """The request to shorten ``text``, which is ``tokens`` long, to about ``aim`` tokens."""
        return (
            f"This summary is {tokens} tokens. Shorten it to about {Tokens.to_words(aim)} "
            f"words (hard limit: {self.max_tokens} tokens):\n\n{text}"
        )

    def _count(self, text: str) -> int:
        """Tokens in ``text``, in this worker's encoding."""
        return Tokens.count(text, self.encoding)
