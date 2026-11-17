"""The summary worker: one step of a summary, linked to the next one in a chain."""

import asyncio
from abc import ABC, abstractmethod
from typing import Self
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt, UserPrompt


class SummaryWorker(ABC):
    """One step of a summary; workers link into a chain, each passing its output to the next.

    The app calls ``summarize``: on the first worker, it runs every worker
    of the chain, one after another, each on the previous one's output
    (chain of responsibility). ``then`` links workers in order::

        summarizer = (
            ChunkedSummarizer(cheap_llm, chunk_tokens=8000)
            .then(SingleCallSummarizer(llm))
            .then(BudgetSummarizer(llm, max_tokens=500))
        )
        summary = summarizer.summarize(text)

    Each worker writes its own step in ``_summarize``, which only the chain
    calls: ``SingleCallSummarizer`` summarizes the text in one call,
    ``ChunkedSummarizer`` shrinks it chunk by chunk until it fits one call,
    ``BudgetSummarizer`` shortens it until it fits a token budget. Alone, a
    worker's ``summarize`` is its own step.

    A worker with nothing to do passes the text on unchanged:
    ``ChunkedSummarizer`` when the text already fits one call,
    ``BudgetSummarizer`` when it is within the budget.

    Workers only see the text. Building it (joining entries, a previous
    summary first, capping a huge entry) is the caller's job.

    In async code, ``summarize_async`` runs the chain the same way, calling
    each worker's ``_summarize_async``. By default that runs ``_summarize`` in a
    worker thread, so it never blocks the event loop; a worker that asks an
    LLM overrides it to use the LLM's async client.
    """

    _next: "SummaryWorker | None" = None

    def then(self, worker: "SummaryWorker") -> Self:
        """Link ``worker``, and the workers after it, at the end of this chain; returns this worker.

        It returns the chain's first worker, so calls can be stacked and the
        result is the summarizer to call. Raises ``TypeError`` for something
        that is not a worker, and ``ValueError`` for a worker already in the
        chain, which would make the chain loop.
        """
        if not isinstance(worker, SummaryWorker):
            raise TypeError(f"a chain links workers, not a {type(worker).__name__}")
        chain = self._chain()
        linked = {id(member) for member in chain}
        if any(id(member) in linked for member in worker._chain()):
            raise ValueError("a worker can be in a chain only once")
        chain[-1]._next = worker
        return self

    def summarize(self, text: str) -> str:
        """Run this worker on ``text``, then hand its output to the next one; returns the last worker's output.

        Raises ``TypeError`` when ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"a worker takes a text, not a {type(text).__name__}")
        text = self._summarize(text)
        return self._next.summarize(text) if self._next is not None else text

    @abstractmethod
    def _summarize(self, text: str) -> str:
        """This worker's own step, called by ``summarize``: ``text`` turned into the text the next worker gets."""

    async def summarize_async(self, text: str) -> str:
        """``summarize``, for async code: each worker runs ``_summarize_async``.

        Raises ``TypeError`` when ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"a worker takes a text, not a {type(text).__name__}")
        text = await self._summarize_async(text)
        return (
            await self._next.summarize_async(text) if self._next is not None else text
        )

    async def _summarize_async(self, text: str) -> str:
        """``_summarize``, in a worker thread by default."""
        return await asyncio.to_thread(self._summarize, text)

    def _chain(self) -> list["SummaryWorker"]:
        """This worker and every worker after it, in order."""
        chain: list[SummaryWorker] = []
        worker: SummaryWorker | None = self
        while worker is not None:
            chain.append(worker)
            worker = worker._next
        return chain


class LLMSummaryWorker(SummaryWorker):
    """A worker that asks an LLM; ``_ask`` makes one call and refuses an answer that is cut off or empty.

    ``_ask`` needs a sync LLM client, and ``_ask_async`` an async one
    (``asynchronous=True`` in its config).
    """

    def __init__(self, llm: BaseLLM) -> None:
        """Use ``llm`` for every call; a text model is enough."""
        self.llm = llm

    def _ask(self, instructions: str, text: str) -> str:
        """One call, with ``instructions`` as the system prompt and ``text`` as the user prompt; returns the answer.

        Raises ``ValueError`` when the answer is not ``completed`` or has no
        text. A response cut off by the model's output limit
        (``max_output_tokens``) comes back ``incomplete``, its text truncated
        mid-sentence; used as a summary, it would hide what it covers while
        missing what was cut.
        """
        return self._answer(
            self.llm.query(
                system_prompt=SystemPrompt(text=instructions),
                user_prompt=UserPrompt(text=text),
            )
        )

    async def _ask_async(self, instructions: str, text: str) -> str:
        """``_ask``, on the LLM's async client."""
        return self._answer(
            await self.llm.query_async(
                system_prompt=SystemPrompt(text=instructions),
                user_prompt=UserPrompt(text=text),
            )
        )

    @staticmethod
    def _answer(response: LLMResponse) -> str:
        """The text of ``response``; raises ``ValueError`` when it is not ``completed`` or has no text."""
        if response.status != "completed":
            raise ValueError(
                f"the answer is {response.status!r}, not 'completed'; a cut-off "
                "summary would lose what it covers (is max_output_tokens too low?)"
            )
        if not response.text.strip():
            raise ValueError("the answer has no text")
        return response.text

    @staticmethod
    def _share(value: object, name: str) -> float:
        """``value`` when it is a number in ``(0, 1]``; raises ``ValueError`` otherwise."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be a number in (0, 1], got {value!r}")
        return value

    @staticmethod
    def _decrease(value: object) -> float:
        """``value`` when it is a number in ``[0, 1)``, for ``min_decrease``; raises ``ValueError`` otherwise."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value < 1
        ):
            raise ValueError(f"min_decrease must be a number in [0, 1), got {value!r}")
        return value
