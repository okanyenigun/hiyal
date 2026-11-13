"""The interface every summarizer implements."""

from abc import ABC, abstractmethod
from ..entry import Entry, SummaryEntry
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt, UserPrompt
from ....prompts.hub import hub


class BaseSummarizer(ABC):
    """Turns history entries into a summary, using an LLM.

    The interface compaction depends on. By default ``summarize`` renders
    the entries into a prompt (``render``) and makes one call with the
    instructions; a summarizer can override it to make more calls, as
    ``Summarizer`` does for chunking and a token budget. ``query`` makes one
    call and refuses a cut-off response.

    A summarizer only produces the summary. Deciding when to summarize and
    which entries to cover is compaction's job.
    """

    def __init__(
        self,
        llm: BaseLLM,
        instructions: str | None = None,
        prompt_name: str = "memory/summarize",
        prompt_version: str = "v1",
    ) -> None:
        """Use ``llm`` for summaries, with the instructions from the prompt hub.

        The instructions are prompt ``prompt_name`` at ``prompt_version`` in
        the hub, so another version can be tried by passing it. Text given
        as ``instructions`` is used instead of the hub's.
        """
        self.llm = llm
        self.instructions = (
            instructions
            if instructions is not None
            else hub.get(prompt_name, prompt_version)
        )

    def summarize(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> LLMResponse:
        """Summarize ``entries`` and return the LLM's response.

        Renders them with ``render`` and makes one ``llm.query`` call, with
        ``instructions`` as the system prompt. The summary is the response's
        ``text``; the response also carries the call's token usage.

        With ``previous`` (rolling summarization), the new entries are folded
        into that summary instead of summarizing from scratch.

        Raises ``ValueError`` when there are no entries, or when the response
        is not complete (see ``query``).
        """
        if not entries:
            raise ValueError("there are no entries to summarize")
        return self.query(
            SystemPrompt(text=self.instructions), self.render(entries, previous)
        )

    def query(
        self, system_prompt: SystemPrompt, user_prompt: UserPrompt
    ) -> LLMResponse:
        """One call to ``llm``; raises ``ValueError`` unless the response's status is ``completed``.

        A response cut off by the model's output limit (``max_output_tokens``)
        comes back ``incomplete``, its text truncated mid-sentence. Installed
        as a summary, it would hide the entries it covers while missing what
        was cut, so it is refused instead.
        """
        return self._completed(
            self.llm.query(system_prompt=system_prompt, user_prompt=user_prompt)
        )

    async def summarize_async(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> LLMResponse:
        """``summarize``, with ``llm.query_async``."""
        if not entries:
            raise ValueError("there are no entries to summarize")
        return await self.query_async(
            SystemPrompt(text=self.instructions), self.render(entries, previous)
        )

    async def query_async(
        self, system_prompt: SystemPrompt, user_prompt: UserPrompt
    ) -> LLMResponse:
        """``query``, with ``llm.query_async``."""
        response = await self.llm.query_async(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        return self._completed(response)

    @staticmethod
    def _completed(response: LLMResponse) -> LLMResponse:
        """``response``, if its status is ``completed``; raises ``ValueError`` otherwise (see ``query``)."""
        if response.status != "completed":
            raise ValueError(
                f"the summary response is {response.status!r}, not 'completed'; "
                "a cut-off summary would lose what it covers (is max_output_tokens too low?)"
            )
        return response

    @abstractmethod
    def render(
        self, entries: list[Entry], previous: SummaryEntry | None = None
    ) -> UserPrompt:
        """Turn the entries, and the previous summary if any, into the prompt to summarize."""
