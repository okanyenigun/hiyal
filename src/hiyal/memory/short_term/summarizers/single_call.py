"""SingleCallSummarizer: summarizes the text in one call."""

from .base import LLMSummaryWorker
from ....models.bases import BaseLLM
from ....prompts.hub import hub


class SingleCallSummarizer(LLMSummaryWorker):
    """Summarizes the whole text in one LLM call.

    The text is sent under ``Conversation to summarize:``. It must fit one
    call: put a ``ChunkedSummarizer`` before it when it may not.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        prompt_name: str = "memory/summarize",
        prompt_version: str = "v1",
    ) -> None:
        """Summarize with ``llm``, following hub prompt ``prompt_name`` at ``prompt_version``.

        ``"memory/summarize"`` covers everything; with working notes,
        ``"memory/summarize_with_notes"`` records what happened and leaves the
        current state to the notes, so the two do not repeat each other.
        """
        super().__init__(llm)
        self.instructions = hub.get(prompt_name, prompt_version)

    def _summarize(self, text: str) -> str:
        """The summary of ``text``; raises ``ValueError`` when it is blank."""
        return self._ask(self.instructions, self._request(text))

    async def _summarize_async(self, text: str) -> str:
        """``_summarize``, on the LLM's async client."""
        return await self._ask_async(self.instructions, self._request(text))

    @staticmethod
    def _request(text: str) -> str:
        """The user prompt asking to summarize ``text``; raises ``ValueError`` when it is blank."""
        if not text.strip():
            raise ValueError("there is nothing to summarize")
        return f"Conversation to summarize:\n{text}"
