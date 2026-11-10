from .translator import OpenAITranslator
from .transport import OpenAITransport
from ..bases import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI models via the Responses API, and Chat Completions for audio."""

    provider = "openai"
    display_name = "OpenAI"
    prompt_media = frozenset({"images", "documents", "audio"})
    generation_adapter = {
        "instructions": "system_prompt",
        "input": "user_prompt",
    }

    def _create_translator(self) -> OpenAITranslator:
        return OpenAITranslator(self.info)

    def _create_transport(self) -> OpenAITransport:
        return OpenAITransport(config=self.config, info=self.info)
