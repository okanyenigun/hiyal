from .translator import GeminiTranslator
from .transport import GeminiTransport
from ..bases import BaseLLM


class GeminiLLM(BaseLLM):
    """Gemini models via the Google GenAI SDK's ``generate_content``."""

    provider = "gemini"
    display_name = "Gemini"
    prompt_media = frozenset({"images", "audio", "documents"})
    generation_adapter = {
        "contents": "user_prompt",
        "system_instruction": "system_prompt",
    }

    def _create_translator(self) -> GeminiTranslator:
        return GeminiTranslator(self.info)

    def _create_transport(self) -> GeminiTransport:
        return GeminiTransport(config=self.config, info=self.info)
