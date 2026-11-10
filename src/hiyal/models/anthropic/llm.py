from .translator import AnthropicTranslator
from .transport import AnthropicTransport
from ..bases import BaseLLM


class AnthropicLLM(BaseLLM):
    """Claude models via the Anthropic Messages API."""

    provider = "anthropic"
    display_name = "Anthropic"
    # Claude reads images and documents; it has no audio input.
    prompt_media = frozenset({"images", "documents"})
    generation_adapter = {
        "messages": "user_prompt",
        "system": "system_prompt",
        "max_tokens": "max_output_tokens",
    }

    def _create_translator(self) -> AnthropicTranslator:
        return AnthropicTranslator(self.info)

    def _create_transport(self) -> AnthropicTransport:
        return AnthropicTransport(config=self.config, info=self.info)
