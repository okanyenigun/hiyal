from .translator import DeepSeekTranslator
from .transport import DeepSeekTransport
from ..bases import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek through its OpenAI-compatible Chat Completions API."""

    provider = "deepseek"
    display_name = "DeepSeek"
    # Images only, and only on the experimental vision model.
    prompt_media = frozenset({"images"})
    generation_adapter = {
        "messages": "user_prompt",
        "max_tokens": "max_output_tokens",
        "stop": "stop_sequences",
    }

    def _create_translator(self) -> DeepSeekTranslator:
        return DeepSeekTranslator(self.info)

    def _create_transport(self) -> DeepSeekTransport:
        return DeepSeekTransport(config=self.config, info=self.info)
