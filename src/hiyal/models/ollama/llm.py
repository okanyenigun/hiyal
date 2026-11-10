from .translator import OllamaTranslator
from .transport import OllamaTransport
from ..bases import BaseLLM


class OllamaLLM(BaseLLM):
    """Local or hosted models via the Ollama SDK; it has no Files API."""

    provider = "ollama"
    display_name = "Ollama"
    # Images go to vision models; Ollama has no audio or document input.
    prompt_media = frozenset({"images"})
    generation_adapter = {
        "messages": "user_prompt",
        "system": "system_prompt",
        "num_predict": "max_output_tokens",
        "think": "thinking",
        "stop": "stop_sequences",
    }

    def _create_translator(self) -> OllamaTranslator:
        return OllamaTranslator(self.info)

    def _create_transport(self) -> OllamaTransport:
        return OllamaTransport(config=self.config, info=self.info)
