"""Provider-agnostic construction of the existing provider implementations."""

from typing import Any
from .anthropic import AnthropicLLM
from .bases import BaseLLM
from .deepseek import DeepSeekLLM
from .gemini import GeminiLLM
from .ollama import OllamaLLM
from .openai import OpenAILLM
from .schemas import LLMConfig, Provider


_PROVIDERS: dict[Provider, type[BaseLLM]] = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "gemini": GeminiLLM,
    "ollama": OllamaLLM,
    "deepseek": DeepSeekLLM,
}


class LLM:
    """Construct a concrete provider class without changing its behavior.

    For example, ``LLM(provider="openai", model="gpt-5-nano")`` returns an
    ``OpenAILLM`` instance. Configs, keywords, queries, asynchronous execution,
    and streaming are handled entirely by that provider's implementation.
    Subclass ``BaseLLM``, rather than this factory, to implement a new provider.
    """

    def __new__(
        cls,
        *,
        provider: Provider,
        config: LLMConfig | None = None,
        **parameters: Any,
    ) -> BaseLLM:
        if not isinstance(provider, str):
            raise TypeError("provider must be a supported provider name")
        provider_class = _PROVIDERS.get(provider)
        if provider_class is None:
            supported = ", ".join(sorted(_PROVIDERS))
            raise ValueError(
                f"Unsupported provider {provider!r}. Supported providers: {supported}"
            )
        return provider_class(config=config, **parameters)
