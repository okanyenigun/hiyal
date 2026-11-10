from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

Provider = Literal["openai", "gemini", "anthropic", "ollama", "deepseek"]


@dataclass(frozen=True)
class ProviderInfo:
    """Static facts about one provider, shared by its resolver, translator and transport.

    ``BaseLLM`` builds it from the subclass's class attributes, so a provider
    declares these facts once and every helper receives them explicitly.
    """

    provider: Provider
    display_name: str
    prompt_media: frozenset[str] = frozenset()
    adapter: Mapping[str, str] = field(default_factory=dict)
    generation_adapter: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Read-only views, so no helper can change another helper's mapping.
        object.__setattr__(self, "adapter", MappingProxyType(dict(self.adapter)))
        object.__setattr__(
            self, "generation_adapter", MappingProxyType(dict(self.generation_adapter))
        )


MessageRole = Literal["assistant", "tool", "other"]

ResponseStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "incomplete",
    "failed",
    "cancelled",
]

FinishReason = Literal[
    "stop",
    "length",
    "context_limit",
    "stop_sequence",
    "tool_call",
    "pause",
    "content_filter",
    "recitation",
    "unsupported_language",
    "refusal",
    "invalid_output",
    "account_restricted",
    "other",
    "unknown",
]
