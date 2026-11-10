"""Groups of ``LLMConfig`` field names that have special meaning, and helpers over them."""

from collections.abc import Mapping
from .config import LLMConfig

# Every field name an ``LLMConfig`` has.
CONFIG_FIELDS = frozenset(LLMConfig.model_fields)
# Connection settings: sent to the SDK client, never in the generation payload.
CONNECTION_FIELDS = frozenset({"api_key", "timeout", "max_retries"})
# Fixed when the client is created; a single call cannot change them.
CLIENT_ONLY_FIELDS = frozenset({"asynchronous", "client_options"})
THINKING_FIELDS = frozenset({"thinking_mode", "thinking_effort", "thinking_budget"})
TOOL_FIELDS = frozenset({"tools", "tool_choice", "parallel_tool_calls"})
# Per-call fields where ``None`` means "keep the client default" rather than "unset".
KEEP_DEFAULT_WHEN_NONE = frozenset({"model"}) | CONNECTION_FIELDS
# Settings the translator handles itself instead of copying into the request.
NOT_GENERATION_FIELDS = (
    TOOL_FIELDS
    | THINKING_FIELDS
    | CONNECTION_FIELDS
    | CLIENT_ONLY_FIELDS
    | {"thinking"}
)
DEFAULT_MAX_RETRIES: int = LLMConfig.model_fields["max_retries"].default


def config_aliases(adapter: Mapping[str, str]) -> dict[str, str]:
    """Return provider-name -> common-config-name mappings.

    Either direction may be written in an adapter: the side that is an
    ``LLMConfig`` field is the common name.
    """
    aliases: dict[str, str] = {}
    for left, right in adapter.items():
        if left in CONFIG_FIELDS and right not in CONFIG_FIELDS:
            aliases[right] = left
        else:
            aliases[left] = right
    return aliases
