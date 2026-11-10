from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ThinkingConfig(BaseModel):
    """Portable thinking controls, with an escape hatch for native options."""

    model_config = ConfigDict(extra="forbid")

    effort: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
        description="Common levels are low, medium and high; native levels such as xhigh are also allowed.",
        examples=["low", "medium", "high", "xhigh"],
    )
    mode: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
        description="Common modes are enabled and disabled; native modes such as adaptive are also allowed. Disabled ignores effort.",
        examples=["enabled", "disabled", "adaptive"],
    )
    budget: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Thinking-token budget on models supporting a numeric budget. Disabled mode ignores it.",
        examples=[2048],
    )
    others: dict[str, Any] = Field(
        default_factory=dict,
        description="Native thinking options, such as budget_tokens, thinking_budget or summary.",
    )


class UnsupportedThinkingError(ValueError):
    """A thinking control the provider or model cannot express.

    Hiyal raises it instead of quietly replacing the request with weaker
    reasoning. The message reads ``<provider> model '<model>': <reason>``.
    """

    def __init__(self, provider: str, model: str, reason: str) -> None:
        super().__init__(f"{provider} model {model!r}: {reason}")
        self.provider = provider
        self.model = model
        self.reason = reason


def flatten_thinking(values: Any) -> Any:
    """Normalize common dictionary controls without mutating the caller's data.

    Only explicitly supplied controls become overrides. Native options remain
    in ``thinking.others``; direct fields win over dictionary controls.
    """
    if not isinstance(values, dict) or "thinking" not in values:
        return values
    result = dict(values)
    value = result.pop("thinking")
    if value is None:
        # Preserve the existing explicit reset of all thinking defaults.
        result["thinking"] = None
        for name in ("mode", "effort", "budget"):
            result.setdefault(f"thinking_{name}", None)
        return result
    if isinstance(value, str):
        value = {"mode" if value in {"enabled", "disabled"} else "effort": value}
    if isinstance(value, dict):
        value = {
            name: option
            for name, option in value.items()
            if name not in {"mode", "effort", "budget"}
            or f"thinking_{name}" not in result
        }
    thinking = ThinkingConfig.model_validate(value)
    for name in ("mode", "effort", "budget"):
        if name in thinking.model_fields_set:
            result.setdefault(f"thinking_{name}", getattr(thinking, name))
    if "others" in thinking.model_fields_set:
        result["thinking"] = ThinkingConfig(others=thinking.others)
    return result
