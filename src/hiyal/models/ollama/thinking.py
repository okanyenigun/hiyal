"""Ollama thinking controls: a single ``think`` value (bool or level)."""

from typing import Any
from ..schemas import ThinkingConfig, UnsupportedThinkingError


def apply_ollama_thinking(
    params: dict[str, Any], thinking: ThinkingConfig, model: str
) -> None:
    if thinking.others:
        raise UnsupportedThinkingError(
            "Ollama",
            model,
            "there are no nested thinking options; use an effort or mode",
        )
    if thinking.mode == "disabled":
        params["think"] = False
    elif thinking.mode in {None, "enabled"}:
        if thinking.effort is not None:
            params["think"] = thinking.effort
        elif thinking.mode == "enabled":
            params["think"] = True
    elif thinking.effort is not None:
        raise UnsupportedThinkingError(
            "Ollama",
            model,
            "a native mode and an effort cannot both be encoded in think",
        )
    else:
        params["think"] = thinking.mode


def validate_ollama_think(value: Any, info: dict[str, Any], model: str) -> bool | str:
    """Check ``think`` against the server's model metadata rather than guessing.

    ``True`` becomes the model's default level when the model only takes levels.
    """
    metadata = info.get("thinking")
    values = metadata.get("values") if isinstance(metadata, dict) else None
    if not isinstance(values, list) or not values:
        raise UnsupportedThinkingError(
            "Ollama",
            model,
            "the server did not report thinking controls; cannot validate the requested setting",
        )
    if not isinstance(value, (bool, str)):
        raise UnsupportedThinkingError(
            "Ollama", model, "think must be a boolean or a supported level"
        )

    def supported(choice: bool | str) -> bool:
        return any(type(item) is type(choice) and item == choice for item in values)

    if supported(value):
        return value
    if value is True:
        assert isinstance(metadata, dict)
        default = metadata.get("default")
        if isinstance(default, str) and supported(default):
            return default
    raise UnsupportedThinkingError(
        "Ollama",
        model,
        f"thinking setting {value!r} is not supported; allowed values: {values!r}",
    )
