"""Gemini thinking controls: ``thinking_config`` with a level or a token budget."""

from typing import Any
from ..schemas import ThinkingConfig, UnsupportedThinkingError
from ...utility import Dicts


# Google's documented OpenAI-compatibility effort mapping for Gemini 2.5.
_LEGACY_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def apply_gemini_thinking(
    params: dict[str, Any], thinking: ThinkingConfig, model: str
) -> None:
    nested = Dicts.get(params, "config", copy=True)
    options = {
        **Dicts.get(nested, "thinking_config", copy=True),
        **Dicts.get(params, "thinking_config", copy=True),
        **thinking.others,
    }
    legacy = model.startswith("gemini-2.5-")
    mode, effort = thinking.mode, thinking.effort
    if thinking.budget is not None and mode != "disabled":
        options.pop("thinking_level", None)
        options["thinking_budget"] = thinking.budget
    if mode not in {None, "enabled", "disabled"}:
        raise UnsupportedThinkingError(
            "Gemini",
            model,
            f"mode={mode!r} is not a native mode; use thinking.others for native controls",
        )
    if (
        mode != "disabled"
        and effort is not None
        and (thinking.budget is not None or "thinking_budget" in thinking.others)
    ):
        raise UnsupportedThinkingError(
            "Gemini", model, "choose an effort or a thinking_budget, not both"
        )
    if mode == "enabled" and options.get("thinking_budget") == 0:
        raise UnsupportedThinkingError(
            "Gemini", model, "enabled mode conflicts with thinking_budget=0"
        )
    if mode == "disabled" or options.get("thinking_budget") == 0:
        if not legacy or model.startswith("gemini-2.5-pro"):
            raise UnsupportedThinkingError(
                "Gemini", model, "thinking cannot be disabled for this model"
            )
        options.pop("thinking_level", None)
        options["thinking_budget"] = 0
    elif effort is not None:
        if legacy:
            if effort not in _LEGACY_BUDGETS:
                raise UnsupportedThinkingError(
                    "Gemini",
                    model,
                    "use low/medium/high effort or a native thinking_budget for Gemini 2.5",
                )
            options.pop("thinking_level", None)
            options["thinking_budget"] = _LEGACY_BUDGETS[effort]
        else:
            if effort == "minimal" and model.startswith(
                ("gemini-3.8-", "gemini-3.7-", "gemini-3.1-pro")
            ):
                raise UnsupportedThinkingError(
                    "Gemini", model, "effort='minimal' is not supported"
                )
            options.pop("thinking_budget", None)
            options["thinking_level"] = effort
    elif mode == "enabled":
        if legacy:
            if options.get("thinking_budget") in {None, 0}:
                options["thinking_budget"] = -1
        elif not model.startswith("gemini-3"):
            raise UnsupportedThinkingError(
                "Gemini",
                model,
                "cannot determine an enabled-mode mapping; specify an effort or native thinking controls",
            )
        # Gemini 3 thinks by default; no token budget is needed to enable it.
    if options:
        params["thinking_config"] = options
