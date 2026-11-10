"""Claude thinking controls: ``thinking={"type": ...}`` plus ``output_config.effort``."""

import re
from typing import Any

from ...utility import Dicts
from ..schemas import ThinkingConfig, UnsupportedThinkingError

_VERSION = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?")
# Models whose thinking cannot be switched off.
_ALWAYS_THINKING = ("claude-fable-", "claude-mythos-", "claude-opus-5-5", "claude-sonnet-5-5")


def apply_claude_thinking(params: dict[str, Any], thinking: ThinkingConfig, model: str) -> None:
    options = {**Dicts.get(params, "thinking", copy=True), **thinking.others}
    output = Dicts.get(params, "output_config", copy=True)
    mode = thinking.mode
    if thinking.budget is not None and mode != "disabled":
        options["budget_tokens"] = thinking.budget
        mode = mode or "enabled"
    version = _VERSION.search(model)
    number = None if version is None else tuple(int(n or 0) for n in version.groups())
    # Before 4.6 only manual budgets exist; from 4.7 only adaptive thinking.
    legacy = number is not None and number < (4, 6)
    manual_unsupported = number is not None and number >= (4, 7)
    if mode == "disabled":
        if model.startswith(_ALWAYS_THINKING):
            raise UnsupportedThinkingError("Claude", model, "thinking cannot be fully disabled; use a supported native mode explicitly")
        options = {key: value for key, value in options.items() if key != "budget_tokens"}
        options["type"] = "disabled"
        output.pop("effort", None)
    else:
        if thinking.effort is not None:
            output["effort"] = thinking.effort
        if mode == "enabled":
            if legacy or "budget_tokens" in options:
                if manual_unsupported:
                    raise UnsupportedThinkingError("Claude", model, "manual thinking budgets are not supported; use adaptive mode")
                budget = options.get("budget_tokens")
                if type(budget) is not int or budget < 1024:
                    raise UnsupportedThinkingError("Claude", model, "manual thinking requires thinking_budget (budget_tokens) >= 1024")
                if params.get("max_tokens", params.get("max_output_tokens")) is None:
                    raise UnsupportedThinkingError("Claude", model, "manual thinking requires an explicit max_output_tokens allowance")
                options["type"] = "enabled"
            else:
                options["type"] = "adaptive"
        elif mode is not None:
            if mode == "adaptive" and legacy:
                raise UnsupportedThinkingError("Claude", model, "adaptive thinking is not supported by this model")
            if mode == "adaptive" and "budget_tokens" in options:
                raise UnsupportedThinkingError("Claude", model, "adaptive thinking uses effort, not budget_tokens")
            if mode == "between_tools":
                if not model.startswith("claude-sonnet-5-5"):
                    raise UnsupportedThinkingError("Claude", model, "between_tools is not supported by this model")
                if output.get("effort") in {"xhigh", "max"}:
                    raise UnsupportedThinkingError("Claude", model, "between_tools does not support xhigh/max effort")
            options["type"] = mode
    if options:
        params["thinking"] = options
    if output:
        params["output_config"] = output
    else:
        params.pop("output_config", None)
