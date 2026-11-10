from typing import Any
from ..schemas import LLMConfig, ProviderInfo
from ..schemas.fields import (
    CLIENT_ONLY_FIELDS,
    CONFIG_FIELDS,
    DEFAULT_MAX_RETRIES,
    KEEP_DEFAULT_WHEN_NONE,
    THINKING_FIELDS,
    config_aliases,
)
from ..schemas.thinking import flatten_thinking


class ConfigResolver:
    """Turns constructor and per-call inputs into validated ``LLMConfig`` objects.

    It is provider-neutral: the only provider knowledge it needs is the alias
    maps in ``ProviderInfo``. Results never share mutable state with the
    caller's config or with the client defaults.
    """

    def __init__(self, info: ProviderInfo) -> None:
        self.info = info
        # Constructor keywords may use client-level and generation names.
        self._client_aliases = config_aliases(
            {**info.adapter, **info.generation_adapter}
        )
        self._call_aliases = config_aliases(info.generation_adapter)

    def client_config(
        self, config: LLMConfig | None, parameters: dict[str, Any]
    ) -> LLMConfig:
        """The client's default config; unknown keywords go to ``client_options``."""
        resolved = self._merge(
            config,
            parameters,
            aliases=self._client_aliases,
            extra_field="client_options",
        )
        if resolved.max_retries is None:
            resolved.max_retries = DEFAULT_MAX_RETRIES
        return resolved

    def call_config(
        self,
        defaults: LLMConfig,
        config: LLMConfig | None,
        parameters: dict[str, Any],
    ) -> LLMConfig:
        """One call's config: ``defaults`` with this call's explicit values on top.

        Unknown keywords go to ``others``. Raises when the call tries to change
        a client-only setting, has no model, or sets tool options without tools.
        """
        overrides = self._merge(
            config,
            parameters,
            aliases=self._call_aliases,
            extra_field="others",
            default_extras=defaults.others,
        )
        fixed = overrides.model_fields_set & CLIENT_ONLY_FIELDS
        if fixed:
            raise ValueError(
                f"{', '.join(sorted(fixed))} is a client setting and cannot be changed per call"
            )
        values = {name: getattr(defaults, name) for name in CONFIG_FIELDS}
        for name in overrides.model_fields_set:
            value = getattr(overrides, name)
            if value is None and name in KEEP_DEFAULT_WHEN_NONE:
                continue
            values[name] = value
        resolved = LLMConfig.model_validate(values).isolated_copy()
        if resolved.tool_choice is not None or resolved.parallel_tool_calls is not None:
            # LLMConfig folds these into tools; leftovers mean there were none.
            raise ValueError(
                "tool_choice and parallel_tool_calls need tools, on this call or the client"
            )
        if resolved.model is None:
            raise ValueError(
                "A model is required: supply it at initialization or for this call."
            )
        return resolved

    def _merge(
        self,
        config: LLMConfig | None,
        parameters: dict[str, Any],
        *,
        aliases: dict[str, str],
        extra_field: str,
        default_extras: dict[str, Any] | None = None,
    ) -> LLMConfig:
        """Merge a config object and keywords; explicit config fields win.

        ``default_extras`` are the extras keyword extras are added to. Two
        spellings of one setting (``input`` and ``user_prompt``) raise.
        """
        default_extras = default_extras or {}
        if "stream" in parameters:
            raise ValueError(
                "Streaming is selected by query_stream/query_stream_async, not a stream parameter"
            )
        # Runtime guard for untyped callers; type checkers see it as unreachable.
        if config is not None and not isinstance(config, LLMConfig):
            raise TypeError("config must be an LLMConfig or None")
        explicit = config.model_fields_set if config is not None else set()
        parameters = self._normalize_thinking(config, parameters, aliases, explicit)

        updates: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        spelled: dict[str, str] = {}
        for name, value in parameters.items():
            config_name = aliases.get(name, name)
            if config_name in explicit:
                continue
            if config_name in CONFIG_FIELDS:
                if config_name in spelled:
                    raise ValueError(
                        f"{spelled[config_name]!r} and {name!r} both set {config_name!r}; pass only one"
                    )
                spelled[config_name] = name
                updates[config_name] = value
            elif extra_field not in explicit:
                # Keep the provider's spelling: these go back to that provider.
                extras[name] = value

        if extras:
            # Keyword extras add to the default extras; an explicit
            # ``others=`` / ``client_options=`` replaces them.
            supplied = updates.get(extra_field, default_extras)
            if not isinstance(supplied, dict):
                raise TypeError(f"'{extra_field}' must be a dictionary")
            updates[extra_field] = {**supplied, **extras}
        if config is not None:
            updates.update({name: getattr(config, name) for name in explicit})
        # Copy after validation so caller-owned tools/extras stay independent
        # without a dump/validate round trip that loses provider subclasses.
        return LLMConfig.model_validate(updates).isolated_copy()

    @staticmethod
    def _normalize_thinking(
        config: LLMConfig | None,
        parameters: dict[str, Any],
        aliases: dict[str, str],
        explicit: set[str],
    ) -> dict[str, Any]:
        """Collect every spelling of the thinking settings into flat ``thinking_*`` keys."""
        parameters = dict(parameters)
        for name, common_name in aliases.items():
            if common_name == "thinking" and name in parameters:
                if "thinking" in parameters:
                    raise ValueError(
                        f"{name!r} and 'thinking' both set 'thinking'; pass only one"
                    )
                parameters["thinking"] = parameters.pop(name)
        if config is not None:
            for name in explicit & THINKING_FIELDS:
                parameters[name] = getattr(config, name)
        return flatten_thinking(parameters)
