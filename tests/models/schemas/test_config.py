from copy import deepcopy

import pytest
from pydantic import BaseModel, ValidationError

from hiyal.models.schemas import LLMConfig, ThinkingConfig
from hiyal.prompts import SystemPrompt, UserPrompt
from hiyal.tools import OpenAIToolSpec, ToolSpec, Tools

SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}


def weather(city: str) -> str:
    """Get the weather.

    Args:
        city: City to look up.
    """
    return city


class TestDefaultsAndValidation:
    def test_defaults(self):
        config = LLMConfig()
        assert config.max_retries == 2 and config.asynchronous is False
        assert config.client_options == {} and config.others == {}
        assert config.model is None and config.user_prompt is None and config.tools is None
        assert config.model_fields_set == set()

    @pytest.mark.parametrize("field, value", [
        ("timeout", -1), ("timeout", float("inf")), ("timeout", float("nan")),
        ("max_retries", -1), ("max_output_tokens", 0), ("temperature", -0.1),
        ("top_p", 1.1), ("top_p", -0.1), ("top_k", -1), ("thinking_budget", -1),
        ("thinking_budget", True), ("thinking_budget", 1.5), ("thinking_budget", "2048"),
        ("thinking_mode", ""), ("thinking_mode", "two words"), ("thinking_effort", ""),
        ("parallel_tool_calls", "yes"), ("tool_choice", "any"),
        ("response_modalities", ["VIDEO"]), ("system_prompt", 3), ("user_prompt", 3),
    ])
    def test_invalid_values_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            LLMConfig(**{field: value})

    def test_api_key_is_secret(self):
        config = LLMConfig(api_key="sk-secret")
        assert "sk-secret" not in repr(config) and "sk-secret" not in str(config.model_dump())
        assert config.api_key.get_secret_value() == "sk-secret"

    def test_string_prompts_become_prompt_objects(self):
        config = LLMConfig(system_prompt="Be brief.", user_prompt="Hi")
        assert config.system_prompt == SystemPrompt(text="Be brief.")
        assert config.user_prompt == UserPrompt(text="Hi")

    def test_prompt_objects_are_kept(self):
        prompt = UserPrompt(text="Hi")
        assert LLMConfig(user_prompt=prompt).user_prompt == prompt

    def test_round_trip_through_dump(self):
        config = LLMConfig(model="m", temperature=0.3, tools=[SPEC], thinking={"others": {"summary": "auto"}},
                           thinking_effort="low", others={"x": 1})
        assert LLMConfig.model_validate(config.model_dump()) == config


class TestToolShorthands:
    def test_a_list_becomes_tools_with_defaults(self):
        config = LLMConfig(tools=[SPEC])
        assert isinstance(config.tools, Tools) and config.tools.tool_choice == "auto"
        assert config.tools.parallel_tool_calls is False

    def test_tuples_dicts_and_functions_are_accepted(self):
        assert isinstance(LLMConfig(tools=(SPEC,)).tools, Tools)
        assert LLMConfig(tools={"tool_list": [weather]}).tools.tool_list[0].name == "weather"
        assert isinstance(LLMConfig(tools=[weather]).tools.tool_list[0], ToolSpec)

    def test_settings_fold_into_tools_and_are_cleared(self):
        config = LLMConfig(tools=[SPEC], tool_choice="required", parallel_tool_calls=True)
        assert (config.tools.tool_choice, config.tools.parallel_tool_calls) == ("required", True)
        assert config.tool_choice is None and config.parallel_tool_calls is None

    def test_settings_override_a_tools_object_but_keep_its_other_set_fields(self):
        tools = Tools(tool_list=[SPEC], tool_choice="none", parallel_tool_calls=True)
        config = LLMConfig(tools=tools, tool_choice="auto")
        assert (config.tools.tool_choice, config.tools.parallel_tool_calls) == ("auto", True)
        assert "parallel_tool_calls" in config.tools.model_fields_set

    def test_unset_fields_stay_unset_after_folding(self):
        config = LLMConfig(tools=[SPEC], tool_choice="auto")
        assert "parallel_tool_calls" not in config.tools.model_fields_set

    def test_settings_without_tools_wait_for_tools(self):
        config = LLMConfig(tool_choice="required", parallel_tool_calls=False)
        assert (config.tool_choice, config.parallel_tool_calls, config.tools) == ("required", False, None)

    def test_required_still_needs_a_tool(self):
        with pytest.raises(ValidationError, match="needs at least one tool"):
            LLMConfig(tools=[], tool_choice="required")

    def test_provider_spec_subclasses_survive(self):
        config = LLMConfig(tools=[OpenAIToolSpec(**SPEC, allowed_callers=["direct"])], tool_choice="none")
        assert type(config.tools.tool_list[0]) is OpenAIToolSpec


class TestThinkingFlattening:
    def test_dictionary_controls_become_flat_fields(self):
        config = LLMConfig(thinking={"effort": "medium", "mode": "enabled", "budget": 1024})
        assert (config.thinking_effort, config.thinking_mode, config.thinking_budget) == ("medium", "enabled", 1024)
        assert config.thinking is None
        assert config.model_fields_set == {"thinking_effort", "thinking_mode", "thinking_budget"}

    def test_native_options_stay_under_thinking(self):
        config = LLMConfig(thinking={"effort": "low", "others": {"summary": "auto"}})
        assert config.thinking == ThinkingConfig(others={"summary": "auto"})
        assert config.thinking_effort == "low"

    @pytest.mark.parametrize("value, field", [("low", "thinking_effort"), ("high", "thinking_effort"),
                                              ("xhigh", "thinking_effort"), ("adaptive", "thinking_effort"),
                                              ("enabled", "thinking_mode"), ("disabled", "thinking_mode")])
    def test_string_shorthand(self, value, field):
        assert getattr(LLMConfig(thinking=value), field) == value

    def test_direct_fields_win_over_dictionary_controls(self):
        dictionary = {"mode": "enabled", "effort": "low", "budget": 2048, "others": {"summary": "auto"}}
        snapshot = deepcopy(dictionary)
        config = LLMConfig(thinking=dictionary, thinking_effort="high", thinking_budget=None)
        assert (config.thinking_mode, config.thinking_effort, config.thinking_budget) == ("enabled", "high", None)
        assert dictionary == snapshot

    def test_none_resets_every_thinking_field(self):
        config = LLMConfig(thinking=None)
        for name in ("thinking", "thinking_mode", "thinking_effort", "thinking_budget"):
            assert getattr(config, name) is None and name in config.model_fields_set

    def test_none_keeps_explicit_direct_fields(self):
        assert LLMConfig(thinking=None, thinking_effort="low").thinking_effort == "low"

    def test_thinking_config_instances_are_flattened(self):
        config = LLMConfig(thinking=ThinkingConfig(effort="low", others={"summary": "auto"}))
        assert config.thinking_effort == "low" and config.thinking.others == {"summary": "auto"}
        assert config.thinking.effort is None

    @pytest.mark.parametrize("value", [True, 100, {"effort": ""}, {"effort": "high effort"},
                                       {"mode": False}, {"mod": "enabled"}, {"budget": -1}, {"budget": 1.5}])
    def test_invalid_shapes_are_rejected(self, value):
        with pytest.raises(ValidationError):
            LLMConfig(thinking=value)


class _Native(BaseModel):
    handle: object
    label: str


class TestIsolatedCopy:
    def test_copy_is_equal_and_keeps_fields_set(self):
        config = LLMConfig(model="m", tools=[SPEC], others={"a": [1, {"b": 2}]})
        copied = config.isolated_copy()
        assert copied == config and copied is not config
        assert copied.model_fields_set == config.model_fields_set

    def test_config_structures_are_independent(self):
        config = LLMConfig(tools=[SPEC], others={"a": [1, {"b": 2}]}, client_options={"headers": {"x": "1"}})
        copied = config.isolated_copy()
        copied.others["a"][1]["b"] = 99
        copied.client_options["headers"]["x"] = "2"
        copied.tools.tool_list[0].description = "Changed."
        assert config.others == {"a": [1, {"b": 2}]}
        assert config.client_options == {"headers": {"x": "1"}}
        assert config.tools.tool_list[0].description == "Get weather."

    def test_opaque_objects_are_shared_not_copied(self):
        handle, callback = object(), (lambda: None)
        config = LLMConfig(client_options={"http_client": handle, "hooks": [callback]},
                           others={"nested": {"deep": (handle,)}},
                           thinking={"others": {"model_object": _Native(handle=handle, label="x")}})
        copied = config.isolated_copy()
        assert copied.client_options["http_client"] is handle
        assert copied.client_options["hooks"][0] is callback
        assert copied.others["nested"]["deep"][0] is handle
        native = copied.thinking.others["model_object"]
        assert native is not config.thinking.others["model_object"] and native.handle is handle

    def test_cyclic_extras_do_not_hang(self):
        cyclic: dict = {}
        cyclic["self"] = cyclic
        copied = LLMConfig(others={"cycle": cyclic}).isolated_copy()
        assert copied.others["cycle"]["self"] is copied.others["cycle"]
