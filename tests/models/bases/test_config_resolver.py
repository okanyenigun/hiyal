import pytest
from pydantic import ValidationError

from hiyal.models.bases import ConfigResolver
from hiyal.models.schemas import LLMConfig, ProviderInfo, ThinkingConfig
from hiyal.tools import Tools

SPEC = {"name": "get_weather", "description": "Get weather.", "parameters": {"properties": []}}
INFO = ProviderInfo(provider="openai", display_name="Fake",
                    adapter={"model_name": "model"},
                    generation_adapter={"instructions": "system_prompt", "max_tokens": "max_output_tokens",
                                        "think": "thinking"})


@pytest.fixture
def resolver():
    return ConfigResolver(INFO)


class TestClientConfig:
    def test_keywords_fill_fields_and_unknown_ones_become_client_options(self, resolver):
        config = resolver.client_config(None, {"model": "m", "temperature": 0.2, "base_url": "http://x"})
        assert (config.model, config.temperature) == ("m", 0.2)
        assert config.client_options == {"base_url": "http://x"}
        assert config.others == {}

    def test_both_adapters_are_accepted_at_construction(self, resolver):
        config = resolver.client_config(None, {"model_name": "m", "instructions": "sys", "max_tokens": 9})
        assert (config.model, config.system_prompt.text, config.max_output_tokens) == ("m", "sys", 9)

    def test_explicit_config_fields_win_over_keywords(self, resolver):
        config = resolver.client_config(LLMConfig(model="explicit", temperature=0.9),
                                        {"model": "keyword", "temperature": 0.1, "top_p": 0.5})
        assert (config.model, config.temperature, config.top_p) == ("explicit", 0.9, 0.5)

    def test_explicit_client_options_block_keyword_extras(self, resolver):
        config = resolver.client_config(LLMConfig(client_options={"a": 1}), {"b": 2})
        assert config.client_options == {"a": 1}

    def test_keyword_extras_merge_with_keyword_client_options(self, resolver):
        config = resolver.client_config(None, {"client_options": {"a": 1}, "b": 2})
        assert config.client_options == {"a": 1, "b": 2}

    def test_non_dict_client_options_with_extras_raise(self, resolver):
        with pytest.raises(TypeError, match="'client_options' must be a dictionary"):
            resolver.client_config(None, {"client_options": [1], "b": 2})

    def test_explicit_none_retries_become_the_default(self, resolver):
        assert resolver.client_config(None, {"max_retries": None}).max_retries == 2
        assert resolver.client_config(LLMConfig(max_retries=None), {}).max_retries == 2
        assert resolver.client_config(None, {"max_retries": 0}).max_retries == 0

    def test_stream_keyword_is_rejected(self, resolver):
        with pytest.raises(ValueError, match="Streaming is selected by query_stream"):
            resolver.client_config(None, {"stream": True})

    def test_non_config_objects_are_rejected(self, resolver):
        with pytest.raises(TypeError, match="config must be an LLMConfig or None"):
            resolver.client_config({"model": "m"}, {})

    def test_invalid_values_raise_validation_errors(self, resolver):
        with pytest.raises(ValidationError):
            resolver.client_config(None, {"temperature": -1})

    def test_result_does_not_share_state_with_the_input(self, resolver):
        source = LLMConfig(others={"a": [1]}, tools=[SPEC])
        config = resolver.client_config(source, {})
        config.others["a"].append(2)
        config.tools.tool_list[0].description = "Changed."
        assert source.others == {"a": [1]} and source.tools.tool_list[0].description == "Get weather."

    def test_keyword_extras_are_copied(self, resolver):
        options = {"headers": {"x": "1"}}
        config = resolver.client_config(None, {"headers": options["headers"]})
        config.client_options["headers"]["x"] = "2"
        assert options == {"headers": {"x": "1"}}

    def test_opaque_keyword_objects_are_shared(self, resolver):
        handle = object()
        assert resolver.client_config(None, {"http_client": handle}).client_options["http_client"] is handle


class TestThinkingNormalization:
    def test_provider_thinking_alias_becomes_thinking(self, resolver):
        config = resolver.client_config(None, {"think": "low"})
        assert config.thinking_effort == "low"

    def test_explicit_config_thinking_fields_win(self, resolver):
        config = resolver.call_config(LLMConfig(model="m"), LLMConfig(thinking_effort="low"),
                                      {"thinking": {"effort": "high", "mode": "enabled"}})
        assert (config.thinking_effort, config.thinking_mode) == ("low", "enabled")

    def test_dictionary_and_flat_forms_merge(self, resolver):
        config = resolver.call_config(LLMConfig(model="m"), None,
                                      {"thinking": {"others": {"summary": "auto"}}, "thinking_effort": "low"})
        assert config.thinking == ThinkingConfig(others={"summary": "auto"}) and config.thinking_effort == "low"


class TestCallConfig:
    defaults = LLMConfig(model="default", temperature=0.5, system_prompt="sys", api_key="k", timeout=30,
                         client_options={"base_url": "x"})

    def test_call_values_override_defaults(self, resolver):
        config = resolver.call_config(self.defaults, None, {"user_prompt": "hi", "temperature": 0.1})
        assert (config.model, config.temperature, config.system_prompt.text, config.user_prompt.text) == (
            "default", 0.1, "sys", "hi")

    def test_unknown_keywords_become_others_with_provider_spelling(self, resolver):
        config = resolver.call_config(self.defaults, None, {"metadata": {"a": 1}})
        assert config.others == {"metadata": {"a": 1}}

    def test_generation_aliases_are_accepted_but_client_aliases_are_not(self, resolver):
        config = resolver.call_config(self.defaults, None, {"instructions": "new", "model_name": "x"})
        assert config.system_prompt.text == "new"
        assert config.others == {"model_name": "x"}

    def test_explicit_none_clears_optional_settings(self, resolver):
        config = resolver.call_config(self.defaults, None, {"temperature": None, "system_prompt": None})
        assert config.temperature is None and config.system_prompt is None

    @pytest.mark.parametrize("name", ["model", "api_key", "timeout", "max_retries"])
    def test_none_keeps_the_default_for_model_and_connection(self, resolver, name):
        config = resolver.call_config(self.defaults, None, {name: None})
        assert getattr(config, name) == getattr(self.defaults, name)

    @pytest.mark.parametrize("name, value", [("asynchronous", True), ("client_options", {})])
    def test_client_only_settings_cannot_change_per_call(self, resolver, name, value):
        with pytest.raises(ValueError, match=f"{name} is a client setting"):
            resolver.call_config(self.defaults, None, {name: value})
        with pytest.raises(ValueError, match="client setting"):
            resolver.call_config(self.defaults, LLMConfig(**{name: value}), {})

    def test_both_client_only_settings_are_named(self, resolver):
        with pytest.raises(ValueError, match="asynchronous, client_options is a client setting"):
            resolver.call_config(self.defaults, None, {"asynchronous": True, "client_options": {}})

    def test_a_model_is_required(self, resolver):
        with pytest.raises(ValueError, match="A model is required"):
            resolver.call_config(LLMConfig(), None, {"user_prompt": "hi"})
        assert resolver.call_config(LLMConfig(), None, {"model": "m"}).model == "m"

    def test_tool_settings_need_tools(self, resolver):
        with pytest.raises(ValueError, match="need tools, on this call or the client"):
            resolver.call_config(self.defaults, None, {"tool_choice": "required"})
        with pytest.raises(ValueError, match="need tools"):
            resolver.call_config(self.defaults, None, {"parallel_tool_calls": True})

    def test_call_tool_settings_apply_to_client_tools(self, resolver):
        defaults = LLMConfig(model="m", tools=[SPEC])
        config = resolver.call_config(defaults, None, {"tool_choice": "none"})
        assert config.tools.tool_choice == "none"
        assert defaults.tools.tool_choice == "auto"

    def test_call_tools_replace_client_tools(self, resolver):
        defaults = LLMConfig(model="m", tools=Tools(tool_list=[SPEC], tool_choice="none"))
        other = {**SPEC, "name": "other"}
        config = resolver.call_config(defaults, None, {"tools": [other]})
        assert [spec.name for spec in config.tools.tool_list] == ["other"] and config.tools.tool_choice == "auto"

    def test_tools_can_be_removed_per_call(self, resolver):
        defaults = LLMConfig(model="m", tools=[SPEC])
        assert resolver.call_config(defaults, None, {"tools": None}).tools is None

    def test_defaults_are_never_mutated(self, resolver):
        defaults = LLMConfig(model="m", others={"a": [1]}, tools=[SPEC])
        snapshot = defaults.model_copy(deep=True)
        config = resolver.call_config(defaults, None, {"others": {"b": 2}, "temperature": 0.3})
        config.tools.tool_list[0].description = "Changed."
        assert defaults == snapshot

    def test_explicit_others_block_keyword_extras(self, resolver):
        config = resolver.call_config(self.defaults, LLMConfig(others={"a": 1}), {"b": 2})
        assert config.others == {"a": 1}

    def test_keyword_extras_are_added_to_the_default_others(self, resolver):
        defaults = LLMConfig(model="m", others={"a": 1, "keep": {"x": 1}})
        assert resolver.call_config(defaults, None, {"b": 2}).others == {"a": 1, "keep": {"x": 1}, "b": 2}
        assert resolver.call_config(defaults, None, {}).others == {"a": 1, "keep": {"x": 1}}

    def test_a_keyword_extra_overrides_the_default_with_the_same_name(self, resolver):
        defaults = LLMConfig(model="m", others={"a": 1, "keep": 1})
        assert resolver.call_config(defaults, None, {"a": 2}).others == {"a": 2, "keep": 1}

    def test_an_explicit_others_keyword_replaces_the_defaults(self, resolver):
        defaults = LLMConfig(model="m", others={"a": 1})
        assert resolver.call_config(defaults, None, {"others": {"b": 2}}).others == {"b": 2}
        assert resolver.call_config(defaults, None, {"others": {"b": 2}, "c": 3}).others == {"b": 2, "c": 3}
        assert resolver.call_config(defaults, None, {"others": {}}).others == {}

    def test_merged_extras_do_not_change_or_share_the_defaults(self, resolver):
        defaults = LLMConfig(model="m", others={"a": {"nested": [1]}})
        config = resolver.call_config(defaults, None, {"b": 2})
        config.others["a"]["nested"].append(2)
        assert defaults.others == {"a": {"nested": [1]}}

    def test_client_extras_still_start_empty(self, resolver):
        assert resolver.client_config(None, {"a": 1}).client_options == {"a": 1}

    def test_two_spellings_of_one_setting_raise(self, resolver):
        for parameters in ({"instructions": "a", "system_prompt": "b"}, {"system_prompt": "b", "instructions": "a"},
                           {"max_tokens": 1, "max_output_tokens": 2}):
            with pytest.raises(ValueError, match="both set .* pass only one"):
                resolver.call_config(self.defaults, None, parameters)
            with pytest.raises(ValueError, match="both set .* pass only one"):
                resolver.client_config(None, parameters)

    def test_the_error_names_both_spellings(self, resolver):
        with pytest.raises(ValueError, match="^'instructions' and 'system_prompt' both set 'system_prompt'; pass only one$"):
            resolver.call_config(self.defaults, None, {"instructions": "a", "system_prompt": "b"})

    def test_provider_thinking_spelling_and_thinking_conflict(self, resolver):
        with pytest.raises(ValueError, match="'think' and 'thinking' both set 'thinking'"):
            resolver.call_config(self.defaults, None, {"think": "low", "thinking": "high"})

    def test_config_wins_over_a_keyword_spelling_without_error(self, resolver):
        config = resolver.call_config(self.defaults, LLMConfig(system_prompt="from config"),
                                      {"instructions": "ignored", "system_prompt": "also ignored"})
        assert config.system_prompt.text == "from config"

    def test_stream_keyword_is_rejected(self, resolver):
        with pytest.raises(ValueError, match="Streaming is selected"):
            resolver.call_config(self.defaults, None, {"stream": False})
