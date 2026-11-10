import pytest
from pydantic import ValidationError

from hiyal.models.bases import Translator
from hiyal.models.schemas import (
    LLMConfig,
    ProviderInfo,
    ResponseContent,
    ThinkingConfig,
    ToolCallContent,
    UnsupportedThinkingError,
)
from hiyal.prompts import Audio, Document, Image, UserPrompt
from hiyal.tools import AnthropicToolSpec, OpenAIToolSpec, ToolCall, ToolSpec, Tools

from .fakes import FakeTranslator

INFO = ProviderInfo(provider="openai", display_name="Fake", prompt_media=frozenset({"images"}),
                    generation_adapter={"instructions": "system_prompt", "max_tokens": "max_output_tokens"})
SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}


class RecordingTranslator(FakeTranslator):
    """Records apply_thinking calls instead of translating them."""

    def __init__(self, info=INFO):
        super().__init__(info)
        self.thinking_calls = []

    def apply_thinking(self, params, thinking, model):
        self.thinking_calls.append((dict(params), thinking, model))
        params["reasoning"] = {"effort": thinking.effort}


@pytest.fixture
def translator():
    return RecordingTranslator()


def config(**values):
    values.setdefault("user_prompt", "hi")
    return LLMConfig(model="m", **values)


class TestValidate:
    def test_text_prompt_passes(self, translator):
        translator.validate(config())

    def test_missing_user_prompt_raises_a_validation_error(self, translator):
        with pytest.raises(ValidationError, match="user_prompt"):
            translator.validate(LLMConfig(model="m"))

    def test_supported_media_passes(self, translator):
        translator.validate(config(user_prompt=UserPrompt(text="x", images=Image(mime_type="image/png", data=b"x"))))

    def test_unsupported_media_is_listed_in_order(self, translator):
        prompt = UserPrompt(text="x", audio=Audio(mime_type="audio/wav", data=b"x"),
                            documents=Document(mime_type="application/pdf", data=b"x"))
        with pytest.raises(NotImplementedError, match="^Fake does not support user_prompt audio, documents$"):
            translator.validate(config(user_prompt=prompt))


class TestBuildRequest:
    def build(self, translator, **values):
        return Translator.build_request(translator, config(**values))

    def test_fields_are_renamed_and_none_values_dropped(self, translator):
        params = self.build(translator, system_prompt="sys", max_output_tokens=9, temperature=None)
        assert params["instructions"] == {"text": "sys", "kind": "system"}
        assert params["max_tokens"] == 9 and "temperature" not in params and "max_output_tokens" not in params

    def test_connection_client_and_tool_fields_are_never_sent(self, translator):
        params = self.build(translator, api_key="k", timeout=3, client_options={"a": 1}, asynchronous=False)
        for name in ("api_key", "timeout", "max_retries", "client_options", "asynchronous", "tools",
                     "tool_choice", "parallel_tool_calls", "thinking", "thinking_mode"):
            assert name not in params

    def test_others_are_merged_but_common_fields_win(self, translator):
        params = self.build(translator, temperature=0.2, others={"temperature": 0.9, "metadata": {"a": 1}})
        assert (params["temperature"], params["metadata"]) == (0.2, {"a": 1})
        assert "others" not in params

    def test_tools_use_the_tool_parameters_hook(self, translator):
        with pytest.raises(NotImplementedError, match="Fake does not support normalized tool configuration"):
            self.build(translator, tools=[SPEC])

    def test_thinking_is_applied_only_when_requested(self, translator):
        self.build(translator)
        assert translator.thinking_calls == []
        params = self.build(translator, thinking_effort="low", thinking={"others": {"summary": "auto"}})
        assert params["reasoning"] == {"effort": "low"}
        _, thinking, model = translator.thinking_calls[0]
        assert thinking == ThinkingConfig(effort="low", others={"summary": "auto"}) and model == "m"

    def test_base_has_no_thinking_adapter(self):
        with pytest.raises(ValueError, match="No thinking adapter for provider 'openai'"):
            Translator.build_request(FakeTranslator(INFO), config(thinking_effort="low"))


class TestAdaptThinking:
    def test_enabled_with_no_effort_conflicts(self, translator):
        with pytest.raises(ValueError, match="mode='enabled' conflicts with effort='none'") as caught:
            translator.adapt_thinking({}, ThinkingConfig(mode="enabled", effort="none"), "m")
        assert not isinstance(caught.value, UnsupportedThinkingError)

    def test_budget_needs_provider_support(self, translator):
        translator.supports_thinking_budget = False
        with pytest.raises(UnsupportedThinkingError, match="openai model 'gpt-x': thinking_budget is not supported"):
            translator.adapt_thinking({}, ThinkingConfig(budget=10), "gpt-x")
        translator.adapt_thinking({}, ThinkingConfig(budget=10, mode="disabled"), "gpt-x")

    def test_model_name_is_normalized_and_params_are_not_mutated(self, translator):
        params = {"a": 1}
        result = translator.adapt_thinking(params, ThinkingConfig(effort="low"), "vendor/Big-Model")
        assert translator.thinking_calls[0][2] == "big-model"
        assert params == {"a": 1} and result == {"a": 1, "reasoning": {"effort": "low"}}

    def test_missing_model_becomes_an_empty_name(self, translator):
        translator.adapt_thinking({}, ThinkingConfig(effort="low"), None)
        assert translator.thinking_calls[0][2] == ""


class TestToolDefinitions:
    def test_common_specs_are_converted_to_the_provider_type(self, translator):
        declarations = translator.tool_definitions(Tools(tool_list=[SPEC]), OpenAIToolSpec)
        assert declarations[0]["type"] == "function" and declarations[0]["strict"] is True

    def test_provider_specs_are_used_as_is(self, translator):
        spec = OpenAIToolSpec(**SPEC, strict=False, allowed_callers=["direct"])
        declaration = translator.tool_definitions(Tools(tool_list=[spec]), OpenAIToolSpec)[0]
        assert declaration["strict"] is False and declaration["allowed_callers"] == ["direct"]

    def test_foreign_provider_specs_are_rejected(self, translator):
        with pytest.raises(ValueError, match="Fake tools must use ToolSpec or OpenAIToolSpec"):
            translator.tool_definitions(Tools(tool_list=[AnthropicToolSpec(**SPEC)]), OpenAIToolSpec)

    def test_explicit_unset_fields_are_not_forced(self, translator):
        spec = ToolSpec(**SPEC, strict=None)
        assert translator.tool_definitions(Tools(tool_list=[spec]), OpenAIToolSpec)[0]["strict"] is None


class TestPromptHelpers:
    def test_prompt_texts(self, translator):
        assert translator.prompt_texts(config(system_prompt="sys")) == ("sys", "hi")
        assert translator.prompt_texts(config()) == (None, "hi")

    def test_prompt_texts_validates(self, translator):
        with pytest.raises(ValidationError):
            translator.prompt_texts(LLMConfig(model="m"))

    def test_chat_messages(self, translator):
        assert translator.chat_messages(config()) == [{"role": "user", "content": "hi"}]
        assert translator.chat_messages(config(system_prompt="sys")) == [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    @pytest.mark.parametrize("modalities", [None, ["TEXT"]])
    def test_require_text_only_accepts_text(self, translator, modalities):
        params = {"response_modalities": modalities, "x": 1}
        translator.require_text_only(params)
        assert params == {"x": 1}

    @pytest.mark.parametrize("modalities", [["IMAGE"], ["TEXT", "AUDIO"], []])
    def test_require_text_only_rejects_other_modalities(self, translator, modalities):
        with pytest.raises(ValueError, match="Fake only supports the TEXT response modality"):
            translator.require_text_only({"response_modalities": modalities})

    def test_dict_param_names_the_provider(self, translator):
        with pytest.raises(TypeError, match="^Fake 'options' must be a dictionary$"):
            translator.dict_param({"options": 1}, "options")


class TestResponseHelpers:
    MAP = {"stop": "stop", "length": "length"}

    def test_finish_reason(self):
        assert Translator.finish_reason(None, self.MAP) is None
        assert Translator.finish_reason("length", self.MAP) == "length"
        assert Translator.finish_reason("weird", self.MAP) == "unknown"
        assert Translator.finish_reason("weird", self.MAP, default="other") == "other"

    def test_stop_with_a_tool_call_becomes_tool_call(self):
        call = ToolCallContent(data=ToolCall.normalize(name="f", arguments={}, raw=None, complete=True))
        assert Translator.finish_reason("stop", self.MAP, contents=[call]) == "tool_call"
        assert Translator.finish_reason("length", self.MAP, contents=[call]) == "length"

    def test_status(self):
        assert Translator.status("stop", {"length"}) == "completed"
        assert Translator.status("length", {"length"}) == "incomplete"
        assert Translator.status(None, {"length"}) == "completed"
        assert Translator.status("stop", {"length"}, interrupted=True) == "incomplete"

    def test_calculated_timing(self):
        assert Translator.calculated_timing(None) is None
        timing = Translator.calculated_timing(1.5)
        assert (timing.source, timing.total_seconds) == ("calculated", 1.5)

    def test_messages_give_reasoning_no_finish_reason(self):
        contents = [ResponseContent(content_type="reasoning", text="r"), ResponseContent(content_type="text", text="t")]
        messages = Translator.messages(["a", "b"], contents, finish_reason="stop", role="tool")
        assert [(m.id, m.role, m.finish_reason) for m in messages] == [("a", "other", None), ("b", "tool", "stop")]

    def test_messages_need_one_id_per_content(self):
        with pytest.raises(ValueError):
            Translator.messages(["a"], [ResponseContent(content_type="text"), ResponseContent(content_type="text")],
                                finish_reason=None)

    def test_text_joins_text_parts_only(self):
        contents = [ResponseContent(content_type="text", text="a"), ResponseContent(content_type="reasoning", text="r"),
                    ResponseContent(content_type="text"), ResponseContent(content_type="text", text="b")]
        assert Translator.text(contents) == "ab"
        assert Translator.text([]) == ""
