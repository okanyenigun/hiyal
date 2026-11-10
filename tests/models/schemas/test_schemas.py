import base64
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from hiyal.models.schemas import (
    ContentDeltaEvent,
    LLMConfig,
    LLMResponse,
    LLMStreamEvent,
    ProviderInfo,
    ResponseCancelledEvent,
    ResponseContent,
    ResponseMessage,
    ResponseTerminalEvent,
    ResponseTiming,
    StreamErrorEvent,
    ThinkingConfig,
    TokenUsage,
    ToolCallContent,
    ToolCallDeltaContent,
    UnsupportedThinkingError,
)
from hiyal.models.schemas.fields import (
    CLIENT_ONLY_FIELDS,
    CONFIG_FIELDS,
    CONNECTION_FIELDS,
    DEFAULT_MAX_RETRIES,
    KEEP_DEFAULT_WHEN_NONE,
    NOT_GENERATION_FIELDS,
    THINKING_FIELDS,
    TOOL_FIELDS,
    config_aliases,
)
from hiyal.models.schemas.thinking import flatten_thinking
from hiyal.prompts import Audio, Image
from hiyal.tools import ToolCall, ToolCallDelta

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def response(*contents, **overrides):
    messages = [ResponseMessage(id=f"m{i}", role="assistant", content=content) for i, content in enumerate(contents)]
    values = {"text": "", "messages": messages, "provider": "openai", "id_": "r", "model": "m",
              "created_at": NOW, "status": "completed", "raw": object()}
    values.update(overrides)
    return LLMResponse(**values)


class TestFields:
    def test_every_group_names_real_config_fields(self):
        assert CONFIG_FIELDS == frozenset(LLMConfig.model_fields)
        for group in (CONNECTION_FIELDS, CLIENT_ONLY_FIELDS, THINKING_FIELDS, TOOL_FIELDS,
                      KEEP_DEFAULT_WHEN_NONE, NOT_GENERATION_FIELDS):
            assert group <= CONFIG_FIELDS, group - CONFIG_FIELDS

    def test_default_retries_follow_the_schema(self):
        assert DEFAULT_MAX_RETRIES == LLMConfig().max_retries == 2

    def test_prompts_and_generation_settings_are_sent(self):
        for name in ("model", "user_prompt", "system_prompt", "temperature", "max_output_tokens", "others"):
            assert name not in NOT_GENERATION_FIELDS

    @pytest.mark.parametrize("adapter, expected", [
        ({"max_tokens": "max_output_tokens"}, {"max_tokens": "max_output_tokens"}),
        ({"max_output_tokens": "max_tokens"}, {"max_tokens": "max_output_tokens"}),
        ({"a": "b"}, {"a": "b"}),
        ({}, {}),
    ])
    def test_config_aliases_accepts_either_direction(self, adapter, expected):
        assert config_aliases(adapter) == expected

    def test_config_aliases_accepts_read_only_mappings(self):
        info = ProviderInfo(provider="openai", display_name="OpenAI", generation_adapter={"input": "user_prompt"})
        assert config_aliases(info.generation_adapter) == {"input": "user_prompt"}


class TestProviderInfo:
    def test_mappings_are_read_only_copies(self):
        source = {"input": "user_prompt"}
        info = ProviderInfo(provider="openai", display_name="OpenAI", generation_adapter=source)
        source["other"] = "x"
        assert dict(info.generation_adapter) == {"input": "user_prompt"}
        with pytest.raises(TypeError):
            info.generation_adapter["x"] = "y"
        with pytest.raises(TypeError):
            info.adapter["x"] = "y"

    def test_frozen(self):
        info = ProviderInfo(provider="openai", display_name="OpenAI")
        with pytest.raises(AttributeError):
            info.display_name = "x"
        assert info.prompt_media == frozenset() and dict(info.adapter) == {}


class TestThinkingSchema:
    def test_native_values_are_allowed(self):
        config = ThinkingConfig(effort="xhigh", mode="adaptive", budget=0, others={"budget_tokens": 2048})
        assert (config.effort, config.mode, config.budget) == ("xhigh", "adaptive", 0)

    @pytest.mark.parametrize("bad", [{"effort": ""}, {"mode": "a b"}, {"budget": -1}, {"budget": True},
                                     {"level": "low"}])
    def test_invalid_values_are_rejected(self, bad):
        with pytest.raises(ValidationError):
            ThinkingConfig(**bad)

    def test_unsupported_thinking_error(self):
        error = UnsupportedThinkingError("OpenAI", "o3", "thinking cannot be disabled")
        assert isinstance(error, ValueError)
        assert str(error) == "OpenAI model 'o3': thinking cannot be disabled"
        assert (error.provider, error.model, error.reason) == ("OpenAI", "o3", "thinking cannot be disabled")


class TestFlattenThinking:
    @pytest.mark.parametrize("values", [None, "x", [1], {"model": "m"}])
    def test_values_without_thinking_pass_through(self, values):
        assert flatten_thinking(values) is values

    def test_input_is_not_mutated(self):
        values = {"thinking": {"effort": "low", "others": {"a": 1}}}
        flatten_thinking(values)
        assert values == {"thinking": {"effort": "low", "others": {"a": 1}}}

    def test_only_supplied_controls_become_overrides(self):
        assert flatten_thinking({"thinking": {"effort": "low"}}) == {"thinking_effort": "low"}

    def test_explicit_none_controls_are_overrides(self):
        assert flatten_thinking({"thinking": {"budget": None}}) == {"thinking_budget": None}


class TestResponseModels:
    def test_raw_is_excluded_from_serialization(self):
        result = response(ResponseContent(content_type="text", text="hi"))
        assert "raw" not in result.model_dump() and "raw" not in result.model_dump_json()

    def test_tool_call_content_rejects_fragments(self):
        with pytest.raises(ValidationError):
            ToolCallContent(data=ToolCallDelta(arguments_delta='{"a":'))
        assert ToolCallDeltaContent(data=ToolCallDelta(name="f")).content_type == "tool_call"

    def test_tool_calls_and_has_tool_calls(self):
        call = ToolCall.normalize(name="f", arguments={}, raw=None, complete=True)
        bad = ToolCall.normalize(name=None, arguments={}, raw=None, complete=True)
        untyped = ResponseContent(content_type="tool_call", data={"name": "f"})
        result = response(ResponseContent(content_type="text", text="x"), ToolCallContent(data=call),
                          untyped, ToolCallContent(data=bad))
        assert result.tool_calls == [call, bad] and result.has_tool_calls
        assert response().has_tool_calls is False

    def test_reasoning_joins_readable_parts(self):
        result = response(ResponseContent(content_type="reasoning", text="a"),
                          ResponseContent(content_type="reasoning"),
                          ResponseContent(content_type="text", text="visible"),
                          ResponseContent(content_type="reasoning", text="b"))
        assert result.reasoning == "a\n\nb"
        assert response(ResponseContent(content_type="reasoning", text="")).reasoning is None

    def test_images_and_audio_from_adapter_dicts(self):
        png = b"\x89PNG"
        result = response(
            ResponseContent(content_type="image", data={"mime_type": "image/png", "data": base64.b64encode(png).decode()}),
            ResponseContent(content_type="image", data={"mime_type": "image/jpeg", "uri": "gs://bucket/a.jpg"}),
            ResponseContent(content_type="audio", data={"mime_type": "audio/wav", "data": b"RIFF", "transcript": "hi"}),
            ResponseContent(content_type="audio", data={"mime_type": "audio/pcm", "data": b"\x00", "transcript": ""}),
            ResponseContent(content_type="image", data={"no": "mime"}),
            ResponseContent(content_type="text", text="ignored"),
        )
        assert [(i.data, i.url, i.mime_type) for i in result.images] == [
            (png, None, "image/png"), (None, "gs://bucket/a.jpg", "image/jpeg")]
        assert [(a.data, a.transcript) for a in result.audio] == [(b"RIFF", "hi"), (b"\x00", None)]

    def test_media_objects_are_returned_as_is(self):
        image = Image(mime_type="image/png", data=b"x")
        audio = Audio(mime_type="audio/wav", data=b"x")
        result = response(ResponseContent(content_type="image", data=image),
                          ResponseContent(content_type="audio", data=audio))
        assert result.images == [image] and result.audio == [audio]

    def test_usage_and_timing_reject_negative_values(self):
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1)
        with pytest.raises(ValidationError):
            ResponseTiming(source="provider", total_seconds=-1)
        with pytest.raises(ValidationError):
            ResponseTiming(source="guessed")


class TestStreamEvents:
    adapter = TypeAdapter(LLMStreamEvent)
    common = {"provider": "openai", "raw": None}

    def test_events_are_discriminated_by_type(self):
        assert isinstance(self.adapter.validate_python(
            {**self.common, "event_type": "content.delta", "delta": {"content_type": "text", "text": "x"}}),
            ContentDeltaEvent)
        assert isinstance(self.adapter.validate_python(
            {**self.common, "event_type": "error", "error": {"message": "boom"}}), StreamErrorEvent)
        assert isinstance(self.adapter.validate_python(
            {**self.common, "event_type": "response.cancelled"}), ResponseCancelledEvent)

    def test_terminal_event_carries_the_response(self):
        event = ResponseTerminalEvent(**self.common, event_type="response.completed", status="completed",
                                      response=response())
        assert event.response.status == "completed"

    def test_cancelled_event_defaults(self):
        event = ResponseCancelledEvent(**self.common, event_type="response.cancelled")
        assert (event.status, event.response) == ("cancelled", None)

    @pytest.mark.parametrize("bad", [{"event_type": "content.update"}, {"sequence_number": -1},
                                     {"message_index": -1}, {"provider": "mistral"}])
    def test_invalid_events_are_rejected(self, bad):
        values = {**self.common, "event_type": "content.delta", "delta": {"content_type": "text"}, **bad}
        with pytest.raises(ValidationError):
            self.adapter.validate_python(values)
