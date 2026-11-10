from datetime import datetime, timezone

import pytest
from openai.types.responses import Response

from hiyal.models import OpenAILLM
from hiyal.models.schemas import UnsupportedThinkingError
from hiyal.prompts import Audio, Document, Image, UserPrompt
from hiyal.tools import AnthropicToolSpec

from tests.support import build_request, openai_function_call, openai_message, openai_response

SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 28


@pytest.fixture
def llm():
    return OpenAILLM(model="gpt-5.5", api_key="test-key")


def parse(llm, payload, **kwargs):
    return llm.translator.parse_response(Response.model_validate(payload), **kwargs)


class TestResponsesRequest:
    def test_text_prompt(self, llm):
        assert build_request(llm, system_prompt="Be brief.", temperature=0.2) == {
            "model": "gpt-5.5", "temperature": 0.2, "instructions": "Be brief.", "input": "Hi"}

    def test_no_system_prompt_sends_no_instructions(self, llm):
        assert "instructions" not in build_request(llm)

    def test_provider_spelling_is_accepted(self, llm):
        params = build_request(llm, instructions="Be brief.", input="Hello")
        assert (params["instructions"], params["input"]) == ("Be brief.", "Hello")

    def test_generation_settings_and_extras(self, llm):
        params = build_request(llm, max_output_tokens=50, top_p=0.9, seed=1, metadata={"a": "b"},
                               stop_sequences=["END"])
        assert params["max_output_tokens"] == 50 and params["metadata"] == {"a": "b"}
        assert params["stop_sequences"] == ["END"]

    def test_media_prompt_becomes_typed_input_parts(self, llm):
        prompt = UserPrompt(text="Describe", images=[
            Image(mime_type="image/png", data=b"x"), Image.from_url("https://e.com/a.png"),
            Image(mime_type="image/png", file_id="file-i")],
            documents=[Document(mime_type="application/pdf", data=b"%PDF", filename="r.pdf"),
                       Document.from_url("https://e.com/r.pdf"), Document(mime_type="application/pdf", file_id="file-d")])
        [message] = build_request(llm, user_prompt=prompt)["input"]
        assert message["role"] == "user"
        assert message["content"] == [
            {"type": "input_text", "text": "Describe"},
            {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,eA=="},
            {"type": "input_image", "detail": "auto", "image_url": "https://e.com/a.png"},
            {"type": "input_image", "detail": "auto", "file_id": "file-i"},
            {"type": "input_file", "filename": "r.pdf", "file_data": "data:application/pdf;base64,JVBERg=="},
            {"type": "input_file", "file_url": "https://e.com/r.pdf"},
            {"type": "input_file", "file_id": "file-d"},
        ]

    def test_inline_document_without_filename_gets_a_generic_name(self, llm):
        prompt = UserPrompt(text="x", documents=Document(mime_type="application/pdf", data=b"%PDF"))
        part = build_request(llm, user_prompt=prompt)["input"][0]["content"][1]
        assert part["filename"] == "document.pdf"

    def test_tools(self, llm):
        params = build_request(llm, tools=[SPEC], tool_choice="required", parallel_tool_calls=True)
        assert params["tool_choice"] == "required" and params["parallel_tool_calls"] is True
        assert params["tools"][0]["name"] == "get_weather" and params["tools"][0]["strict"] is True

    def test_tools_default_to_no_parallel_calls(self, llm):
        params = build_request(llm, tools=[SPEC])
        assert (params["tool_choice"], params["parallel_tool_calls"]) == ("auto", False)

    def test_foreign_tool_specs_are_rejected(self, llm):
        with pytest.raises(ValueError, match="OpenAI tools must use ToolSpec or OpenAIToolSpec"):
            build_request(llm, tools=[AnthropicToolSpec(**SPEC)])

    def test_native_tools_can_still_be_passed_as_extras(self, llm):
        native = [{"type": "web_search"}]
        assert build_request(llm, others={"tools": native})["tools"] == native

    def test_image_output_adds_the_image_generation_tool(self, llm):
        params = build_request(llm, response_modalities=["TEXT", "IMAGE"], image_generation={"size": "1024x1024"},
                               tools=[SPEC])
        assert params["tools"][-1] == {"size": "1024x1024", "type": "image_generation"}
        assert params["tools"][0]["name"] == "get_weather"
        assert "response_modalities" not in params and "image_generation" not in params

    def test_text_modality_is_dropped(self, llm):
        assert "response_modalities" not in build_request(llm, response_modalities=["TEXT"])

    def test_image_generation_options_need_image_output(self, llm):
        with pytest.raises(ValueError, match="image_generation options need response_modalities"):
            build_request(llm, image_generation={"size": "1024x1024"})


class TestThinking:
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_effort(self, llm, effort):
        assert build_request(llm, thinking_effort=effort)["reasoning"] == {"effort": effort}

    def test_disabled_becomes_effort_none(self, llm):
        assert build_request(llm, thinking_mode="disabled", thinking_effort="high")["reasoning"] == {"effort": "none"}

    @pytest.mark.parametrize("model", ["o3", "o4-mini", "o1-pro", "gpt-6-astra", "openai/O3"])
    def test_models_that_always_reason_cannot_disable_it(self, llm, model):
        with pytest.raises(UnsupportedThinkingError, match="thinking cannot be disabled"):
            build_request(llm, model=model, thinking_mode="disabled")
        with pytest.raises(UnsupportedThinkingError, match="thinking cannot be disabled"):
            build_request(llm, model=model, thinking_effort="none")

    def test_enabled_defaults_to_medium_effort(self, llm):
        assert build_request(llm, thinking_mode="enabled")["reasoning"] == {"effort": "medium"}
        assert build_request(llm, thinking_mode="enabled", thinking_effort="low")["reasoning"] == {"effort": "low"}

    def test_enabled_with_native_none_effort_conflicts(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="enabled mode conflicts"):
            build_request(llm, thinking_mode="enabled", thinking={"others": {"effort": "none"}})

    def test_native_modes_and_options_are_kept(self, llm):
        params = build_request(llm, thinking_mode="auto", thinking={"others": {"summary": "auto"}},
                               others={"reasoning": {"generate_summary": "concise"}})
        assert params["reasoning"] == {"generate_summary": "concise", "summary": "auto", "mode": "auto"}

    def test_budget_is_not_supported(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="thinking_budget is not supported"):
            build_request(llm, thinking_budget=1024)
        assert "reasoning" in build_request(llm, thinking_budget=1024, thinking_mode="disabled")

    def test_non_dict_native_reasoning_raises(self, llm):
        with pytest.raises(TypeError, match="reasoning must be a dictionary"):
            build_request(llm, thinking_effort="low", others={"reasoning": "high"})


class TestChatCompletionsRouting:
    def audio_prompt(self, **values):
        return UserPrompt(text="When?", audio=Audio(mime_type="audio/wav", data=WAV), **values)

    def test_audio_prompt_becomes_a_chat_request(self, llm):
        params = build_request(llm, system_prompt="sys", user_prompt=self.audio_prompt(), max_output_tokens=20,
                               stop_sequences=["X"], thinking_effort="low")
        assert "input" not in params and "instructions" not in params
        assert params["messages"][0] == {"role": "system", "content": "sys"}
        content = params["messages"][1]["content"]
        assert content[0] == {"type": "text", "text": "When?"}
        assert content[1]["type"] == "input_audio" and content[1]["input_audio"]["format"] == "wav"
        assert (params["max_completion_tokens"], params["stop"], params["reasoning_effort"]) == (20, ["X"], "low")
        assert "reasoning" not in params and "max_output_tokens" not in params

    def test_audio_output_requests_spoken_output_with_a_default_voice(self, llm):
        params = build_request(llm, response_modalities=["AUDIO"])
        assert params["modalities"] == ["text", "audio"]
        assert params["audio"] == {"voice": "alloy", "format": "wav"}
        assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_explicit_audio_settings_are_kept(self, llm):
        params = build_request(llm, response_modalities=["AUDIO"], audio={"voice": "verse", "format": "mp3"})
        assert params["audio"] == {"voice": "verse", "format": "mp3"}

    def test_reasoning_options_other_than_effort_are_rejected(self, llm):
        with pytest.raises(ValueError, match="only takes a reasoning effort"):
            build_request(llm, user_prompt=self.audio_prompt(), thinking={"others": {"summary": "auto"}})

    def test_tools_are_not_supported_with_audio(self, llm):
        with pytest.raises(NotImplementedError, match="tools are not supported with audio"):
            build_request(llm, user_prompt=self.audio_prompt(), tools=[SPEC])

    def test_images_cannot_be_returned_with_audio(self, llm):
        with pytest.raises(ValueError, match="cannot return images together with audio"):
            build_request(llm, user_prompt=self.audio_prompt(), response_modalities=["IMAGE"])
        with pytest.raises(ValueError, match="cannot return images together with audio"):
            build_request(llm, response_modalities=["AUDIO", "IMAGE"])

    def test_chat_media_rules(self, llm):
        with pytest.raises(ValueError, match="cannot take an image by file_id"):
            build_request(llm, user_prompt=self.audio_prompt(images=Image(mime_type="image/png", file_id="f")))
        with pytest.raises(ValueError, match="cannot take a document by URL"):
            build_request(llm, user_prompt=self.audio_prompt(documents=Document.from_url("https://e.com/r.pdf")))
        with pytest.raises(ValueError, match="must be WAV or MP3, got 'audio/ogg'"):
            build_request(llm, user_prompt=UserPrompt(text="x", audio=Audio(mime_type="audio/ogg", data=b"x")))
        with pytest.raises(ValueError, match="must be inline data"):
            build_request(llm, user_prompt=UserPrompt(text="x", audio=Audio.from_url("https://e.com/a.wav")))

    def test_chat_media_parts(self, llm):
        prompt = self.audio_prompt(images=Image.from_url("https://e.com/a.png"),
                                   documents=[Document(mime_type="application/pdf", file_id="file-d"),
                                              Document(mime_type="application/pdf", data=b"%PDF", filename="r.pdf")])
        content = build_request(llm, user_prompt=prompt)["messages"][0]["content"]
        assert content[1] == {"type": "image_url", "image_url": {"url": "https://e.com/a.png"}}
        assert content[2] == {"type": "file", "file": {"file_id": "file-d"}}
        assert content[3]["file"]["filename"] == "r.pdf"
        assert content[4]["type"] == "input_audio"

    @pytest.mark.parametrize("mime, fmt", [("audio/wav", "wav"), ("audio/x-wav", "wav"), ("audio/mpeg", "mp3"),
                                           ("audio/mp3", "mp3")])
    def test_audio_formats(self, llm, mime, fmt):
        prompt = UserPrompt(text="x", audio=Audio(mime_type=mime, data=b"x"))
        assert build_request(llm, user_prompt=prompt)["messages"][0]["content"][1]["input_audio"]["format"] == fmt


class TestResponsesParsing:
    def test_text_usage_timing_and_metadata(self, llm):
        result = parse(llm, openai_response([openai_message("Hello.")]))
        assert (result.text, result.provider, result.id_, result.model, result.status) == (
            "Hello.", "openai", "resp_1", "gpt-5.5", "completed")
        assert result.created_at == datetime.fromtimestamp(1792499155, tz=timezone.utc)
        assert result.usage.model_dump() == {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2,
                                             "total_tokens": 15, "cache_read_tokens": 1, "cache_write_tokens": 0}
        assert (result.timing.source, result.timing.total_seconds) == ("calculated", 2)
        assert [(m.id, m.role, m.finish_reason) for m in result.messages] == [("msg_1", "assistant", None)]

    def test_missing_usage_and_completion_time(self, llm):
        result = parse(llm, openai_response([openai_message()], usage=None, completed_at=None))
        assert result.usage is None and result.timing is None

    def test_reasoning_summary_and_content_are_joined(self, llm):
        item = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "a"}],
                "content": [{"type": "reasoning_text", "text": "b"}]}
        empty = {"type": "reasoning", "id": "rs_2", "summary": []}
        result = parse(llm, openai_response([item, empty, openai_message()]))
        assert [(m.content.content_type, m.content.text, m.role) for m in result.messages[:2]] == [
            ("reasoning", "a\nb", "other"), ("reasoning", None, "other")]
        assert result.reasoning == "a\nb"

    def test_refusal(self, llm):
        item = {"type": "message", "id": "msg_r", "role": "assistant", "status": "completed",
                "content": [{"type": "refusal", "refusal": "No."}]}
        message = parse(llm, openai_response([item])).messages[0]
        assert (message.content.content_type, message.content.text) == ("refusal", "No.")

    def test_function_calls_in_order_with_issues_kept_local(self, llm):
        result = parse(llm, openai_response([
            openai_function_call(0), openai_function_call(1, arguments="not json"),
            openai_function_call(2, arguments="{}"), openai_function_call(3, arguments='{"a":', status="incomplete"),
        ]))
        calls = result.tool_calls
        assert [(c.name, c.arguments, c.status) for c in calls] == [
            ("get_weather", {"city": "Istanbul"}, "complete"), ("get_weather", None, "invalid"),
            ("get_weather", {}, "complete"), ("get_weather", None, "incomplete")]
        assert calls[1].issue.code == "invalid_json"
        assert calls[0].raw.call_id == "call_0" and [m.id for m in result.messages] == ["fc_0", "fc_1", "fc_2", "fc_3"]

    def test_absent_item_status_uses_the_response_status(self, llm):
        done = parse(llm, openai_response([openai_function_call(status=None)]))
        cut = parse(llm, openai_response([openai_function_call(status=None)], status="incomplete",
                                         incomplete_reason="max_output_tokens"))
        assert done.tool_calls[0].status == "complete" and cut.tool_calls[0].status == "incomplete"

    def test_image_generation_output(self, llm):
        item = {"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": "aGVsbG8=",
                "output_format": "webp", "revised_prompt": "a cat"}
        result = parse(llm, openai_response([item]))
        assert result.messages[0].content.text == "a cat"
        [image] = result.images
        assert (image.mime_type, image.data) == ("image/webp", b"hello")

    def test_image_generation_without_result_is_other(self, llm):
        item = {"type": "image_generation_call", "id": "ig_1", "status": "generating", "result": None}
        assert parse(llm, openai_response([item])).messages[0].content.content_type == "other"

    def test_unknown_items_are_other(self, llm):
        item = {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "q"}}
        message = parse(llm, openai_response([item])).messages[0]
        assert (message.id, message.role, message.content.content_type) == ("ws_1", "other", "other")

    @pytest.mark.parametrize("status, expected", [("queued", "pending"), ("in_progress", "in_progress"),
                                                  ("failed", "failed"), ("cancelled", "cancelled"),
                                                  ("incomplete", "incomplete")])
    def test_status_mapping(self, llm, status, expected):
        assert parse(llm, openai_response([], status=status)).status == expected

    @pytest.mark.parametrize("reason, finish", [("max_output_tokens", "length"), ("content_filter", "content_filter"),
                                                ("max_messages", "other"), (None, "unknown")])
    def test_incomplete_finish_reasons(self, llm, reason, finish):
        result = parse(llm, openai_response([openai_message()], status="incomplete", incomplete_reason=reason))
        assert result.messages[0].finish_reason == finish

    def test_error_details(self, llm):
        result = parse(llm, openai_response([], status="failed", error={"code": "server_error", "message": "boom"}))
        assert (result.error.code, result.error.message) == ("server_error", "boom")

    def test_raw_is_kept(self, llm):
        raw = Response.model_validate(openai_response([openai_message()]))
        assert llm.translator.parse_response(raw).raw is raw


class TestEdgeCases:
    def test_unsupported_modalities_through_extras_are_rejected(self, llm):
        # response_modalities is validated on LLMConfig, but others bypasses it.
        with pytest.raises(ValueError, match=r"OpenAI Responses cannot return \['VIDEO'\]"):
            build_request(llm, others={"response_modalities": ["VIDEO"]})
        prompt = UserPrompt(text="x", audio=Audio(mime_type="audio/wav", data=WAV))
        with pytest.raises(ValueError, match=r"OpenAI Chat Completions cannot return \['VIDEO'\]"):
            build_request(llm, user_prompt=prompt, others={"response_modalities": ["VIDEO"]})

    def test_empty_native_thinking_adds_no_reasoning(self, llm):
        assert "reasoning" not in build_request(llm, thinking={"others": {}})

    def test_cancelled_events_with_and_without_a_response(self, llm):
        from types import SimpleNamespace as N
        cancelled = llm.translator.parse_stream_event(
            N(type="response.cancelled", sequence_number=4, response=None), response_id="r1")
        assert (cancelled.event_type, cancelled.status, cancelled.response, cancelled.response_id) == (
            "response.cancelled", "cancelled", None, "r1")
        partial = Response.model_validate(openai_response([openai_message("par")], status="cancelled"))
        with_response = llm.translator.parse_stream_event(
            N(type="response.cancelled", sequence_number=5, response=partial), response_id="r1")
        assert with_response.response.text == "par" and with_response.response.status == "cancelled"

    def test_lifecycle_events_are_skipped(self, llm):
        from types import SimpleNamespace as N
        for event_type in ("response.in_progress", "response.output_item.done", "response.content_part.added"):
            assert llm.translator.parse_stream_event(N(type=event_type, sequence_number=1), response_id=None) is None

    def test_non_function_items_added_are_skipped(self, llm):
        from types import SimpleNamespace as N
        event = N(type="response.output_item.added", sequence_number=1, output_index=0, item=N(type="message"))
        assert llm.translator.parse_stream_event(event, response_id=None) is None
