from datetime import datetime, timezone

import anthropic
import httpx2
import pytest
from anthropic._models import construct_type
from anthropic.types import Message

from hiyal.models import AnthropicLLM
from hiyal.models.schemas import UnsupportedThinkingError
from hiyal.prompts import Audio, Document, Image, UserPrompt
from hiyal.tools import OpenAIToolSpec

from tests.support import anthropic_message, auth, body, build_request, make_llm, run, sse

SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}
REQUEST_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def llm():
    return AnthropicLLM(model="claude-sonnet-4-5", api_key="test-key")


def parse(llm, payload, **kwargs):
    kwargs.setdefault("created_at", REQUEST_TIME)
    return llm.translator.parse_response(construct_type(type_=Message, value=payload), **kwargs)


class TestRequest:
    def test_text_prompt_with_default_max_tokens(self, llm):
        assert build_request(llm, system_prompt="Be brief.") == {
            "model": "claude-sonnet-4-5", "max_tokens": 1024, "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hi"}]}

    def test_max_tokens_by_either_name(self, llm):
        assert build_request(llm, max_output_tokens=50)["max_tokens"] == 50
        assert build_request(llm, max_tokens=60)["max_tokens"] == 60

    def test_stop_sequences_and_extras(self, llm):
        params = build_request(llm, stop_sequences=["END"], others={"metadata": {"user_id": "u"}}, top_k=5)
        assert params["stop_sequences"] == ["END"] and params["metadata"] == {"user_id": "u"} and params["top_k"] == 5

    def test_media_blocks_come_before_the_text(self, llm):
        prompt = UserPrompt(text="Describe", images=[
            Image(mime_type="image/png", data=b"x"), Image.from_url("https://e.com/a.jpg"),
            Image(mime_type="image/png", file_id="file-i")], documents=[
            Document(mime_type="application/pdf", data=b"%PDF", filename="r.pdf"),
            Document(mime_type="text/plain", data=b"plain text"),
            Document.from_url("https://e.com/r.pdf"),
            Document(mime_type="text/csv", file_id="file-d")])
        content = build_request(llm, user_prompt=prompt)["messages"][0]["content"]
        assert content == [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
            {"type": "image", "source": {"type": "url", "url": "https://e.com/a.jpg"}},
            {"type": "image", "source": {"type": "file", "file_id": "file-i"}},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="},
             "title": "r.pdf"},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain text"}},
            {"type": "document", "source": {"type": "url", "url": "https://e.com/r.pdf"}},
            {"type": "document", "source": {"type": "file", "file_id": "file-d"}},
            {"type": "text", "text": "Describe"},
        ]

    def test_unsupported_media(self, llm):
        with pytest.raises(ValueError, match="images must be JPEG, PNG, GIF or WebP, got 'image/bmp'"):
            build_request(llm, user_prompt=UserPrompt(text="x", images=Image(mime_type="image/bmp", data=b"x")))
        with pytest.raises(ValueError, match="documents must be PDF or plain text"):
            build_request(llm, user_prompt=UserPrompt(text="x", documents=Document(mime_type="text/csv", data=b"a")))
        with pytest.raises(NotImplementedError, match="Anthropic does not support user_prompt audio"):
            build_request(llm, user_prompt=UserPrompt(text="x", audio=Audio(mime_type="audio/wav", data=b"x")))

    def test_bmp_by_url_is_passed_through(self, llm):
        prompt = UserPrompt(text="x", images=Image(mime_type="image/bmp", url="https://e.com/a.bmp"))
        assert build_request(llm, user_prompt=prompt)["messages"][0]["content"][0]["source"]["type"] == "url"

    @pytest.mark.parametrize("modalities", [["IMAGE"], ["AUDIO"], ["TEXT", "IMAGE"]])
    def test_text_only_output(self, llm, modalities):
        with pytest.raises(ValueError, match="Anthropic only supports the TEXT response modality"):
            build_request(llm, response_modalities=modalities)

    @pytest.mark.parametrize("choice, parallel, wire", [
        ("auto", False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", True, {"type": "any", "disable_parallel_tool_use": False}),
        ("none", False, {"type": "none"}),
    ])
    def test_tool_choice(self, llm, choice, parallel, wire):
        params = build_request(llm, tools=[SPEC], tool_choice=choice, parallel_tool_calls=parallel)
        assert params["tool_choice"] == wire and params["tools"][0]["input_schema"]["required"] == ["city"]

    def test_foreign_tool_specs_are_rejected(self, llm):
        with pytest.raises(ValueError, match="Anthropic tools must use ToolSpec or AnthropicToolSpec"):
            build_request(llm, tools=[OpenAIToolSpec(**SPEC)])


class TestThinking:
    def test_effort_goes_to_output_config(self, llm):
        params = build_request(llm, model="claude-opus-4-7", thinking_effort="high")
        assert params["output_config"] == {"effort": "high"} and "thinking" not in params

    def test_manual_budget_on_a_legacy_model(self, llm):
        params = build_request(llm, thinking_budget=2048, max_output_tokens=4096)
        assert params["thinking"] == {"budget_tokens": 2048, "type": "enabled"}

    def test_manual_budget_needs_enough_tokens_and_an_explicit_limit(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="budget_tokens\\) >= 1024"):
            build_request(llm, thinking_budget=512, max_output_tokens=4096)
        with pytest.raises(UnsupportedThinkingError, match="explicit max_output_tokens"):
            build_request(llm, thinking_budget=2048)
        with pytest.raises(UnsupportedThinkingError, match=">= 1024"):
            build_request(llm, thinking_mode="enabled", max_output_tokens=4096)

    def test_enabled_on_newer_models_is_adaptive(self, llm):
        assert build_request(llm, model="claude-sonnet-4-6", thinking_mode="enabled")["thinking"] == {"type": "adaptive"}

    def test_manual_budgets_are_not_supported_from_4_7(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="manual thinking budgets are not supported"):
            build_request(llm, model="claude-opus-4-7", thinking_budget=2048, max_output_tokens=4096)

    def test_adaptive_rules(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="adaptive thinking is not supported by this model"):
            build_request(llm, thinking_mode="adaptive")
        with pytest.raises(UnsupportedThinkingError, match="adaptive thinking uses effort, not budget_tokens"):
            build_request(llm, model="claude-sonnet-4-6", thinking_mode="adaptive",
                          thinking={"others": {"budget_tokens": 2048}})
        assert build_request(llm, model="claude-sonnet-4-6", thinking_mode="adaptive",
                             thinking_effort="low")["output_config"] == {"effort": "low"}

    def test_disabled_removes_budget_and_effort(self, llm):
        params = build_request(llm, thinking_mode="disabled", thinking_budget=2048, thinking_effort="high",
                               others={"output_config": {"effort": "low"}})
        assert params["thinking"] == {"type": "disabled"} and "output_config" not in params

    def test_disabled_keeps_other_output_config(self, llm):
        params = build_request(llm, thinking_mode="disabled", others={"output_config": {"effort": "low", "x": 1}})
        assert params["output_config"] == {"x": 1}

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1",
                                       "claude-mythos-1"])
    def test_always_thinking_models_cannot_disable(self, llm, model):
        with pytest.raises(UnsupportedThinkingError, match="cannot be fully disabled"):
            build_request(llm, model=model, thinking_mode="disabled")

    def test_between_tools(self, llm):
        params = build_request(llm, model="claude-sonnet-5-5", thinking_mode="between_tools", thinking_effort="high")
        assert params["thinking"] == {"type": "between_tools"}
        with pytest.raises(UnsupportedThinkingError, match="between_tools is not supported by this model"):
            build_request(llm, model="claude-opus-5-5", thinking_mode="between_tools")
        with pytest.raises(UnsupportedThinkingError, match="does not support xhigh/max"):
            build_request(llm, model="claude-sonnet-5-5", thinking_mode="between_tools", thinking_effort="max")

    def test_native_options_are_merged(self, llm):
        params = build_request(llm, model="claude-sonnet-4-6", thinking_mode="enabled",
                               thinking={"others": {"display": "summarized"}})
        assert params["thinking"] == {"display": "summarized", "type": "adaptive"}

    def test_vendor_prefixed_model_names(self, llm):
        params = build_request(llm, model="anthropic/Claude-Sonnet-4-6", thinking_mode="enabled")
        assert params["thinking"] == {"type": "adaptive"}

    def test_unrecognized_models_are_treated_as_current(self, llm):
        assert build_request(llm, model="custom-model", thinking_mode="enabled")["thinking"] == {"type": "adaptive"}


class TestParsing:
    def test_blocks_become_messages(self, llm):
        result = parse(llm, anthropic_message([
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            {"type": "redacted_thinking", "data": "xxx"},
            {"type": "text", "text": "Hello."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Istanbul"}},
            {"type": "server_tool_use", "id": "st_1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "st_1", "content": []},
            {"type": "brand_new_block"},
        ], stop_reason="tool_use"), elapsed_seconds=2.0)
        types = [m.content.content_type for m in result.messages]
        assert types == ["reasoning", "reasoning", "text", "tool_call", "other", "tool_result", "other"]
        assert result.reasoning == "hmm" and result.text == "Hello."
        assert result.tool_calls[0].arguments == {"city": "Istanbul"} and result.tool_calls[0].status == "complete"
        assert result.messages[5].content.data == {"id": "st_1", "result": []}
        assert result.messages[2].finish_reason == "tool_call" and result.messages[0].finish_reason is None
        assert len({m.id for m in result.messages}) == 7
        assert (result.created_at, result.timing.total_seconds, result.status) == (REQUEST_TIME, 2.0, "completed")

    @pytest.mark.parametrize("stop, finish, status", [
        ("end_turn", "stop", "completed"), ("max_tokens", "length", "incomplete"),
        ("stop_sequence", "stop_sequence", "completed"), ("tool_use", "tool_call", "completed"),
        ("pause_turn", "pause", "completed"), ("refusal", "refusal", "completed"),
        ("model_context_window_exceeded", "context_limit", "incomplete"),
    ])
    def test_stop_reasons(self, llm, stop, finish, status):
        result = parse(llm, anthropic_message(stop_reason=stop))
        assert (result.messages[0].finish_reason, result.status) == (finish, status)

    def test_tool_calls_cut_off_by_max_tokens_are_incomplete(self, llm):
        result = parse(llm, anthropic_message([{"type": "tool_use", "id": "t", "name": "f", "input": {}}],
                                              stop_reason="max_tokens"))
        assert result.tool_calls[0].status == "incomplete"

    def test_usage(self, llm):
        usage = {"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 1,
                 "cache_creation_input_tokens": 2, "output_tokens_details": {"thinking_tokens": 3}}
        result = parse(llm, anthropic_message(usage=usage))
        assert result.usage.model_dump(exclude_none=True) == {"input_tokens": 5, "output_tokens": 7,
                                                              "reasoning_tokens": 3, "cache_read_tokens": 1,
                                                              "cache_write_tokens": 2}

    def test_request_time_defaults_to_now(self, llm):
        result = llm.translator.parse_response(construct_type(type_=Message, value=anthropic_message()))
        assert result.created_at.tzinfo is not None and result.timing is None


def stream_body(*events):
    start = {"type": "message_start", "message": {**anthropic_message([], stop_reason=None),
                                                  "usage": {"input_tokens": 5, "output_tokens": 1}}}
    return sse([start, *events])


def block(index, content):
    return {"type": "content_block_start", "index": index, "content_block": content}


def delta(index, **value):
    return {"type": "content_block_delta", "index": index, "delta": value}


def stop(index):
    return {"type": "content_block_stop", "index": index}


def message_end(reason="end_turn", output_tokens=12):
    return [{"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
             "usage": {"output_tokens": output_tokens}}, {"type": "message_stop"}]


def stream(events_body, **settings):
    llm, requests = make_llm(AnthropicLLM, lambda request: httpx2.Response(
        200, text=events_body, headers={"content-type": "text/event-stream"}), **settings)
    return list(llm.query_stream(user_prompt="Hi")), requests


class TestStreaming:
    def test_text_thinking_and_tool_json(self):
        events, requests = stream(stream_body(
            block(0, {"type": "thinking", "thinking": "", "signature": ""}),
            delta(0, type="thinking_delta", thinking="hm"),
            delta(0, type="signature_delta", signature="sig"),
            stop(0),
            block(1, {"type": "text", "text": ""}), delta(1, type="text_delta", text="Hel"),
            delta(1, type="text_delta", text="lo."), stop(1),
            block(2, {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {}}),
            delta(2, type="input_json_delta", partial_json='{"city":'),
            delta(2, type="input_json_delta", partial_json=""),
            delta(2, type="input_json_delta", partial_json='"Paris"}'), stop(2),
            *message_end("tool_use"),
        ))
        assert body(requests[0])["stream"] is True
        *deltas, terminal = events
        assert [(e.message_index, e.delta.content_type) for e in deltas] == [
            (0, "reasoning"), (1, "text"), (1, "text"), (2, "tool_call"), (2, "tool_call"), (2, "tool_call")]
        assert deltas[3].delta.data.name == "get_weather"
        assert [e.delta.data.arguments_delta for e in deltas[4:]] == ['{"city":', '"Paris"}']
        response = terminal.response
        assert (response.text, response.reasoning, response.id_, response.status) == ("Hello.", "hm", "msg_1", "completed")
        assert response.tool_calls[0].arguments == {"city": "Paris"} and response.tool_calls[0].status == "complete"
        assert response.usage.output_tokens == 12 and response.usage.input_tokens == 5
        assert [m.id for m in response.messages] == [deltas[0].message_id, deltas[1].message_id, deltas[3].message_id]

    def test_tool_with_initial_input_and_no_json(self):
        events, _ = stream(stream_body(
            block(0, {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}}), stop(0), *message_end("tool_use")))
        call = events[-1].response.tool_calls[0]
        assert (call.arguments, call.status) == ({"a": 1}, "complete")

    def test_initial_input_and_streamed_json_conflict(self):
        events, _ = stream(stream_body(
            block(0, {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}}),
            delta(0, type="input_json_delta", partial_json='{"b": 2}'), stop(0), *message_end("tool_use")))
        call = events[-1].response.tool_calls[0]
        assert (call.status, call.issue.code) == ("invalid", "conflicting_arguments")

    def test_malformed_streamed_json(self):
        events, _ = stream(stream_body(
            block(0, {"type": "tool_use", "id": "t", "name": "f", "input": {}}),
            delta(0, type="input_json_delta", partial_json='{"b":'), stop(0), *message_end("tool_use")))
        call = events[-1].response.tool_calls[0]
        assert (call.status, call.issue.code) == ("invalid", "invalid_json")

    def test_max_tokens_keeps_calls_incomplete(self):
        events, _ = stream(stream_body(
            block(0, {"type": "tool_use", "id": "t", "name": "f", "input": {}}),
            delta(0, type="input_json_delta", partial_json='{"a": 1}'), stop(0), *message_end("max_tokens")))
        assert events[-1].response.tool_calls[0].status == "incomplete"
        assert events[-1].status == "incomplete"

    def test_interrupted_stream_is_incomplete(self):
        events, _ = stream(stream_body(
            block(0, {"type": "tool_use", "id": "t", "name": "f", "input": {}}),
            delta(0, type="input_json_delta", partial_json='{"a": 1}')))
        terminal = events[-1]
        assert (terminal.event_type, terminal.response.tool_calls[0].status) == ("response.incomplete", "incomplete")

    def test_stream_without_any_event_still_ends_with_a_terminal_event(self):
        events, _ = stream("")
        assert [e.event_type for e in events] == ["response.incomplete"]
        assert events[0].sequence_number is None and events[0].response.messages == []


class TestEndToEnd:
    def test_request_headers_and_response(self):
        llm, requests = make_llm(AnthropicLLM, lambda r: httpx2.Response(200, json=anthropic_message()))
        result = llm.query(user_prompt="Hi")
        request = requests[0]
        assert (request.url.path, auth(request), body(request)["stream"]) == ("/v1/messages", "test-key", False)
        assert "anthropic-version" in request.headers
        assert result.text == "Hello." and result.timing.source == "calculated"

    def test_per_call_connection_overrides(self):
        llm, requests = make_llm(AnthropicLLM, lambda r: httpx2.Response(200, json=anthropic_message()), timeout=30)
        llm.query(user_prompt="Hi", api_key="call-key", timeout=5)
        llm.query(user_prompt="Hi")
        assert [auth(r) for r in requests] == ["call-key", "test-key"]
        assert [r.extensions["timeout"]["read"] for r in requests] == [5, 30]

    def test_http_errors_surface_as_sdk_errors(self):
        llm, _ = make_llm(AnthropicLLM, lambda r: httpx2.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}))
        with pytest.raises(anthropic.AuthenticationError):
            llm.query(user_prompt="Hi")

    def test_files_upload_and_delete(self):
        def respond(request):
            if request.method == "DELETE":
                return httpx2.Response(200, json={"id": "file_1", "type": "file_deleted"})
            return httpx2.Response(200, json={"id": "file_1", "type": "file", "filename": "r.pdf",
                                              "mime_type": "application/pdf", "size_bytes": 4,
                                              "created_at": "2026-01-01T00:00:00Z"})
        llm, requests = make_llm(AnthropicLLM, respond)
        media = llm.upload_file(b"%PDF", "application/pdf", filename="r.pdf")
        assert type(media) is Document and media.file_id == "file_1"
        llm.delete_file(media)
        assert [(r.method, r.url.path) for r in requests] == [("POST", "/v1/files"), ("DELETE", "/v1/files/file_1")]

    def test_async_query_and_stream(self):
        events_body = stream_body(block(0, {"type": "text", "text": ""}), delta(0, type="text_delta", text="x"),
                                  stop(0), *message_end())

        def respond(request):
            if body(request)["stream"]:
                return httpx2.Response(200, text=events_body, headers={"content-type": "text/event-stream"})
            return httpx2.Response(200, json=anthropic_message())
        llm, _ = make_llm(AnthropicLLM, respond, asynchronous=True)

        async def scenario():
            result = await llm.query_async(user_prompt="Hi")
            events = [e async for e in await llm.query_stream_async(user_prompt="Hi")]
            return result, events
        result, events = run(scenario())
        assert result.text == "Hello." and events[-1].response.text == "x"


class TestEdgeCases:
    def test_ping_and_unknown_events_add_nothing(self):
        events, _ = stream(stream_body({"type": "ping"}, block(0, {"type": "text", "text": "x"}), stop(0),
                                       *message_end()))
        assert [e.event_type for e in events] == ["content.delta", "response.completed"]
        assert events[0].delta.text == "x"

    def test_json_for_a_non_tool_block_is_ignored(self):
        events, _ = stream(stream_body(block(0, {"type": "text", "text": ""}),
                                       delta(0, type="input_json_delta", partial_json='{"a":1}'), stop(0),
                                       *message_end()))
        assert [e.event_type for e in events] == ["response.completed"]

    def test_async_files(self):
        def respond(request):
            if request.method == "DELETE":
                return httpx2.Response(200, json={"id": "file_1", "type": "file_deleted"})
            return httpx2.Response(200, json={"id": "file_1", "type": "file", "filename": "a.png",
                                              "mime_type": "image/png", "size_bytes": 1,
                                              "created_at": "2026-01-01T00:00:00Z"})
        llm, requests = make_llm(AnthropicLLM, respond, asynchronous=True)

        async def scenario():
            media = await llm.upload_file_async(b"\x89PNG", "image/png")
            await llm.delete_file_async(media)
            return media
        assert run(scenario()).file_id == "file_1"
        assert [r.method for r in requests] == ["POST", "DELETE"]
