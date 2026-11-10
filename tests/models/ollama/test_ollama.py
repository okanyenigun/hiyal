import base64
from datetime import datetime, timezone
from types import SimpleNamespace as N
from unittest.mock import Mock

import httpx
import ollama
import pytest

from hiyal.models import OllamaLLM
from hiyal.models.bases import LLMStream
from hiyal.models.ollama import request as ollama_request
from hiyal.models.ollama.request import ENDPOINT_KEY, request_with_retries, request_with_retries_async
from hiyal.models.ollama.thinking import validate_ollama_think
from hiyal.models.schemas import UnsupportedThinkingError
from hiyal.prompts import Document, Image, UserPrompt
from hiyal.tools import OpenAIToolSpec

from tests.support import auth, body, build_request, make_llm, ndjson, ollama_chat, run

SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()


@pytest.fixture
def llm():
    return OllamaLLM(model="qwen3")


class TestRequest:
    def test_chat_messages_and_options(self, llm):
        assert build_request(llm, system_prompt="Be brief.", temperature=0.2, max_output_tokens=50, top_k=5,
                             stop_sequences=["END"], seed=1, keep_alive="5m", format="json") == {
            "model": "qwen3", "keep_alive": "5m", "format": "json",
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
            "options": {"temperature": 0.2, "num_predict": 50, "top_k": 5, "stop": ["END"], "seed": 1}}

    def test_native_options_merge_and_flat_settings_win(self, llm):
        params = build_request(llm, temperature=0.2, options={"num_ctx": 4096, "temperature": 0.9}, num_gpu=1)
        assert params["options"] == {"num_ctx": 4096, "temperature": 0.2, "num_gpu": 1}

    def test_no_options_key_when_empty(self, llm):
        assert "options" not in build_request(llm)

    def test_non_dict_options_raise(self, llm):
        with pytest.raises(TypeError, match="Ollama 'options' must be a dictionary"):
            build_request(llm, options=[1])

    def test_images_go_on_the_user_message_as_base64(self, llm):
        prompt = UserPrompt(text="Describe", images=[Image(mime_type="image/png", data=b"a"),
                                                     Image(mime_type="image/png", data=b"b")])
        message = build_request(llm, user_prompt=prompt)["messages"][-1]
        assert message == {"role": "user", "content": "Describe", "images": ["YQ==", "Yg=="]}

    def test_images_must_be_inline(self, llm):
        with pytest.raises(ValueError, match="needs images as inline data"):
            build_request(llm, user_prompt=UserPrompt(text="x", images=Image.from_url("https://e.com/a.png")))

    def test_documents_and_audio_are_not_supported(self, llm):
        with pytest.raises(NotImplementedError, match="Ollama does not support user_prompt documents"):
            build_request(llm, user_prompt=UserPrompt(text="x", documents=Document(mime_type="text/plain", data=b"x")))

    def test_audio_output_is_not_supported(self, llm):
        with pytest.raises(ValueError, match="Ollama only supports the TEXT response modality"):
            build_request(llm, response_modalities=["AUDIO"])

    def test_tools(self, llm):
        params = build_request(llm, tools=[SPEC])
        assert params["tools"][0]["function"]["name"] == "get_weather"
        assert build_request(llm, tools=[SPEC], tool_choice="none")["tools"] == []
        assert build_request(llm, tools=[SPEC], parallel_tool_calls=True)["tools"]

    def test_unsupported_tool_controls(self, llm):
        with pytest.raises(ValueError, match="does not support tool_choice='required'"):
            build_request(llm, tools=[SPEC], tool_choice="required")
        with pytest.raises(ValueError, match="cannot enforce parallel_tool_calls=False"):
            build_request(llm, tools=[SPEC], parallel_tool_calls=False)
        with pytest.raises(ValueError, match="Ollama tools must use ToolSpec or OllamaToolSpec"):
            build_request(llm, tools=[OpenAIToolSpec(**SPEC)])


class TestImageGeneration:
    def test_generate_request(self, llm):
        prompt = UserPrompt(text="Draw a cat", images=Image(mime_type="image/png", data=b"a"))
        params = build_request(llm, user_prompt=prompt, system_prompt="sys", response_modalities=["IMAGE"],
                               width=64, height=32, steps=4, temperature=0.5, keep_alive="1m", other="dropped")
        assert params == {ENDPOINT_KEY: "generate", "model": "qwen3", "prompt": "Draw a cat", "system": "sys",
                          "images": ["YQ=="], "width": 64, "height": 32, "steps": 4,
                          "options": {"temperature": 0.5}, "keep_alive": "1m"}

    def test_unsupported_combinations(self, llm):
        with pytest.raises(ValueError, match=r"cannot return \['AUDIO'\]"):
            build_request(llm, response_modalities=["IMAGE", "AUDIO"])
        with pytest.raises(ValueError, match="image generation does not take tools"):
            build_request(llm, response_modalities=["IMAGE"], tools=[SPEC])


class TestThinking:
    @pytest.mark.parametrize("values, think", [
        ({"thinking_mode": "disabled"}, False), ({"thinking_mode": "enabled"}, True),
        ({"thinking_effort": "low"}, "low"), ({"thinking_mode": "enabled", "thinking_effort": "high"}, "high"),
        ({"thinking_mode": "disabled", "thinking_effort": "high"}, False), ({"thinking_mode": "custom"}, "custom"),
    ])
    def test_think_value(self, llm, values, think):
        assert build_request(llm, **values)["think"] == think

    def test_provider_spelling(self, llm):
        assert build_request(llm, think="high")["think"] == "high"
        assert build_request(llm, think="enabled")["think"] is True

    def test_unsupported_controls(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="no nested thinking options"):
            build_request(llm, thinking={"others": {"x": 1}})
        with pytest.raises(UnsupportedThinkingError, match="mode and an effort cannot both"):
            build_request(llm, thinking_mode="custom", thinking_effort="low")
        with pytest.raises(UnsupportedThinkingError, match="thinking_budget is not supported"):
            build_request(llm, thinking_budget=100)

    def test_validate_think_against_server_metadata(self):
        info = {"thinking": {"values": [True, False, "low", "high"], "default": "high"}}
        assert validate_ollama_think("low", info, "m") == "low"
        assert validate_ollama_think(False, info, "m") is False
        level_only = {"thinking": {"values": ["low", "high"], "default": "high"}}
        assert validate_ollama_think(True, level_only, "m") == "high"
        with pytest.raises(UnsupportedThinkingError, match="allowed values: \\[True, False, 'low', 'high'\\]"):
            validate_ollama_think("medium", info, "m")
        with pytest.raises(UnsupportedThinkingError, match="think must be a boolean or a supported level"):
            validate_ollama_think(1, {"thinking": {"values": [True, 1]}}, "m")
        with pytest.raises(UnsupportedThinkingError, match="did not report thinking controls"):
            validate_ollama_think(True, {}, "m")
        with pytest.raises(UnsupportedThinkingError, match="did not report"):
            validate_ollama_think(True, {"thinking": {"values": []}}, "m")
        with pytest.raises(UnsupportedThinkingError, match="did not report"):
            validate_ollama_think(True, {"thinking": "yes"}, "m")
        with pytest.raises(UnsupportedThinkingError, match="not supported"):
            validate_ollama_think(True, {"thinking": {"values": ["low"], "default": "medium"}}, "m")

    def test_booleans_and_strings_are_not_confused(self):
        with pytest.raises(UnsupportedThinkingError):
            validate_ollama_think(True, {"thinking": {"values": [1]}}, "m")


def chat_response(**overrides):
    return ollama.ChatResponse.model_validate(ollama_chat(**overrides))


class TestParsing:
    def test_message_parts_usage_and_timing(self, llm):
        result = llm.translator.parse_response(chat_response(message={"thinking": "hmm", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}))
        assert [m.content.content_type for m in result.messages] == ["reasoning", "text", "tool_call"]
        assert (result.text, result.reasoning, result.status, result.model) == ("Hello.", "hmm", "completed", "qwen3")
        assert result.tool_calls[0].arguments == {"city": "Paris"} and result.tool_calls[0].status == "complete"
        assert result.messages[1].finish_reason == "tool_call"
        assert result.usage.model_dump(exclude_none=True) == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        assert (result.timing.source, result.timing.total_seconds) == ("provider", 1.0)
        assert result.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert result.id_.startswith("ollama_")

    def test_length_is_incomplete_and_calls_incomplete(self, llm):
        result = llm.translator.parse_response(chat_response(done_reason="length", message={"tool_calls": [
            {"function": {"name": "f", "arguments": {}}}]}))
        assert result.status == "incomplete" and result.tool_calls[0].status == "incomplete"
        assert result.messages[0].finish_reason == "length"

    def test_unknown_done_reason_and_missing_metrics(self, llm):
        result = llm.translator.parse_response(chat_response(done_reason="load", eval_count=None,
                                                             total_duration=None))
        assert result.messages[0].finish_reason == "unknown" and result.timing is None
        assert result.usage.total_tokens is None

    def test_generated_images(self, llm):
        for image, mime in ((PNG_B64, "image/png"), (base64.b64encode(b"\xff\xd8\xff").decode(), "image/jpeg"),
                            (base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8").decode(), "image/webp")):
            response = ollama.GenerateResponse.model_validate({
                "model": "img", "created_at": "2026-01-01T00:00:00Z", "response": "", "image": image,
                "done": True, "done_reason": "stop"})
            result = llm.translator.parse_response(response)
            assert result.images[0].mime_type == mime and result.text == ""

    def test_generate_text(self, llm):
        response = ollama.GenerateResponse.model_validate({
            "model": "m", "created_at": "2026-01-01T00:00:00Z", "response": "A cat.", "done": True,
            "done_reason": "stop"})
        assert llm.translator.parse_response(response).text == "A cat."


def stream(llm, chunks):
    accumulator = llm.translator.stream_accumulator(N(model="requested"))
    raw = iter([ollama.ChatResponse.model_validate(chunk) for chunk in chunks])
    return list(LLMStream(raw=raw, event_adapter=accumulator.adapt, finalizer=accumulator.finalize))


class TestStreaming:
    def test_chunks_accumulate_and_the_done_chunk_finishes(self, llm):
        events = stream(llm, [
            ollama_chat({"content": "Hel", "thinking": "h"}, done=False, done_reason=None,
                        created_at="2026-01-01T00:00:00Z", eval_count=None, total_duration=None),
            ollama_chat({"content": "lo."}, done=False, done_reason=None, created_at="2026-01-01T00:00:05Z",
                        eval_count=None, total_duration=None),
            ollama_chat({"content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}),
        ])
        *deltas, terminal = events
        assert [(d.message_index, d.delta.content_type) for d in deltas] == [
            (0, "reasoning"), (1, "text"), (1, "text"), (2, "tool_call")]
        response = terminal.response
        assert (response.text, response.reasoning, response.status) == ("Hello.", "h", "completed")
        assert response.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert response.tool_calls[0].status == "complete" and response.usage.total_tokens == 7
        assert all(e.response_id == response.id_ for e in events)

    def test_stream_without_done_is_incomplete(self, llm):
        events = stream(llm, [ollama_chat({"content": "x"}, done=False, done_reason=None)])
        assert events[-1].status == "incomplete" and events[-1].response.messages[0].finish_reason is None

    def test_empty_stream_has_no_terminal_event(self, llm):
        assert stream(llm, []) == []


def respond(*, think_values=(True, False, "low", "high"), chat=None, stream_chunks=None):
    def handler(request):
        path = request.url.path
        if path == "/api/show":
            return httpx.Response(200, json={"thinking": {"values": list(think_values), "default": "high"}})
        if path == "/api/generate":
            return httpx.Response(200, json={"model": "img", "created_at": "2026-01-01T00:00:00Z", "response": "",
                                             "image": PNG_B64, "done": True, "done_reason": "stop"})
        if body(request).get("stream"):
            return httpx.Response(200, content=ndjson(stream_chunks or [ollama_chat()]),
                                  headers={"content-type": "application/x-ndjson"})
        return httpx.Response(200, json=chat or ollama_chat())
    return handler


class TestEndToEnd:
    def test_chat_request(self):
        llm, requests = make_llm(OllamaLLM, respond(), api_key=None)
        result = llm.query(user_prompt="Hi", temperature=0.2)
        assert [r.url.path for r in requests] == ["/api/chat"]
        sent = body(requests[0])
        assert sent["messages"] == [{"role": "user", "content": "Hi"}] and sent["options"] == {"temperature": 0.2}
        assert sent["stream"] is False and result.text == "Hello."

    def test_api_key_and_headers(self):
        llm, requests = make_llm(OllamaLLM, respond(), client_options={"headers": {"x-team": "a"}})
        llm.query(user_prompt="Hi")
        assert auth(requests[0]) == "test-key" and requests[0].headers["x-team"] == "a"

    def test_think_is_validated_before_the_chat(self):
        llm, requests = make_llm(OllamaLLM, respond(think_values=("low", "high")))
        llm.query(user_prompt="Hi", thinking_mode="enabled")
        assert [r.url.path for r in requests] == ["/api/show", "/api/chat"]
        assert body(requests[0]) == {"model": "qwen3"} and body(requests[1])["think"] == "high"

    def test_unsupported_think_fails_without_chatting(self):
        llm, requests = make_llm(OllamaLLM, respond(think_values=(True, False)))
        with pytest.raises(UnsupportedThinkingError):
            llm.query(user_prompt="Hi", thinking_effort="low")
        assert [r.url.path for r in requests] == ["/api/show"]

    def test_streaming(self):
        chunks = [ollama_chat({"content": "Hel"}, done=False, done_reason=None),
                  ollama_chat({"content": "lo."})]
        llm, requests = make_llm(OllamaLLM, respond(stream_chunks=chunks))
        events = list(llm.query_stream(user_prompt="Hi"))
        assert body(requests[0])["stream"] is True and events[-1].response.text == "Hello."

    def test_image_generation(self):
        llm, requests = make_llm(OllamaLLM, respond())
        result = llm.query(user_prompt="Draw", response_modalities=["IMAGE"], width=64)
        assert requests[0].url.path == "/api/generate"
        assert ENDPOINT_KEY not in body(requests[0]) and body(requests[0])["width"] == 64
        assert result.images[0].mime_type == "image/png"
        with pytest.raises(NotImplementedError, match="image generation cannot be streamed"):
            llm.query_stream(user_prompt="Draw", response_modalities=["IMAGE"])

    def test_per_call_overrides(self):
        llm, requests = make_llm(OllamaLLM, respond(), timeout=30)
        llm.query(user_prompt="Hi", api_key="call-key", timeout=5)
        llm.query(user_prompt="Hi")
        assert [auth(r) for r in requests] == ["call-key", "test-key"]
        assert [r.extensions["timeout"]["read"] for r in requests] == [5, 30]

    def test_retries(self, monkeypatch):
        monkeypatch.setattr(ollama_request, "sleep", lambda seconds: None)
        attempts = []

        def flaky(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json=ollama_chat())
        llm, _ = make_llm(OllamaLLM, flaky)
        assert llm.query(user_prompt="Hi", max_retries=1).text == "Hello."
        assert len(attempts) == 2
        attempts.clear()
        with pytest.raises(ollama.ResponseError):
            llm.query(user_prompt="Hi")
        assert len(attempts) == 1

    def test_async_query_stream_and_think(self):
        llm, requests = make_llm(OllamaLLM, respond(), asynchronous=True)

        async def scenario():
            result = await llm.query_async(user_prompt="Hi", thinking_effort="low")
            events = [e async for e in await llm.query_stream_async(user_prompt="Hi")]
            return result, events
        result, events = run(scenario())
        assert result.text == "Hello." and events[-1].response.text == "Hello."
        assert [r.url.path for r in requests] == ["/api/show", "/api/chat", "/api/chat"]

    def test_no_files_api(self):
        llm = OllamaLLM(model="qwen3")
        with pytest.raises(NotImplementedError, match="Ollama has no Files API"):
            llm.upload_file(b"x", "image/png")


class TestRetryHelpers:
    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        async def no_async_sleep(seconds):
            return None
        monkeypatch.setattr(ollama_request, "sleep", lambda seconds: None)
        monkeypatch.setattr(ollama_request.asyncio, "sleep", no_async_sleep)

    @pytest.mark.parametrize("error", [httpx.ReadTimeout("t"), httpx.ConnectError("c"), ConnectionError("x"),
                                       ollama.ResponseError("busy", 503), ollama.ResponseError("slow", 429),
                                       ollama.ResponseError("conflict", 409), ollama.ResponseError("timeout", 408)])
    def test_retryable_errors_are_retried(self, error):
        send = Mock(side_effect=[error, "ok"])
        assert request_with_retries(send, 1) == "ok" and send.call_count == 2

    @pytest.mark.parametrize("error", [ollama.ResponseError("bad", 400), ollama.ResponseError("missing", 404),
                                       httpx.UnsupportedProtocol("scheme"), ValueError("bug")])
    def test_permanent_errors_are_not_retried(self, error):
        send = Mock(side_effect=error)
        with pytest.raises(type(error)):
            request_with_retries(send, 3)
        assert send.call_count == 1

    def test_gives_up_after_the_limit(self):
        send = Mock(side_effect=httpx.ReadTimeout("t"))
        with pytest.raises(httpx.ReadTimeout):
            request_with_retries(send, 2)
        assert send.call_count == 3

    def test_arguments_are_forwarded(self):
        send = Mock(return_value="ok")
        request_with_retries(send, 0, "POST", "/api/chat", json={"a": 1}, timeout=5)
        send.assert_called_once_with("POST", "/api/chat", stream=False, json={"a": 1}, timeout=5)

    def test_delay_backs_off_and_caps(self):
        assert [ollama_request._delay(attempt) for attempt in range(6)] == [0.25, 0.5, 1.0, 2.0, 2.0, 2.0]

    def test_stream_retries_only_before_output_and_closes_each_attempt(self):
        closed = []

        def attempt(number):
            try:
                if number == 1:
                    raise httpx.ReadTimeout("before output")
                yield "first"
                raise httpx.ReadTimeout("after output")
            finally:
                closed.append(number)

        send = Mock(side_effect=[attempt(1), attempt(2)])
        result = request_with_retries(send, 3, stream=True)
        assert next(result) == "first"
        with pytest.raises(httpx.ReadTimeout, match="after output"):
            next(result)
        assert send.call_count == 2 and closed == [1, 2]

    def test_async_retries(self):
        calls = []

        async def send(*args, stream=False, **kwargs):
            calls.append(stream)
            if len(calls) == 1:
                raise httpx.ReadTimeout("t")
            return "ok"
        assert run(request_with_retries_async(send, 1)) == "ok" and calls == [False, False]

    def test_async_stream_retries_before_output(self):
        attempts = []

        async def send(*args, stream=False, **kwargs):
            attempts.append(stream)

            async def items():
                if len(attempts) == 1:
                    raise httpx.ReadTimeout("t")
                yield "a"
                yield "b"
            return items()

        async def scenario():
            return [item async for item in await request_with_retries_async(send, 2, stream=True)]
        assert run(scenario()) == ["a", "b"] and attempts == [True, True]


class TestRetryEdgeCases:
    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        async def no_async_sleep(seconds):
            return None
        monkeypatch.setattr(ollama_request, "sleep", lambda seconds: None)
        monkeypatch.setattr(ollama_request.asyncio, "sleep", no_async_sleep)

    def test_sync_stream_gives_up_after_the_limit(self):
        def attempt():
            raise httpx.ReadTimeout("t")
            yield  # pragma: no cover

        send = Mock(side_effect=lambda *a, **k: attempt())
        with pytest.raises(httpx.ReadTimeout):
            list(request_with_retries(send, 1, stream=True))
        assert send.call_count == 2

    def test_sync_stream_permanent_errors_are_not_retried(self):
        send = Mock(side_effect=ollama.ResponseError("bad", 400))
        with pytest.raises(ollama.ResponseError):
            list(request_with_retries(send, 3, stream=True))
        assert send.call_count == 1

    def test_async_permanent_errors_and_limits(self):
        calls = []

        async def bad(*args, **kwargs):
            calls.append(1)
            raise ollama.ResponseError("bad", 400)
        with pytest.raises(ollama.ResponseError):
            run(request_with_retries_async(bad, 3))
        assert calls == [1]

        async def busy(*args, **kwargs):
            calls.append(2)
            raise httpx.ReadTimeout("t")
        with pytest.raises(httpx.ReadTimeout):
            run(request_with_retries_async(busy, 2))
        assert calls == [1, 2, 2, 2]

    def test_async_stream_errors_after_output_are_not_retried_and_close_each_attempt(self):
        attempts, closed = [], []

        class Items:
            def __init__(self, number):
                self.number, self.sent = number, False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.sent:
                    self.sent = True
                    return "a"
                raise httpx.ReadTimeout("after output")

            async def aclose(self):
                closed.append(self.number)

        async def send(*args, stream=False, **kwargs):
            attempts.append(1)
            return Items(len(attempts))

        async def scenario():
            items = []
            with pytest.raises(httpx.ReadTimeout):
                async for item in await request_with_retries_async(send, 3, stream=True):
                    items.append(item)
            return items
        assert run(scenario()) == ["a"] and attempts == [1] and closed == [1]

    def test_think_without_a_mode_or_effort_adds_nothing(self):
        assert "think" not in build_request(OllamaLLM(model="qwen3"), thinking={"others": {}})
