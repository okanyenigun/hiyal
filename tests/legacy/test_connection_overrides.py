"""Exercise request-local connections through real SDKs and fake HTTP transports."""

import asyncio
import json
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

import httpx
import httpx2
from pydantic import ValidationError

from hiyal.models import AnthropicLLM, DeepSeekLLM, GeminiLLM, OllamaLLM, OpenAILLM
from hiyal.models.schemas import LLMConfig
from hiyal.models._ollama_request import request_with_retries, request_with_retries_async


PROVIDERS = (OpenAILLM, AnthropicLLM, DeepSeekLLM, GeminiLLM, OllamaLLM)
PROMPTS = {"system_prompt": "Be helpful.", "user_prompt": "Hello."}
OVERRIDES = {"api_key": "one-call-key", "timeout": 5, "max_retries": 0}


def _response(kind):
    if kind is OpenAILLM:
        return {
            "id": "resp_test", "object": "response", "created_at": 1792499155,
            "model": "test-model", "status": "completed", "output": [],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        }
    if kind is AnthropicLLM:
        return {
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "test-model", "content": [], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if kind is DeepSeekLLM:
        return {
            "id": "chat_test", "object": "chat.completion", "created": 1792499155,
            "model": "test-model", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "Hello."},
                "finish_reason": "stop",
            }],
        }
    if kind is GeminiLLM:
        return {
            "responseId": "gemini_test", "modelVersion": "test-model",
            "candidates": [{"index": 0, "finishReason": "STOP", "content": {
                "role": "model", "parts": [{"text": "Hello."}],
            }}],
        }
    return {
        "model": "test-model", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": "Hello."},
    }


def _stream_body(kind):
    response = _response(kind)
    if kind is OllamaLLM:
        return json.dumps(response) + "\n"
    if kind is OpenAILLM:
        events = [{"type": "response.completed", "sequence_number": 0, "response": response}]
    elif kind is AnthropicLLM:
        events = [
            {"type": "message_start", "message": {**response, "stop_reason": None}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    elif kind is DeepSeekLLM:
        events = [{**response, "object": "chat.completion.chunk", "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": "Hello."},
            "finish_reason": "stop",
        }]}]
    else:
        events = [response]
    return "".join(
        (f"event: {event['type']}\n" if "type" in event else "")
        + "data: " + json.dumps(event) + "\n\n" for event in events
    ) + ("data: [DONE]\n\n" if kind is DeepSeekLLM else "")


def _make_model(kind, *, asynchronous=False, failures=0, constructor=None, model="test-model",
                thinking_values=(True, False, "low", "medium", "high"), **generation_defaults):
    """No sockets are opened. Keep caller-owned HTTP clients available for assertions."""
    requests = []
    http = httpx2 if kind in (OpenAILLM, AnthropicLLM, DeepSeekLLM) else httpx

    def respond(request):
        requests.append(request)
        if kind is OllamaLLM and request.url.path.endswith("/api/show"):
            return http.Response(200, json={"thinking": {"values": list(thinking_values), "default": "medium"}})
        if len(requests) <= failures:
            return http.Response(500, json={"error": {"message": "temporary", "type": "server_error"}})
        body = json.loads(request.content)
        streaming = body.get("stream") or "streamGenerateContent" in request.url.path
        if streaming:
            return http.Response(200, content=_stream_body(kind), headers={
                "Content-Type": "application/x-ndjson" if kind is OllamaLLM else "text/event-stream",
            })
        return http.Response(200, json=_response(kind))

    transport = http.MockTransport(respond)
    http_client = None if kind is OllamaLLM else (
        http.AsyncClient if asynchronous else http.Client
    )(transport=transport)
    if kind is GeminiLLM:
        options = {"http_options": {
            "httpx_async_client" if asynchronous else "httpx_client": http_client,
            "retry_options": {"initial_delay": 0.001, "max_delay": 0.001},
            "headers": {"x-fixed": "keep"},
        }}
    elif kind is OllamaLLM:
        # Ollama constructs its HTTP client but still uses this in-memory transport.
        options = {"transport": transport, "headers": {"x-fixed": "keep"}}
    else:
        options = {"http_client": http_client, "default_headers": {"x-fixed": "keep"}}
    config = LLMConfig(
        api_key="default-key", model=model, timeout=30,
        max_retries=2, asynchronous=asynchronous, client_options=options,
        **generation_defaults,
    )
    llm = kind(config=config) if constructor is None else constructor(provider=kind.provider, config=config)
    if kind is OllamaLLM:
        http_client = llm.client._client
    return llm, http_client, requests


def _auth(request, kind):
    name = "x-goog-api-key" if kind is GeminiLLM else "x-api-key" if kind is AnthropicLLM else "authorization"
    value = request.headers[name]
    return value.removeprefix("Bearer ")


def _output_limit(request, kind):
    body = json.loads(request.content)
    if kind is GeminiLLM:
        return body["generationConfig"]["maxOutputTokens"]
    if kind is OllamaLLM:
        return body["options"]["num_predict"]
    return body["max_output_tokens" if kind is OpenAILLM else "max_tokens"]


def _close_sync(llm, http):
    if isinstance(llm, OllamaLLM):
        llm.client.__exit__(None, None, None)
    else:
        llm.client.close()
    http.close()


async def _close_async(llm, http):
    if isinstance(llm, GeminiLLM):
        await llm.client.aio.aclose()
        llm.client.close()
    elif isinstance(llm, OllamaLLM):
        await llm.client.__aexit__(None, None, None)
    else:
        await llm.client.close()
    await http.aclose()


def _assert_request(test, request, kind, key, timeout):
    test.assertEqual(_auth(request, kind), key)
    test.assertEqual(request.headers["x-fixed"], "keep")
    test.assertEqual(request.extensions["timeout"]["read"], timeout)
    body = json.loads(request.content)
    test.assertNotIn("api_key", body)
    test.assertNotIn("timeout", body)
    test.assertNotIn("max_retries", body)
    test.assertNotIn("http_options", body)
    test.assertNotIn("one-call-key", request.content.decode())


@contextmanager
def _fast_retries():
    with (
        patch("openai._base_client.BaseClient._calculate_retry_timeout", return_value=0),
        patch("anthropic._base_client.BaseClient._calculate_retry_timeout", return_value=0),
        patch("hiyal.models._ollama_request.sleep"),
        patch("hiyal.models._ollama_request._delay", return_value=0),
    ):
        yield


class ConnectionOverrideTests(TestCase):
    def test_method_controls_streaming_despite_conflicting_provider_extras(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind)
                try:
                    llm.query(**PROMPTS, others={"stream": True})
                    with llm.query_stream(**PROMPTS, others={"stream": False}) as stream:
                        self.assertTrue(list(stream))
                    if kind is GeminiLLM:
                        self.assertNotIn("streamGenerateContent", requests[0].url.path)
                        self.assertIn("streamGenerateContent", requests[1].url.path)
                    else:
                        self.assertIs(json.loads(requests[0].content)["stream"], False)
                        self.assertIs(json.loads(requests[1].content)["stream"], True)
                    if kind is DeepSeekLLM:
                        self.assertNotIn("stream_options", json.loads(requests[0].content))
                        self.assertTrue(json.loads(requests[1].content)["stream_options"]["include_usage"])
                finally:
                    _close_sync(llm, http)

    def test_unified_generation_defaults_reach_every_provider(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, max_output_tokens=4096)
                try:
                    llm.query(config=LLMConfig(max_output_tokens=1024), **PROMPTS)
                    llm.query(**PROMPTS)
                    self.assertEqual([_output_limit(r, kind) for r in requests], [1024, 4096])
                    self.assertEqual(llm.config.max_output_tokens, 4096)
                finally:
                    _close_sync(llm, http)

    def test_streams_keep_overrides_when_consumed_after_a_default_call(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind)
                try:
                    with llm.query_stream(**PROMPTS, **OVERRIDES) as stream:
                        llm.query(**PROMPTS)
                        next(stream)
                    llm.query(**PROMPTS)
                    pairs = [(_auth(r, kind), r.extensions["timeout"]["read"]) for r in requests]
                    self.assertCountEqual(pairs, [
                        ("one-call-key", 5), ("default-key", 30), ("default-key", 30),
                    ])
                    self.assertFalse(http.is_closed)
                finally:
                    _close_sync(llm, http)

    def test_stream_request_retries_use_the_selected_limit(self):
        for kind in PROVIDERS:
            for retries, failures, expected in ((0, 1, 1), (1, 1, 2), (None, 2, 3)):
                with self.subTest(provider=kind.provider, retries=retries):
                    llm, http, requests = _make_model(kind, failures=failures)
                    try:
                        def consume():
                            with llm.query_stream(**PROMPTS, max_retries=retries) as stream:
                                return list(stream)

                        with _fast_retries():
                            if retries == 0:
                                with self.assertRaises(Exception):
                                    consume()
                            else:
                                self.assertTrue(consume())
                        self.assertEqual(len(requests), expected)
                    finally:
                        _close_sync(llm, http)

    def test_individual_overrides_inherit_other_defaults_and_none_inherits_all(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind)
                try:
                    for options, key, timeout in (
                        ({"api_key": "one-call-key"}, "one-call-key", 30),
                        ({"timeout": 5}, "default-key", 5),
                        ({"max_retries": 0}, "default-key", 30),
                        ({"api_key": None, "timeout": None, "max_retries": None}, "default-key", 30),
                    ):
                        llm.query(**PROMPTS, **options)
                        _assert_request(self, requests[-1], kind, key, timeout)
                finally:
                    _close_sync(llm, http)

    def test_defaults_survive_failed_override(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, failures=2)
                try:
                    with _fast_retries():
                        with self.assertRaises(Exception):
                            llm.query(**PROMPTS, **OVERRIDES)
                        llm.query(**PROMPTS)
                    self.assertEqual(len(requests), 3)
                    _assert_request(self, requests[0], kind, "one-call-key", 5)
                    for request in requests[1:]:
                        _assert_request(self, request, kind, "default-key", 30)
                    self.assertEqual(llm.config.max_retries, 2)
                finally:
                    _close_sync(llm, http)

    def test_overrides_are_request_local_for_every_provider_and_sync_mode(self):
        for kind in PROVIDERS:
            for streaming in (False, True):
                with self.subTest(provider=kind.provider, streaming=streaming):
                    llm, http, requests = _make_model(kind)
                    stored_client = llm.client
                    original = llm.config.model_dump()
                    query = llm.query_stream if streaming else llm.query
                    try:
                        if streaming:
                            with query(**PROMPTS, **OVERRIDES) as stream:
                                list(stream)
                            with query(**PROMPTS) as stream:
                                list(stream)
                        else:
                            query(**PROMPTS, **OVERRIDES)
                            query(**PROMPTS)
                        self.assertEqual(len(requests), 2)
                        _assert_request(self, requests[0], kind, "one-call-key", 5)
                        _assert_request(self, requests[1], kind, "default-key", 30)
                        self.assertIs(llm.client, stored_client)
                        self.assertEqual(llm.config.model_dump(), original)
                        self.assertFalse(http.is_closed)
                    finally:
                        _close_sync(llm, http)

    def test_generation_config_wins_and_is_not_modified(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind)
                config = LLMConfig(**PROMPTS, **OVERRIDES)
                snapshot = config.model_dump()
                try:
                    llm.query(config=config, api_key="ignored", timeout=-1, max_retries=-1)
                    _assert_request(self, requests[0], kind, "one-call-key", 5)
                    self.assertEqual(config.model_dump(), snapshot)
                    self.assertNotIn("one-call-key", repr(config))
                    llm.query(config=LLMConfig(**PROMPTS))
                    _assert_request(self, requests[1], kind, "default-key", 30)
                finally:
                    _close_sync(llm, http)

    def test_zero_retries_and_then_default_retries(self):
        for kind in PROVIDERS:
            for retries, failures, expected in ((0, 1, 1), (1, 1, 2), (None, 2, 3)):
                with self.subTest(provider=kind.provider, retries=retries):
                    llm, http, requests = _make_model(kind, failures=failures)
                    try:
                        with _fast_retries():
                            if retries == 0:
                                with self.assertRaises(Exception):
                                    llm.query(**PROMPTS, max_retries=0)
                            else:
                                llm.query(**PROMPTS, max_retries=retries)
                        self.assertEqual(len(requests), expected)
                        self.assertEqual(llm.config.max_retries, 2)
                    finally:
                        _close_sync(llm, http)

    def test_invalid_overrides_and_execution_mode_fail_before_requests(self):
        llm, http, requests = _make_model(OpenAILLM)
        try:
            for options in ({"timeout": -1}, {"timeout": float("nan")}, {"max_retries": -1}, {"api_key": 123}):
                with self.subTest(options=options), self.assertRaises(ValidationError):
                    llm.query(**PROMPTS, **options)
            with self.assertRaisesRegex(ValueError, "client setting"):
                llm.query(**PROMPTS, asynchronous=True)
            self.assertEqual(requests, [])
        finally:
            llm.client.close()
            http.close()


class AsyncConnectionOverrideTests(IsolatedAsyncioTestCase):
    async def test_async_generation_defaults_reach_every_provider(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, asynchronous=True, max_output_tokens=4096)
                try:
                    await llm.query_async(config=LLMConfig(max_output_tokens=1024), **PROMPTS)
                    await llm.query_async(**PROMPTS)
                    self.assertEqual([_output_limit(r, kind) for r in requests], [1024, 4096])
                    self.assertEqual(llm.config.max_output_tokens, 4096)
                finally:
                    await _close_async(llm, http)

    async def test_async_stream_request_retries_use_the_selected_limit(self):
        for kind in PROVIDERS:
            for retries, failures, expected in ((0, 1, 1), (1, 1, 2), (None, 2, 3)):
                with self.subTest(provider=kind.provider, retries=retries):
                    llm, http, requests = _make_model(kind, asynchronous=True, failures=failures)
                    try:
                        async def consume():
                            async with await llm.query_stream_async(**PROMPTS, max_retries=retries) as stream:
                                return [event async for event in stream]

                        with _fast_retries():
                            if retries == 0:
                                with self.assertRaises(Exception):
                                    await consume()
                            else:
                                self.assertTrue(await consume())
                        self.assertEqual(len(requests), expected)
                    finally:
                        await _close_async(llm, http)

    async def test_async_retry_counts_for_each_provider(self):
        for kind in PROVIDERS:
            for retries, failures, expected in ((0, 1, 1), (1, 1, 2), (None, 2, 3)):
                with self.subTest(provider=kind.provider, retries=retries):
                    llm, http, requests = _make_model(kind, asynchronous=True, failures=failures)
                    try:
                        with _fast_retries():
                            if retries == 0:
                                with self.assertRaises(Exception):
                                    await llm.query_async(**PROMPTS, max_retries=0)
                            else:
                                await llm.query_async(**PROMPTS, max_retries=retries)
                        self.assertEqual(len(requests), expected)
                        self.assertEqual(llm.config.max_retries, 2)
                    finally:
                        await _close_async(llm, http)

    async def test_overrides_are_request_local_for_every_async_mode(self):
        for kind in PROVIDERS:
            for streaming in (False, True):
                with self.subTest(provider=kind.provider, streaming=streaming):
                    llm, http, requests = _make_model(kind, asynchronous=True)
                    stored_client = llm.client
                    snapshot = llm.config.model_dump()
                    try:
                        if streaming:
                            async with await llm.query_stream_async(**PROMPTS, **OVERRIDES) as stream:
                                events = [event async for event in stream]
                            self.assertTrue(events)
                            async with await llm.query_stream_async(**PROMPTS) as stream:
                                [event async for event in stream]
                        else:
                            await llm.query_async(**PROMPTS, **OVERRIDES)
                            await llm.query_async(**PROMPTS)
                        self.assertEqual(len(requests), 2)
                        _assert_request(self, requests[0], kind, "one-call-key", 5)
                        _assert_request(self, requests[1], kind, "default-key", 30)
                        self.assertIs(llm.client, stored_client)
                        self.assertEqual(llm.config.model_dump(), snapshot)
                        self.assertFalse(http.is_closed)
                    finally:
                        await _close_async(llm, http)

    async def test_concurrent_credentials_and_timeouts_do_not_leak(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, asynchronous=True)
                try:
                    await asyncio.gather(
                        llm.query_async(**PROMPTS, api_key="key-a", timeout=5),
                        llm.query_async(**PROMPTS, api_key="key-b", timeout=10),
                        llm.query_async(**PROMPTS),
                    )
                    pairs = {(_auth(r, kind), r.extensions["timeout"]["read"]) for r in requests}
                    self.assertEqual(pairs, {("key-a", 5), ("key-b", 10), ("default-key", 30)})
                    self.assertEqual(llm.config.api_key.get_secret_value(), "default-key")
                    self.assertEqual(llm.config.timeout, 30)
                finally:
                    await _close_async(llm, http)


class OllamaRetryTests(TestCase):
    def test_stream_retries_only_before_output_and_closes_each_attempt(self):
        closed = []

        def stream(attempt):
            try:
                if attempt == 1:
                    raise httpx.ReadTimeout("temporary")
                yield "first"
                raise httpx.ReadTimeout("after output")
            finally:
                closed.append(attempt)

        send = Mock(side_effect=[stream(1), stream(2)])
        with patch("hiyal.models._ollama_request.sleep"):
            result = request_with_retries(send, 3, stream=True)
            self.assertEqual(next(result), "first")
            with self.assertRaises(httpx.ReadTimeout):
                next(result)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(closed, [1, 2])

    def test_permanent_errors_are_not_retried(self):
        from ollama import ResponseError

        send = Mock(side_effect=ResponseError("invalid request", 400))
        with self.assertRaises(ResponseError):
            request_with_retries(send, 3)
        self.assertEqual(send.call_count, 1)

        send = Mock(side_effect=httpx.UnsupportedProtocol("invalid URL scheme"))
        with self.assertRaises(httpx.UnsupportedProtocol):
            request_with_retries(send, 3)
        self.assertEqual(send.call_count, 1)


class AsyncOllamaRetryTests(IsolatedAsyncioTestCase):
    async def test_cancellation_is_not_retried(self):
        send = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await request_with_retries_async(send, 3)
        self.assertEqual(send.call_count, 1)

    async def test_async_retries_are_nonblocking_and_never_replay_partial_output(self):
        closed = []

        async def stream(attempt):
            try:
                if attempt == 1:
                    raise httpx.ReadTimeout("temporary")
                yield "first"
                raise httpx.ReadTimeout("after output")
            finally:
                closed.append(attempt)

        send = AsyncMock(side_effect=[stream(1), stream(2)])
        with patch("hiyal.models._ollama_request.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await request_with_retries_async(send, 3, stream=True)
            self.assertEqual(await anext(result), "first")
            with self.assertRaises(httpx.ReadTimeout):
                await anext(result)
            sleep.assert_awaited_once()
        self.assertEqual(send.call_count, 2)
        self.assertEqual(closed, [1, 2])
