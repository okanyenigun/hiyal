"""OpenAILLM through the real OpenAI SDK, over mocked HTTP."""

import httpx2
import openai
import pytest

from hiyal.models import OpenAILLM
from hiyal.prompts import Audio, Image, UserPrompt

from tests.support import (
    auth,
    body,
    chat_chunk,
    chat_completion,
    make_llm,
    openai_function_call,
    openai_message,
    openai_response,
    run,
    sse,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 28
PNG = b"\x89PNG\r\n\x1a\n0000"


def respond(payload=None, *, stream=None, chat=None, chat_stream=None, status=200):
    def handler(request):
        path = request.url.path
        if path.endswith("/files") and request.method == "POST":
            return httpx2.Response(200, json={"id": "file-1", "object": "file", "bytes": 3, "created_at": 1,
                                              "filename": "a.png", "purpose": "user_data"})
        if "/files/" in path and request.method == "DELETE":
            return httpx2.Response(200, json={"id": "file-1", "object": "file", "deleted": True})
        data = body(request)
        if path.endswith("/chat/completions"):
            if data.get("stream"):
                return httpx2.Response(200, text=chat_stream, headers={"content-type": "text/event-stream"})
            return httpx2.Response(status, json=chat or chat_completion())
        if data.get("stream"):
            return httpx2.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx2.Response(status, json=payload or openai_response([openai_message()]))
    return handler


class TestQuery:
    def test_request_and_response(self):
        llm, requests = make_llm(OpenAILLM, respond(), system_prompt="Be brief.")
        result = llm.query(user_prompt="Hi", temperature=0.2)
        [request] = requests
        assert (request.method, request.url.path, auth(request)) == ("POST", "/v1/responses", "test-key")
        assert body(request) == {"model": "gpt-5.5", "temperature": 0.2, "instructions": "Be brief.",
                                 "input": "Hi", "stream": False}
        assert result.text == "Hello." and result.provider == "openai"

    def test_client_options_reach_the_sdk(self):
        llm, requests = make_llm(OpenAILLM, respond(), client_options={
            "base_url": "https://proxy.example/v9", "default_headers": {"x-team": "a"}})
        llm.query(user_prompt="Hi")
        assert str(requests[0].url) == "https://proxy.example/v9/responses"
        assert requests[0].headers["x-team"] == "a"

    def test_http_errors_surface_as_sdk_errors(self):
        llm, _ = make_llm(OpenAILLM, respond({"error": {"message": "bad", "type": "invalid_request_error"}},
                                             status=400))
        with pytest.raises(openai.BadRequestError):
            llm.query(user_prompt="Hi")

    def test_tool_calls_round_trip(self):
        llm, requests = make_llm(OpenAILLM, respond(openai_response([openai_function_call(0)])))
        tool = {"name": "get_weather", "description": "Weather.", "parameters": {"properties": [
            {"name": "city", "type": "string", "description": "City.", "required": True}]}}
        result = llm.query(user_prompt="Weather?", tools=[tool], tool_choice="required")
        assert body(requests[0])["tools"][0]["strict"] is True
        assert result.tool_calls[0].arguments == {"city": "Istanbul"}


class TestConnection:
    def test_per_call_overrides_are_request_local(self):
        llm, requests = make_llm(OpenAILLM, respond(), timeout=30)
        llm.query(user_prompt="Hi", api_key="call-key", timeout=5)
        llm.query(user_prompt="Hi")
        assert auth(requests[0]) == "call-key" and requests[0].extensions["timeout"]["read"] == 5
        assert auth(requests[1]) == "test-key" and requests[1].extensions["timeout"]["read"] == 30
        assert "call-key" not in requests[0].content.decode()

    def test_retries_follow_the_per_call_limit(self):
        attempts = []

        def flaky(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx2.Response(500, json={"error": {"message": "temporary"}}, headers={"retry-after-ms": "1"})
            return httpx2.Response(200, json=openai_response([openai_message()]))

        llm, _ = make_llm(OpenAILLM, flaky)
        assert llm.query(user_prompt="Hi", max_retries=1).text == "Hello."
        assert len(attempts) == 2
        attempts.clear()
        with pytest.raises(openai.InternalServerError):
            llm.query(user_prompt="Hi")
        assert len(attempts) == 1


class TestResponsesStreaming:
    def events(self, *events):
        created = {"type": "response.created", "sequence_number": 0,
                   "response": openai_response([], status="in_progress")}
        return sse([created, *events])

    def test_text_tool_and_terminal_events(self):
        stream = self.events(
            {"type": "response.output_text.delta", "sequence_number": 1, "output_index": 0, "item_id": "msg_1",
             "content_index": 0, "delta": "Hel", "logprobs": []},
            {"type": "response.output_item.added", "sequence_number": 2, "output_index": 1,
             "item": openai_function_call(0, arguments="", status="in_progress")},
            {"type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 1,
             "item_id": "fc_0", "delta": '{"city":"Istanbul"}'},
            {"type": "response.output_text.done", "sequence_number": 4, "output_index": 0, "item_id": "msg_1",
             "content_index": 0, "text": "Hel", "logprobs": []},
            {"type": "response.completed", "sequence_number": 5,
             "response": openai_response([openai_message("Hel"), openai_function_call(0)])},
        )
        llm, requests = make_llm(OpenAILLM, respond(stream=stream))
        events = list(llm.query_stream(user_prompt="Hi"))
        assert body(requests[0])["stream"] is True
        assert [e.event_type for e in events] == ["content.delta"] * 3 + ["response.completed"]
        text, added, arguments, terminal = events
        assert (text.delta.text, text.message_index, text.content_index, text.sequence_number) == ("Hel", 0, 0, 1)
        assert (added.delta.data.name, added.message_id) == ("get_weather", "fc_0")
        assert arguments.delta.data.arguments_delta == '{"city":"Istanbul"}'
        assert all(e.response_id == "resp_1" for e in events)
        assert terminal.response.tool_calls[0].status == "complete"

    def test_refusal_reasoning_and_hosted_tool_fragments(self):
        stream = self.events(
            {"type": "response.refusal.delta", "sequence_number": 1, "output_index": 0, "item_id": "m",
             "content_index": 0, "delta": "No"},
            {"type": "response.reasoning_summary_text.delta", "sequence_number": 2, "output_index": 1,
             "item_id": "r", "summary_index": 3, "delta": "hm"},
            {"type": "response.mcp_call_arguments.delta", "sequence_number": 3, "output_index": 2,
             "item_id": "h", "delta": "{"},
            {"type": "response.completed", "sequence_number": 4, "response": openai_response([])},
        )
        llm, _ = make_llm(OpenAILLM, respond(stream=stream))
        refusal, reasoning, hosted, _ = list(llm.query_stream(user_prompt="Hi"))
        assert (refusal.delta.content_type, reasoning.delta.content_type, reasoning.content_index) == (
            "refusal", "reasoning", 3)
        assert (hosted.delta.content_type, hosted.delta.data) == ("other", {"arguments": "{"})

    def test_incomplete_failed_cancelled_and_error_events(self):
        for event, expected in (
            ({"type": "response.incomplete", "sequence_number": 1,
              "response": openai_response([], status="incomplete", incomplete_reason="max_output_tokens")},
             ("response.incomplete", "incomplete")),
            ({"type": "response.failed", "sequence_number": 1, "response": openai_response(
                [], status="failed", error={"code": "server_error", "message": "boom"})}, ("response.failed", "failed")),
        ):
            llm, _ = make_llm(OpenAILLM, respond(stream=self.events(event)))
            terminal = list(llm.query_stream(user_prompt="Hi"))[-1]
            assert (terminal.event_type, terminal.status) == expected
        llm, _ = make_llm(OpenAILLM, respond(stream=self.events(
            {"type": "error", "sequence_number": 1, "code": "rate_limit", "message": "slow down", "param": None})))
        [error] = list(llm.query_stream(user_prompt="Hi"))
        assert (error.event_type, error.error.code, error.error.message) == ("error", "rate_limit", "slow down")


class TestChatRoute:
    def audio_prompt(self):
        return UserPrompt(text="When?", audio=Audio(mime_type="audio/wav", data=WAV))

    def test_audio_prompt_goes_to_chat_completions(self):
        llm, requests = make_llm(OpenAILLM, respond())
        result = llm.query(user_prompt=self.audio_prompt(), others={"stream_options": {"include_usage": True}})
        assert requests[0].url.path == "/v1/chat/completions"
        assert body(requests[0])["stream"] is False and "stream_options" not in body(requests[0])
        assert result.text == "Hello." and result.timing.source == "calculated"

    def test_streaming_requests_usage(self):
        stream = sse([chat_chunk({"content": "Hel"}), chat_chunk({"content": "lo."}, finish_reason="stop"),
                      chat_chunk(choices=False, usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})],
                     named=False, done=True)
        llm, requests = make_llm(OpenAILLM, respond(chat_stream=stream))
        events = list(llm.query_stream(user_prompt=self.audio_prompt(), others={"stream_options": {"x": 1}}))
        assert body(requests[0])["stream_options"] == {"x": 1, "include_usage": True}
        assert events[-1].response.text == "Hello." and events[-1].response.usage.total_tokens == 3

    def test_explicit_include_usage_false_is_respected(self):
        stream = sse([chat_chunk({"content": "x"}, finish_reason="stop")], named=False, done=True)
        llm, requests = make_llm(OpenAILLM, respond(chat_stream=stream))
        list(llm.query_stream(user_prompt=self.audio_prompt(), others={"stream_options": {"include_usage": False}}))
        assert body(requests[0])["stream_options"] == {"include_usage": False}

    def test_spoken_output(self):
        audio = {"id": "a1", "data": "UklGRg==", "transcript": "hi", "expires_at": 1}
        llm, requests = make_llm(OpenAILLM, respond(chat=chat_completion({"content": None, "audio": audio})))
        result = llm.query(user_prompt="Say hi", response_modalities=["AUDIO"])
        assert body(requests[0])["modalities"] == ["text", "audio"]
        [clip] = result.audio
        assert (clip.data, clip.transcript, clip.mime_type) == (b"RIFF", "hi", "audio/pcm")


class TestFiles:
    def test_upload_and_delete(self):
        llm, requests = make_llm(OpenAILLM, respond())
        media = llm.upload_file(PNG, "image/png", filename="chart.png")
        assert type(media) is Image and (media.file_id, media.filename) == ("file-1", "chart.png")
        upload = requests[0]
        assert (upload.method, upload.url.path) == ("POST", "/v1/files")
        content = upload.content
        assert b'name="purpose"' in content and b"user_data" in content and b'filename="chart.png"' in content
        llm.delete_file(media)
        assert (requests[1].method, requests[1].url.path) == ("DELETE", "/v1/files/file-1")

    def test_uploaded_media_can_be_used_in_a_prompt(self):
        llm, requests = make_llm(OpenAILLM, respond())
        media = llm.upload_file(PNG, "image/png")
        llm.query(user_prompt=UserPrompt(text="Describe", images=media))
        assert body(requests[1])["input"][0]["content"][1] == {"type": "input_image", "detail": "auto",
                                                               "file_id": "file-1"}


class TestAsync:
    def test_query_stream_and_files(self):
        stream = sse([{"type": "response.completed", "sequence_number": 0,
                       "response": openai_response([openai_message("Done.")])}])
        llm, requests = make_llm(OpenAILLM, respond(stream=stream), asynchronous=True)

        async def scenario():
            result = await llm.query_async(user_prompt="Hi", api_key="async-key")
            events = [event async for event in await llm.query_stream_async(user_prompt="Hi")]
            media = await llm.upload_file_async(PNG, "image/png")
            await llm.delete_file_async(media)
            return result, events, media
        result, events, media = run(scenario())
        assert result.text == "Hello." and auth(requests[0]) == "async-key"
        assert events[-1].response.text == "Done." and media.file_id == "file-1"
        assert [r.method for r in requests] == ["POST", "POST", "POST", "DELETE"]

    def test_wrong_mode_fails_before_sending(self):
        llm, requests = make_llm(OpenAILLM, respond(), asynchronous=True)
        with pytest.raises(TypeError, match="use query_async"):
            llm.query(user_prompt="Hi")
        assert requests == []
