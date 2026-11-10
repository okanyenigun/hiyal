import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from google.genai import types

from hiyal.models.bases import AsyncLLMStream, LLMStream
from hiyal.models.gemini import GeminiLLM
from hiyal.models.schemas import LLMConfig, LLMResponse
from hiyal.prompts.messages import SystemPrompt, UserPrompt


def candidate(text=None, *, index=0, reason=None, parts=None):
    return types.Candidate(
        index=index, finish_reason=reason,
        content=types.Content(
            role="model", parts=parts if parts is not None else (
                [types.Part(text=text)] if text is not None else []
            ),
        ),
    )


def chunk(*candidates, usage=None, **metadata):
    return types.GenerateContentResponse(
        candidates=list(candidates), usage_metadata=usage, **metadata,
    )


def make_llm():
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=Mock(), generate_content_stream=Mock(),
        ),
        aio=SimpleNamespace(models=SimpleNamespace(
            generate_content=AsyncMock(), generate_content_stream=AsyncMock(),
        )),
    )
    return llm


def config():
    return LLMConfig(
        model="gemini-test", system_prompt=SystemPrompt(text="Be helpful."),
        user_prompt=UserPrompt(text="Who are you?"),
    )


class GeminiResponseTests(unittest.TestCase):
    def setUp(self):
        self.llm = make_llm()

    def test_tool_free_requests_disable_sdk_automatic_function_calling(self):
        original_config = config()
        params = self.llm._generation_parameters(original_config)
        sdk_config = types.GenerateContentConfig.model_validate(params["config"])
        self.assertTrue(sdk_config.automatic_function_calling.disable)
        self.assertEqual(original_config.others, {})

    def test_explicit_afc_settings_and_tool_requests_are_preserved(self):
        for extras in [
            {"automatic_function_calling": {"disable": False}},
            {"config": {"automatic_function_calling": {"disable": False}}},
            {"tools": [{"function_declarations": [{"name": "weather"}]}]},
        ]:
            with self.subTest(extras=extras):
                original_config = config().model_copy(update={"others": extras})
                params = self.llm._generation_parameters(original_config)
                sdk_config = types.GenerateContentConfig.model_validate(params["config"])
                if "tools" in extras:
                    self.assertIsNone(sdk_config.automatic_function_calling)
                    self.assertEqual(sdk_config.tools[0].function_declarations[0].name, "weather")
                else:
                    self.assertFalse(sdk_config.automatic_function_calling.disable)

    def test_response_text_excludes_thoughts_and_uses_first_candidate(self):
        timestamp = datetime(2026, 10, 3, tzinfo=timezone.utc)
        raw = chunk(
            candidate(parts=[
                types.Part(text="Private thought", thought=True),
                types.Part(text="Mer", thought_signature=b"opaque"),
                types.Part(text="haba"),
            ], reason="STOP"),
            candidate("Alternative", index=1, reason="STOP"),
            response_id="gemini-id", model_version="gemini-version",
            create_time=timestamp,
            usage=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=16, candidates_token_count=21,
                thoughts_token_count=175, total_token_count=212,
                cached_content_token_count=4,
            ),
        )
        self.llm.client.models.generate_content.return_value = raw
        response = self.llm.query(config=config())
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text, "Merhaba")
        # Each part is its own message; text fragments merge into one part.
        self.assertEqual(len(response.messages), 2)
        self.assertEqual(response.messages[0].content.content_type, "reasoning")
        self.assertEqual(response.messages[0].content.text, "Private thought")
        self.assertEqual(response.messages[0].role, "other")
        self.assertIsNone(response.messages[0].finish_reason)
        self.assertEqual(response.messages[1].content.content_type, "text")
        self.assertEqual(response.messages[1].content.text, "Merhaba")
        self.assertEqual(response.messages[1].finish_reason, "stop")
        self.assertEqual(response.raw.candidates[1].content.parts[0].text, "Alternative")
        self.assertEqual(response.created_at, timestamp)
        self.assertEqual(response.id_, "gemini-id")
        self.assertEqual(response.model, "gemini-version")
        self.assertEqual(response.usage.total_tokens, 212)
        self.assertEqual(response.usage.cache_read_tokens, 4)
        self.assertIs(response.raw, raw)
        self.assertNotIn("raw", response.model_dump())

    def test_finish_reasons_and_missing_metadata(self):
        for provider_reason, reason, status in [
            ("MAX_TOKENS", "length", "incomplete"),
            ("SAFETY", "content_filter", "completed"),
            ("RECITATION", "recitation", "completed"),
            ("MALFORMED_FUNCTION_CALL", "invalid_output", "completed"),
            (None, "unknown", "completed"),
        ]:
            with self.subTest(reason=provider_reason):
                raw = chunk(candidate(reason=provider_reason))
                response = self.llm._parse_response(raw, model="requested-model")
                self.assertEqual(response.messages[0].finish_reason, reason)
                self.assertEqual(response.status, status)
                self.assertEqual(response.id_, "")
                self.assertEqual(response.model, "requested-model")
                self.assertIsNone(response.usage)

    def test_tool_and_media_content(self):
        parts = [
            types.Part(function_call=types.FunctionCall(
                id="call-1", name="weather", args={"city": "Istanbul"},
            )),
            types.Part(inline_data=types.Blob(mime_type="image/png", data=b"image")),
            types.Part(file_data=types.FileData(mime_type="audio/wav", file_uri="gs://audio")),
        ]
        response = self.llm._parse_response(chunk(candidate(parts=parts, reason="STOP")))
        self.assertEqual(response.messages[0].finish_reason, "tool_call")
        self.assertEqual(response.messages[0].content.data.arguments, {"city": "Istanbul"})
        self.assertEqual(self.llm._parse_part(parts[1]).content_type, "image")
        self.assertEqual(self.llm._parse_part(parts[2]).data["uri"], "gs://audio")

    def test_stream_is_lazy_accumulates_text_and_latest_usage(self):
        chunks = [
            chunk(candidate("Mer"), response_id="stream-id", model_version="version"),
            chunk(candidate("haba", reason="STOP"), usage=types.GenerateContentResponseUsageMetadata(
                candidates_token_count=16, total_token_count=207,
            )),
            chunk(usage=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=16, candidates_token_count=21,
                thoughts_token_count=175, total_token_count=212,
            )),
        ]
        consumed = []

        def raw_stream():
            for value in chunks:
                consumed.append(value)
                yield value

        raw = raw_stream()
        self.llm.client.models.generate_content_stream.return_value = raw
        original_config = config()
        stream = self.llm.query_stream(config=original_config)
        self.assertIsInstance(stream, LLMStream)
        self.assertIs(stream.raw, raw)
        self.assertNotIn("stream", original_config.model_dump())
        self.assertEqual(consumed, [])
        events = list(stream)
        deltas = [event for event in events if event.event_type == "content.delta"]
        self.assertEqual("".join(event.delta.text for event in deltas), "Merhaba")
        self.assertEqual([event.content_index for event in deltas], [0, 0])
        self.assertTrue(all(event.response_id == "stream-id" for event in events))
        self.assertEqual(events[-1].event_type, "response.completed")
        response = events[-1].response
        self.assertEqual(response.text, "Merhaba")
        self.assertEqual(response.messages[0].content.text, "Merhaba")
        self.assertEqual(response.usage.total_tokens, 212)
        self.assertIs(response.raw, chunks[-1])
        self.assertEqual(sum(event.event_type.startswith("response.") for event in events), 1)
        self.assertEqual(list(stream), [])

    def test_alternative_candidate_does_not_change_primary_completion(self):
        chunks = [
            chunk(candidate("First", reason="STOP"), candidate("Alt", index=1)),
            chunk(candidate("ernative", index=1, reason="MAX_TOKENS")),
        ]
        self.llm.client.models.generate_content_stream.return_value = iter(chunks)
        events = list(self.llm.query_stream(config=config()))
        final = events[-1]
        self.assertEqual(final.event_type, "response.completed")
        self.assertEqual(final.response.text, "First")
        self.assertEqual(len(final.response.messages), 1)
        self.assertEqual([event.delta.text for event in events if event.event_type == "content.delta"], ["First"])
        self.assertIs(final.response.raw, chunks[-1])

    def test_stream_accumulates_thoughts_and_audio(self):
        for content_type, parts, expected in [
            ("reasoning", [types.Part(text="Think", thought=True),
                           types.Part(text=" more", thought=True)], "Think more"),
            ("audio", [types.Part(inline_data=types.Blob(mime_type="audio/pcm", data=b"one")),
                       types.Part(inline_data=types.Blob(mime_type="audio/pcm", data=b"two"))], b"onetwo"),
        ]:
            with self.subTest(content_type=content_type):
                self.llm.client.models.generate_content_stream.return_value = iter([
                    chunk(candidate(parts=[parts[0]])),
                    chunk(candidate(parts=[parts[1]], reason="STOP")),
                ])
                events = list(self.llm.query_stream(config=config()))
                response = events[-1].response
                content = response.messages[0].content
                self.assertEqual(content.content_type, content_type)
                self.assertEqual(content.text if content_type == "reasoning" else content.data["data"], expected)
                self.assertEqual(response.text, "")

    def test_signature_blocked_and_unfinished_streams(self):
        streams = [
            ([chunk(candidate("", reason="STOP", parts=[
                types.Part(text="", thought_signature=b"opaque"),
            ]))], "completed"),
            ([types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason="SAFETY",
            ))], "completed"),
            ([chunk(candidate("Partial"))], "incomplete"),
            ([], "incomplete"),
        ]
        for chunks, status in streams:
            with self.subTest(chunks=chunks):
                self.llm.client.models.generate_content_stream.return_value = iter(chunks)
                events = list(self.llm.query_stream(config=config()))
                self.assertEqual(events[-1].response.status, status)
                self.assertFalse(any(
                    event.event_type == "content.delta" and event.delta.text == ""
                    for event in events
                ))

    def test_transport_error_does_not_fabricate_terminal_response(self):
        def raw_stream():
            yield chunk(candidate("Partial"))
            raise RuntimeError("Transport failed")

        self.llm.client.models.generate_content_stream.return_value = raw_stream()
        stream = self.llm.query_stream(config=config())
        self.assertEqual(next(stream).event_type, "content.delta")
        with self.assertRaisesRegex(RuntimeError, "Transport failed"):
            next(stream)
        self.assertEqual(list(stream), [])

    def test_closing_stream_does_not_finalize_partial_response(self):
        self.llm.client.models.generate_content_stream.return_value = iter([
            chunk(candidate("Partial")), chunk(candidate(reason="STOP")),
        ])
        stream = self.llm.query_stream(config=config())
        next(stream)
        stream.close()
        self.assertEqual(list(stream), [])


class GeminiAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_query_and_stream(self):
        llm = make_llm()
        raw = chunk(candidate("Hello", reason="STOP"))
        llm.client.aio.models.generate_content.return_value = raw
        response = await llm.query_async(config=config())
        self.assertEqual(response.text, "Hello")
        self.assertIs(response.raw, raw)

        async def raw_stream():
            yield chunk(candidate("Hel"))
            yield chunk(candidate("lo", reason="STOP"))

        llm.client.aio.models.generate_content_stream.return_value = raw_stream()
        stream = await llm.query_stream_async(config=config())
        self.assertIsInstance(stream, AsyncLLMStream)
        async with stream:
            events = [event async for event in stream]
        self.assertEqual(events[-1].response.text, "Hello")
        self.assertEqual([event async for event in stream], [])

    async def test_async_context_closes_generator_after_early_exit(self):
        llm = make_llm()
        closed = []

        async def raw_stream():
            try:
                yield chunk(candidate("Hello"))
                yield chunk(candidate(reason="STOP"))
            finally:
                closed.append(True)

        llm.client.aio.models.generate_content_stream.return_value = raw_stream()
        async with await llm.query_stream_async(config=config()) as stream:
            self.assertEqual((await anext(stream)).event_type, "content.delta")
        self.assertEqual(closed, [True])
        self.assertEqual([event async for event in stream], [])


if __name__ == "__main__":
    unittest.main()
