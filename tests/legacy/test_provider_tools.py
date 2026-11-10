"""Tool request/response parity using real SDK models and mocked transports."""

import unittest
from types import SimpleNamespace as N
from unittest.mock import AsyncMock, Mock

from anthropic.types import Message, ToolUseBlock
from google.genai import types
from ollama import ChatResponse
from pydantic import ValidationError

from hiyal.models.anthropic import AnthropicLLM
from hiyal.models.gemini import GeminiLLM
from hiyal.models.ollama import OllamaLLM
from hiyal.models.schemas import LLMConfig, ToolCallContent, ToolCallDeltaContent
from hiyal.prompts.messages import SystemPrompt, UserPrompt
from hiyal.tools import (
    AnthropicToolSpec, GeminiToolSpec, OllamaToolSpec, OpenAIToolSpec,
    ToolArgument, ToolCallDelta, ToolParameters, ToolSpec, Tools,
)


PROVIDERS = (AnthropicLLM, GeminiLLM, OllamaLLM)


def spec(kind=ToolSpec, **options):
    return kind(
        name="weather", description="Get weather.", parameters=ToolParameters(
            properties=[ToolArgument(name="city", type_="string", description="City.", required=True)],
        ), **options,
    )


def config(**options):
    return LLMConfig(
        model="test-model", system_prompt=SystemPrompt(text="Be helpful."),
        user_prompt=UserPrompt(text="Get weather in Istanbul and Paris."), **options,
    )


def make_llm(kind, *, asynchronous=False):
    llm = kind.__new__(kind)
    if kind is GeminiLLM:
        llm.client = N(models=N(generate_content=Mock(), generate_content_stream=Mock()),
                       aio=N(models=N(generate_content=AsyncMock(), generate_content_stream=AsyncMock())))
    else:
        send = AsyncMock() if asynchronous else Mock()
        llm.client = N(messages=N(create=send)) if kind is AnthropicLLM else N(chat=send)
    return llm


def sender(llm, *, asynchronous=False, stream=False):
    if isinstance(llm, GeminiLLM):
        models = llm.client.aio.models if asynchronous else llm.client.models
        return models.generate_content_stream if stream else models.generate_content
    return llm.client.messages.create if isinstance(llm, AnthropicLLM) else llm.client.chat


def anthropic_response(*, reason="tool_use", cities=("Istanbul", "Paris")):
    return Message.model_validate({
        "id": "msg-test", "model": "test-model", "role": "assistant", "type": "message",
        "stop_reason": reason, "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": [{"type": "text", "text": "Checking."}] + [
            {"type": "tool_use", "id": f"toolu-{i}", "name": "weather", "input": {"city": city}}
            for i, city in enumerate(cities)
        ],
    })


def gemini_chunk(*, reason=None, cities=(), parts=None):
    return types.GenerateContentResponse(
        response_id="gemini-test", model_version="test-model", candidates=[types.Candidate(
            index=0, finish_reason=reason, content=types.Content(role="model", parts=parts if parts is not None else [
                types.Part(function_call=types.FunctionCall(name="weather", args={"city": city}))
                for city in cities
            ]),
        )],
    )


def ollama_chunk(*, done=False, reason=None, cities=()):
    return ChatResponse.model_validate({
        "model": "test-model", "done": done, "done_reason": reason,
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "weather", "arguments": {"city": city}}} for city in cities
        ]},
    })


def response_for(kind, *, cutoff=False):
    if kind is AnthropicLLM:
        return anthropic_response(reason="max_tokens" if cutoff else "tool_use")
    if kind is GeminiLLM:
        return gemini_chunk(reason="MAX_TOKENS" if cutoff else "STOP", cities=("Istanbul", "Paris"))
    return ollama_chunk(done=True, reason="length" if cutoff else "stop", cities=("Istanbul", "Paris"))


def anthropic_events(*, closed=True, arguments='{"city":"Istanbul"}', initial=None):
    block = ToolUseBlock(type="tool_use", id="toolu-1", name="weather", input={} if initial is None else initial)
    events = [
        N(type="message_start", message=anthropic_response(cities=())),
        N(type="content_block_start", index=0, content_block=block),
    ]
    if arguments:
        events.append(N(type="content_block_delta", index=0, delta=N(type="input_json_delta", partial_json=arguments)))
    if closed:
        events.extend([
            N(type="content_block_stop", index=0),
            N(type="message_delta", delta=N(stop_reason="tool_use"), usage=N(model_dump=lambda: {"output_tokens": 20})),
            N(type="message_stop"),
        ])
    return events, block


def chunks_for(kind):
    if kind is AnthropicLLM:
        return anthropic_events()[0]
    if kind is GeminiLLM:
        return [gemini_chunk(cities=("Istanbul",)), gemini_chunk(reason="STOP")]
    return [ollama_chunk(cities=("Istanbul",)), ollama_chunk(done=True, reason="stop")]


class ProviderToolRequestTests(unittest.TestCase):
    def test_declarations_and_caller_config_are_preserved(self):
        for kind, spec_type in zip(PROVIDERS, (AnthropicToolSpec, GeminiToolSpec, OllamaToolSpec)):
            with self.subTest(provider=kind.provider):
                tool = spec(spec_type)
                tools = Tools(tool_list=[tool], parallel_tool_calls=True)
                original = config(tools=tools)
                snapshot = original.model_dump()
                llm = make_llm(kind)
                sender(llm).return_value = response_for(kind)
                llm.query(config=original)
                params = sender(llm).call_args.kwargs
                declarations = params.get("config", params)["tools"]
                self.assertEqual(declarations, [tool.to_sdk_tool()])
                self.assertEqual(original.model_dump(), snapshot)
                local = kind._build_generation_config(config=original, parameters={})
                self.assertIsNot(local.tools, tools)
                self.assertIsNot(local.tools.tool_list[0], tool)
                self.assertIsInstance(local.tools.tool_list[0], spec_type)
                self.assertEqual(kind._tool_parameters(Tools(tool_list=[spec()], parallel_tool_calls=True))["tools"], declarations)

    def test_anthropic_controls_and_provider_options(self):
        for choice, native in [("auto", "auto"), ("required", "any"), ("none", "none")]:
            for parallel in (False, True):
                params = AnthropicLLM._tool_parameters(Tools(tool_list=[spec()], tool_choice=choice, parallel_tool_calls=parallel))
                expected = {"type": native}
                if choice != "none":
                    expected["disable_parallel_tool_use"] = not parallel
                self.assertEqual(params["tool_choice"], expected)
        tool = spec(AnthropicToolSpec, input_examples=[{"city": "Istanbul"}], eager_input_streaming=True)
        self.assertTrue(AnthropicLLM._tool_parameters(Tools(tool_list=[tool]))["tools"][0]["eager_input_streaming"])

    def test_gemini_modes_and_manual_execution(self):
        for choice, mode in [("auto", "AUTO"), ("required", "ANY"), ("none", "NONE")]:
            params = GeminiLLM._generation_parameters(config(tools=Tools(tool_list=[spec()], tool_choice=choice)))
            sdk = types.GenerateContentConfig.model_validate(params["config"])
            self.assertEqual(sdk.tool_config.function_calling_config.mode.value, mode)
            self.assertTrue(sdk.automatic_function_calling.disable)
        with self.assertRaisesRegex(ValueError, "Live-only"):
            GeminiLLM._tool_parameters(Tools(tool_list=[spec(result_behavior="blocking")]))

    def test_unsupported_controls_fail_before_transport(self):
        cases = [
            (GeminiLLM, Tools(tool_list=[spec()], parallel_tool_calls=False)),
            (OllamaLLM, Tools(tool_list=[spec()], parallel_tool_calls=False)),
            (OllamaLLM, Tools(tool_list=[spec()], tool_choice="required")),
        ]
        for kind, tools in cases:
            with self.subTest(provider=kind.provider, tools=tools):
                llm = make_llm(kind)
                with self.assertRaises(ValueError):
                    llm.query(config=config(tools=tools))
                sender(llm).assert_not_called()

    def test_ollama_none_removes_definitions_and_auto_omits_unsupported_flags(self):
        self.assertEqual(OllamaLLM._tool_parameters(Tools(tool_list=[spec()], tool_choice="none")), {"tools": []})
        self.assertEqual(set(OllamaLLM._tool_parameters(Tools(tool_list=[spec()]))), {"tools"})

    def test_foreign_provider_specs_are_not_silently_converted(self):
        for kind in PROVIDERS:
            llm = make_llm(kind)
            with self.assertRaisesRegex(ValueError, "ToolSpec"):
                llm.query(config=config(tools=Tools(tool_list=[spec(OpenAIToolSpec)])))
            sender(llm).assert_not_called()

    def test_tools_in_config_win_over_keywords_and_can_be_removed_in_a_new_config(self):
        for kind in PROVIDERS:
            llm = make_llm(kind)
            sender(llm).return_value = response_for(kind)
            original = config(tools=Tools(tool_list=[spec()]))
            llm.query(config=original, tools=None)
            params = sender(llm).call_args.kwargs
            self.assertIn("tools", params.get("config", params))
            llm.query(config=original.model_copy(update={"tools": None}))
            params = sender(llm).call_args.kwargs
            self.assertNotIn("tools", params.get("config", params))
            self.assertIsNotNone(original.tools)


class ProviderToolResponseTests(unittest.TestCase):
    def test_multiple_same_name_calls_preserve_order_arguments_and_raw(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, raw = make_llm(kind), response_for(kind)
                sender(llm).return_value = raw
                result = llm.query(config=config())
                self.assertIs(result.raw, raw)
                self.assertEqual([c.arguments for c in result.tool_calls], [{"city": "Istanbul"}, {"city": "Paris"}])
                self.assertEqual([c.status for c in result.tool_calls], ["complete", "complete"])
                self.assertEqual(len({m.id for m in result.messages}), len(result.messages))
                self.assertTrue(result.has_tool_calls)
                self.assertTrue(all(isinstance(m.content, ToolCallContent) for m in result.messages if m.content.content_type == "tool_call"))
                first = result.tool_calls[0]
                native = raw.content[1] if kind is AnthropicLLM else raw.candidates[0].content.parts[0] if kind is GeminiLLM else raw.message.tool_calls[0]
                self.assertIs(first.raw, native)
                result.model_dump_json()
                self.assertNotIn("raw", result.model_dump())
                for m in result.model_dump()["messages"]:
                    if m["content"]["content_type"] == "tool_call":
                        self.assertNotIn("raw", m["content"]["data"])
                if kind is AnthropicLLM:
                    self.assertEqual(result.text, "Checking.")
                    self.assertEqual(first.raw.id, "toolu-0")

    def test_token_cutoffs_do_not_report_complete_calls(self):
        for kind in PROVIDERS:
            result = make_llm(kind)._parse_response(response_for(kind, cutoff=True))
            self.assertEqual(result.status, "incomplete")
            self.assertTrue(all(c.status == "incomplete" for c in result.tool_calls))

    def test_invalid_calls_are_local_to_one_call(self):
        llm = make_llm(AnthropicLLM)
        raw = anthropic_response()
        raw.content[1].input = []
        result = llm._parse_response(raw)
        self.assertEqual(result.tool_calls[0].status, "invalid")
        self.assertEqual(result.tool_calls[0].issue.code, "arguments_not_object")
        self.assertEqual(result.tool_calls[1].status, "complete")
        self.assertEqual(result.status, "completed")

    def test_hosted_anthropic_tools_are_not_application_calls(self):
        content = AnthropicLLM._parse_block(N(type="server_tool_use"))
        self.assertEqual(content.content_type, "other")

    def test_gemini_preserves_part_signature_and_ignores_alternative_calls(self):
        raw = gemini_chunk(reason="STOP", parts=[types.Part(
            thought_signature=b"signature", function_call=types.FunctionCall(id="call-1", name="weather", args={}),
        )])
        raw.candidates.append(types.Candidate(index=1, content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="alternative", args={}),
        )])))
        result = make_llm(GeminiLLM)._parse_response(raw)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertEqual(result.tool_calls[0].raw.thought_signature, b"signature")
        self.assertEqual(result.tool_calls[0].raw.function_call.id, "call-1")

    def test_gemini_missing_arguments_do_not_become_an_empty_object(self):
        raw = gemini_chunk(reason="STOP", parts=[
            types.Part(function_call=types.FunctionCall(name="weather")),
            types.Part(function_call=types.FunctionCall(name="weather", args={})),
        ])
        calls = make_llm(GeminiLLM)._parse_response(raw).tool_calls
        self.assertIsNone(calls[0].arguments)
        self.assertEqual(calls[0].status, "invalid")
        self.assertEqual(calls[0].issue.code, "missing_arguments")
        self.assertEqual(calls[1].arguments, {})
        self.assertEqual(calls[1].status, "complete")


class ProviderToolStreamTests(unittest.TestCase):
    def test_multiple_streamed_calls_keep_distinct_message_addresses(self):
        anthropic, _ = anthropic_events()
        # Interleave two JSON argument buffers, including repeated function names.
        anthropic[2:2] = [N(type="content_block_start", index=1, content_block=ToolUseBlock(
            type="tool_use", id="toolu-2", name="weather", input={},
        ))]
        anthropic[4:4] = [
            N(type="content_block_delta", index=1, delta=N(type="input_json_delta", partial_json='{"city":"Paris"}')),
            N(type="content_block_stop", index=1),
        ]
        fixtures = [
            anthropic,
            [gemini_chunk(cities=("Istanbul", "Paris")), gemini_chunk(reason="STOP")],
            [ollama_chunk(cities=("Istanbul",)), ollama_chunk(cities=("Paris",)), ollama_chunk(done=True, reason="stop")],
        ]
        for kind, chunks in zip(PROVIDERS, fixtures):
            with self.subTest(provider=kind.provider):
                llm = make_llm(kind)
                sender(llm, stream=True).return_value = iter(chunks)
                events = list(llm.query_stream(config=config()))
                result = events[-1].response
                self.assertEqual([call.arguments for call in result.tool_calls], [{"city": "Istanbul"}, {"city": "Paris"}])
                self.assertEqual([call.status for call in result.tool_calls], ["complete", "complete"])
                updates = [event for event in events if event.event_type == "content.delta"]
                self.assertEqual({event.message_id for event in updates}, {message.id for message in result.messages})
                for event in updates:
                    self.assertEqual(event.message_id, result.messages[event.message_index].id)

    def test_streams_use_general_events_and_accumulate_typed_calls(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm = make_llm(kind)
                chunks = chunks_for(kind)
                sender(llm, stream=True).return_value = iter(chunks)
                events = list(llm.query_stream(config=config(tools=Tools(tool_list=[spec()]))))
                updates = [e for e in events if e.event_type == "content.delta" and e.delta.content_type == "tool_call"]
                self.assertTrue(updates)
                self.assertTrue(all(isinstance(e.delta, ToolCallDeltaContent) for e in updates))
                result = events[-1].response
                self.assertEqual(events[-1].event_type, "response.completed")
                self.assertEqual(result.tool_calls[0].arguments, {"city": "Istanbul"})
                self.assertEqual(result.tool_calls[0].status, "complete")
                self.assertEqual(updates[0].message_id, result.messages[0].id)
                self.assertIs(result.raw, chunks[-1])
                if kind is not AnthropicLLM:
                    self.assertEqual(updates[0].delta.data.arguments, {"city": "Istanbul"})

    def test_anthropic_json_fragments_leave_original_block_unchanged(self):
        llm = make_llm(AnthropicLLM)
        chunks, block = anthropic_events()
        sender(llm, stream=True).return_value = iter(chunks)
        events = list(llm.query_stream(config=config()))
        call = events[-1].response.tool_calls[0]
        self.assertEqual(block.input, {})
        self.assertIs(call.raw, block)
        self.assertEqual(call.arguments, {"city": "Istanbul"})
        self.assertEqual(events[1].delta.data.arguments_delta, '{"city":"Istanbul"}')

    def test_anthropic_empty_and_initial_arguments_and_malformed_json(self):
        for initial, arguments, status, expected in [
            ({}, "", "complete", {}),
            ({"city": "Paris"}, "", "complete", {"city": "Paris"}),
            ({}, '{"city":', "invalid", None),
            ({"city": "Paris"}, '{"city":"Istanbul"}', "invalid", {"city": "Istanbul"}),
        ]:
            with self.subTest(initial=initial, arguments=arguments):
                llm = make_llm(AnthropicLLM)
                chunks, _ = anthropic_events(initial=initial, arguments=arguments)
                sender(llm, stream=True).return_value = iter(chunks)
                call = list(llm.query_stream(config=config()))[-1].response.tool_calls[0]
                self.assertEqual(call.status, status)
                self.assertEqual(call.arguments, expected)

    def test_interrupted_streams_keep_calls_incomplete(self):
        for kind in PROVIDERS:
            chunks = anthropic_events(closed=False)[0] if kind is AnthropicLLM else chunks_for(kind)[:1]
            llm = make_llm(kind)
            sender(llm, stream=True).return_value = iter(chunks)
            event = list(llm.query_stream(config=config()))[-1]
            self.assertEqual(event.event_type, "response.incomplete")
            self.assertEqual(event.response.tool_calls[0].status, "incomplete")

    def test_stream_token_limits_keep_calls_incomplete(self):
        anthropic = anthropic_events()[0]
        anthropic[-2].delta.stop_reason = "max_tokens"
        fixtures = [
            anthropic,
            [gemini_chunk(cities=("Istanbul",)), gemini_chunk(reason="MAX_TOKENS")],
            [ollama_chunk(cities=("Istanbul",)), ollama_chunk(done=True, reason="length")],
        ]
        for kind, chunks in zip(PROVIDERS, fixtures):
            llm = make_llm(kind)
            sender(llm, stream=True).return_value = iter(chunks)
            result = list(llm.query_stream(config=config()))[-1].response
            self.assertEqual(result.status, "incomplete")
            self.assertEqual(result.tool_calls[0].status, "incomplete")

    def test_gemini_repeated_ids_are_snapshots_not_duplicate_calls(self):
        llm = make_llm(GeminiLLM)
        first = types.Part(function_call=types.FunctionCall(id="a", name="weather", args={"city": "Istanbul"}))
        last = types.Part(function_call=types.FunctionCall(id="a", name="weather", args={"city": "Paris"}))
        sender(llm, stream=True).return_value = iter([
            gemini_chunk(parts=[first]), gemini_chunk(parts=[last], reason="STOP"),
        ])
        events = list(llm.query_stream(config=config()))
        result = events[-1].response
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments, {"city": "Paris"})
        self.assertIs(result.tool_calls[0].raw, first)
        self.assertEqual(events[0].message_id, events[1].message_id)

    def test_gemini_unassembled_incremental_fields_are_not_executable(self):
        llm = make_llm(GeminiLLM)
        parts = [types.Part(function_call=types.FunctionCall(
            id="a", name="weather", args={}, will_continue=True,
        )), types.Part(function_call=types.FunctionCall(
            id="a", name="weather", args={"city": "Istanbul"}, will_continue=False,
        ))]
        sender(llm, stream=True).return_value = iter([
            gemini_chunk(parts=[parts[0]]), gemini_chunk(parts=[parts[1]], reason="STOP"),
        ])
        result = list(llm.query_stream(config=config()))[-1].response
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].status, "incomplete")
        self.assertEqual(result.tool_calls[0].issue.code, "unsupported_partial_arguments")

    def test_snapshot_and_append_updates_are_distinct(self):
        self.assertEqual(ToolCallDelta(arguments={}).arguments, {})
        with self.assertRaises(ValidationError):
            ToolCallDelta(arguments={}, arguments_delta="{}")


class ProviderAsyncToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_queries_and_streams_match_sync_normalization(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm = make_llm(kind, asynchronous=True)
                send = sender(llm, asynchronous=True)
                send.return_value = response_for(kind)
                original = config(tools=Tools(tool_list=[spec()]))
                result = await llm.query_async(config=original)
                self.assertEqual(len(result.tool_calls), 2)
                self.assertTrue(all(call.status == "complete" for call in result.tool_calls))
                async def source():
                    for event in chunks_for(kind):
                        yield event
                sender(llm, asynchronous=True, stream=True).return_value = source()
                stream = await llm.query_stream_async(config=original)
                events = [event async for event in stream]
                self.assertEqual(events[-1].response.tool_calls[0].arguments, {"city": "Istanbul"})
                self.assertEqual(events[-1].response.tool_calls[0].status, "complete")
                self.assertNotIn("stream", original.model_dump())


if __name__ == "__main__":
    unittest.main()
