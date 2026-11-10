import unittest
from types import SimpleNamespace as N
from unittest.mock import AsyncMock, Mock

from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import ValidationError

from hiyal.models.openai import OpenAILLM
from hiyal.models.schemas import (
    LLMConfig, ToolCallContent, ToolCallDeltaContent,
)
from hiyal.prompts.messages import SystemPrompt, UserPrompt
from hiyal.tools import (
    OpenAIToolSpec, ToolArgument, ToolCall, ToolCallDelta, ToolParameters,
    ToolSpec, Tools,
)
from hiyal.tools.calls.anthropic import AnthropicToolSpec


def function_call(index=0, *, name="get_weather", arguments='{"city":"Istanbul"}', status="completed"):
    return ResponseFunctionToolCall(
        id=f"fc_{index}", call_id=f"call_{index}", name=name,
        arguments=arguments, status=status, type="function_call",
    )


def sdk_response(output=None, *, status="completed"):
    return Response.model_validate({
        "id": "resp_test", "created_at": 1792499155.0, "model": "gpt-5.5-2026-04-23",
        "object": "response", "output": output or [], "status": status,
        "parallel_tool_calls": True, "tool_choice": "required", "tools": [],
        "usage": {
            "input_tokens": 165, "output_tokens": 81, "total_tokens": 246,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
    })


def tool_spec(spec_type=OpenAIToolSpec, **options):
    return spec_type(
        name="get_weather", description="Get the weather for a city.",
        parameters=ToolParameters(properties=[ToolArgument(
            name="city", type_="string", description="City to look up.", required=True,
        )]), **options,
    )


def config(**updates):
    return LLMConfig(
        model="gpt-5.5", system_prompt=SystemPrompt(text="Be helpful."),
        user_prompt=UserPrompt(text="Get the weather in Istanbul."), **updates,
    )


def make_llm(*, asynchronous=False):
    llm = OpenAILLM.__new__(OpenAILLM)
    create = AsyncMock() if asynchronous else Mock()
    llm.client = N(responses=N(create=create))
    return llm


class OpenAIToolRequestTests(unittest.TestCase):
    def test_query_maps_tools_and_controls_without_mutating_caller_config(self):
        spec = tool_spec(allowed_callers=["direct"], defer_loading=False)
        tools = Tools(tool_list=[spec], tool_choice="required", parallel_tool_calls=True)
        original = config(tools=tools)
        snapshot = tools.model_dump()
        local = OpenAILLM._build_generation_config(config=original, parameters={})
        self.assertIsNot(local, original)
        self.assertIsNot(local.tools, tools)
        self.assertIsNot(local.tools.tool_list[0], spec)
        self.assertIsInstance(local.tools.tool_list[0], OpenAIToolSpec)
        llm = make_llm()
        llm.client.responses.create.return_value = sdk_response()
        llm.query(config=original)
        params = llm.client.responses.create.call_args.kwargs
        self.assertEqual(params["tool_choice"], "required")
        self.assertIs(params["parallel_tool_calls"], True)
        declaration = params["tools"][0]
        self.assertEqual(declaration, spec.to_sdk_tool())
        self.assertEqual(declaration["parameters"]["required"], ["city"])
        self.assertEqual(declaration["allowed_callers"], ["direct"])
        self.assertNotIn("tool_list", declaration)
        self.assertEqual(tools.model_dump(), snapshot)
        self.assertNotIn("stream", original.model_dump())

    def test_common_tools_receive_openai_defaults_without_losing_explicit_values(self):
        for options, expected in [({}, True), ({"strict": None}, None), ({"strict": False}, False)]:
            with self.subTest(options=options):
                spec = tool_spec(ToolSpec, **options)
                original = config(tools=Tools(tool_list=[spec]))
                local = OpenAILLM._build_generation_config(config=original, parameters={})
                params = OpenAILLM._generation_parameters(local)
                self.assertIs(params["tools"][0]["strict"], expected)
                self.assertIs(spec.strict, options.get("strict"))
                self.assertEqual("strict" in spec.model_fields_set, "strict" in options)

    def test_query_keyword_tools_and_config_precedence(self):
        llm = make_llm()
        llm.client.responses.create.return_value = sdk_response()
        tools = Tools(tool_list=[tool_spec()], tool_choice="required")
        base = config()
        llm.query(model=base.model, system_prompt=base.system_prompt,
                  user_prompt=base.user_prompt, tools=tools)
        self.assertEqual(llm.client.responses.create.call_args.kwargs["tools"], [tool_spec().to_sdk_tool()])
        llm.query(config=base, tools=tools)
        self.assertIn("tools", llm.client.responses.create.call_args.kwargs)
        llm.query(config=config(tools=None), tools=tools)
        self.assertNotIn("tools", llm.client.responses.create.call_args.kwargs)
        original = config(tools=tools)
        llm.query(config=original, tools=None)
        self.assertIn("tools", llm.client.responses.create.call_args.kwargs)
        llm.query(config=original.model_copy(update={"tools": None}))
        params = llm.client.responses.create.call_args.kwargs
        self.assertNotIn("tools", params)
        self.assertNotIn("tool_choice", params)
        self.assertNotIn("parallel_tool_calls", params)
        self.assertIs(original.tools, tools)

    def test_foreign_provider_tool_is_rejected_before_sending(self):
        llm = make_llm()
        with self.assertRaisesRegex(ValueError, "OpenAIToolSpec"):
            llm.query(config=config(tools=Tools(tool_list=[tool_spec(AnthropicToolSpec)])))
        llm.client.responses.create.assert_not_called()

    def test_native_tool_parameters_remain_available_through_others(self):
        native = [tool_spec().to_sdk_tool()]
        params = OpenAILLM._generation_parameters(config(others={
            "tools": native, "tool_choice": "required", "parallel_tool_calls": True,
        }))
        self.assertEqual(params["tools"], native)
        self.assertEqual(params["tool_choice"], "required")


class OpenAIToolResponseTests(unittest.TestCase):
    def test_three_calls_like_attachment_are_normalized_in_order(self):
        raw = sdk_response([
            function_call(0),
            function_call(1, name="get_activities", arguments='{"city":"Istanbul","preference":"outdoors"}'),
            function_call(2, name="convert_currency", arguments='{"amount":100,"from_currency":"USD","to_currency":"TRY"}'),
        ])
        llm = make_llm()
        llm.client.responses.create.return_value = raw
        response = llm.query(config=config())
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.text, "")
        self.assertTrue(response.has_tool_calls)
        self.assertEqual([call.name for call in response.tool_calls], [
            "get_weather", "get_activities", "convert_currency",
        ])
        self.assertEqual(response.tool_calls[2].arguments, {
            "amount": 100, "from_currency": "USD", "to_currency": "TRY",
        })
        self.assertEqual(response.usage.total_tokens, 246)
        self.assertIs(response.raw, raw)
        for index, (message, call) in enumerate(zip(response.messages, response.tool_calls)):
            self.assertIsInstance(message.content, ToolCallContent)
            self.assertIsInstance(call, ToolCall)
            self.assertIs(message.content.data, call)
            self.assertEqual(message.id, f"fc_{index}")
            self.assertEqual(call.status, "complete")
            self.assertIsNone(call.issue)
            self.assertIs(call.raw, raw.output[index])
            self.assertEqual(call.raw.call_id, f"call_{index}")
        dumped = response.model_dump()
        self.assertNotIn("raw", dumped)
        self.assertNotIn("tool_calls", dumped)
        self.assertNotIn("has_tool_calls", dumped)
        for message in dumped["messages"]:
            self.assertNotIn("raw", message["content"]["data"])
            self.assertNotIn("call_id", message["content"]["data"])
            self.assertNotIn("id", message["content"]["data"])
        response.model_dump_json()
        response.messages.pop()
        self.assertEqual(len(response.tool_calls), 2)

    def test_text_and_calls_can_coexist(self):
        raw = sdk_response([
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "I will check.", "annotations": [], "logprobs": []}]},
            function_call(0),
        ])
        response = make_llm()._parse_response(raw)
        self.assertEqual(response.text, "I will check.")
        self.assertEqual(len(response.messages), 2)
        self.assertEqual(len(response.tool_calls), 1)
        response.messages.pop()
        self.assertFalse(response.has_tool_calls)

    def test_malformed_arguments_do_not_discard_other_calls(self):
        for arguments, expected_code in [
            ('{"city":', "invalid_json"),
            ("", "missing_arguments"), ("  ", "missing_arguments"),
            ("[]", "arguments_not_object"), ("null", "arguments_not_object"),
            ('{"value": NaN}', "invalid_arguments"),
            ('{"nested": [1e999]}', "invalid_arguments"),
        ]:
            with self.subTest(arguments=arguments):
                response = make_llm()._parse_response(sdk_response([
                    function_call(0, arguments=arguments), function_call(1),
                ]))
                invalid, valid = response.tool_calls
                self.assertEqual(response.status, "completed")
                self.assertEqual(invalid.status, "invalid")
                self.assertEqual(invalid.issue.code, expected_code)
                self.assertIsNone(invalid.arguments)
                self.assertEqual(valid.status, "complete")
                self.assertEqual(invalid.raw.arguments, arguments)

    def test_empty_object_and_missing_names_are_distinguished(self):
        raw = sdk_response([function_call(0, arguments="{}"), function_call(1, name="")])
        calls = make_llm()._parse_response(raw).tool_calls
        self.assertEqual(calls[0].arguments, {})
        self.assertEqual(calls[0].status, "complete")
        self.assertEqual(calls[1].status, "invalid")
        self.assertIsNone(calls[1].name)
        self.assertEqual(calls[1].issue.code, "missing_name")

    def test_partial_calls_remain_incomplete_even_when_json_parses(self):
        for arguments in ('{"city":"Istanbul"}', '{"city":'):
            with self.subTest(arguments=arguments):
                response = make_llm()._parse_response(sdk_response([
                    function_call(0, arguments=arguments, status="incomplete"),
                ], status="incomplete"))
                self.assertEqual(response.status, "incomplete")
                self.assertEqual(response.tool_calls[0].status, "incomplete")

    def test_absent_item_status_uses_generation_completion(self):
        for status, expected in [("completed", "complete"), ("incomplete", "incomplete")]:
            response = make_llm()._parse_response(sdk_response([
                function_call(0, status=None),
            ], status=status))
            self.assertEqual(response.tool_calls[0].status, expected)

    def test_typed_content_rejects_fragment_payloads(self):
        with self.assertRaises(ValidationError):
            ToolCallContent(data=ToolCallDelta(arguments_delta='{"city":'))


def stream_events():
    """Interleave two calls to the same function, then repeat full JSON in done events."""
    return [
        N(type="response.created", sequence_number=0, response=sdk_response(status="in_progress")),
        N(type="response.output_item.added", sequence_number=1, output_index=0,
          item=function_call(0, arguments="", status="in_progress")),
        N(type="response.output_item.added", sequence_number=2, output_index=1,
          item=function_call(1, arguments="", status="in_progress")),
        N(type="response.function_call_arguments.delta", sequence_number=3, output_index=0,
          item_id="fc_0", delta='{"city":'),
        N(type="response.function_call_arguments.delta", sequence_number=4, output_index=1,
          item_id="fc_1", delta='{"city":"Paris"}'),
        N(type="response.function_call_arguments.delta", sequence_number=5, output_index=0,
          item_id="fc_0", delta='"Istanbul"}'),
        N(type="response.function_call_arguments.done", sequence_number=6, output_index=0,
          item_id="fc_0", arguments='{"city":"Istanbul"}'),
        N(type="response.completed", sequence_number=7, response=sdk_response([
            function_call(0), function_call(1, arguments='{"city":"Paris"}'),
        ])),
    ]


class OpenAIToolStreamTests(unittest.TestCase):
    def test_stream_fragments_are_typed_and_terminal_calls_are_complete(self):
        events = stream_events()
        consumed = []

        def native_stream():
            for event in events:
                consumed.append(event)
                yield event

        llm = make_llm()
        llm.client.responses.create.return_value = native_stream()
        original = config(tools=Tools(tool_list=[tool_spec()], parallel_tool_calls=True))
        stream = llm.query_stream(config=original)
        self.assertEqual(consumed, [])
        normalized = list(stream)
        fragments = normalized[:-1]
        self.assertEqual(len(fragments), 5)
        buffers = {}
        for fragment in fragments:
            self.assertIsInstance(fragment.delta, ToolCallDeltaContent)
            self.assertIsInstance(fragment.delta.data, ToolCallDelta)
            self.assertEqual(fragment.content_index, 0)
            self.assertEqual(fragment.response_id, "resp_test")
            buffers.setdefault(fragment.message_id, "")
            buffers[fragment.message_id] += fragment.delta.data.arguments_delta or ""
        self.assertEqual(buffers, {"fc_0": '{"city":"Istanbul"}', "fc_1": '{"city":"Paris"}'})
        final = normalized[-1]
        self.assertEqual(final.event_type, "response.completed")
        self.assertEqual([call.arguments["city"] for call in final.response.tool_calls], ["Istanbul", "Paris"])
        self.assertEqual([message.id for message in final.response.messages], ["fc_0", "fc_1"])
        self.assertIs(final.response.tool_calls[0].raw, events[-1].response.output[0])
        self.assertNotIn("stream", original.model_dump())
        self.assertTrue(llm.client.responses.create.call_args.kwargs["stream"])

    def test_custom_and_mcp_fragments_are_not_application_function_calls(self):
        for event_type in ("response.custom_tool_call_input.delta", "response.mcp_call_arguments.delta"):
            event = N(type=event_type, output_index=0, item_id="hosted_1", delta="fragment")
            normalized = make_llm()._parse_stream_event(event, response_id="resp_test")
            self.assertEqual(normalized.delta.content_type, "other")


class OpenAIAsyncToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_query_and_stream_use_the_same_request_and_response_schemas(self):
        llm = make_llm(asynchronous=True)
        raw = sdk_response([function_call(0)])
        llm.client.responses.create.return_value = raw
        original = config(tools=Tools(tool_list=[tool_spec()], tool_choice="required"))
        response = await llm.query_async(config=original)
        self.assertIsInstance(response.tool_calls[0], ToolCall)
        self.assertEqual(response.tool_calls[0].arguments, {"city": "Istanbul"})
        self.assertEqual(llm.client.responses.create.call_args.kwargs["tool_choice"], "required")

        async def native_stream():
            for event in stream_events():
                yield event

        llm.client.responses.create.return_value = native_stream()
        stream = await llm.query_stream_async(config=original)
        async with stream:
            normalized = [event async for event in stream]
        self.assertIsInstance(normalized[0].delta.data, ToolCallDelta)
        self.assertEqual(len(normalized[-1].response.tool_calls), 2)
        self.assertTrue(all(call.status == "complete" for call in normalized[-1].response.tool_calls))


if __name__ == "__main__":
    unittest.main()
