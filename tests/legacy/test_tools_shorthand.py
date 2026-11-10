"""tools= accepts a list, with tool_choice / parallel_tool_calls passed alongside."""

import json
from unittest import TestCase

from hiyal.models import GeminiLLM, LLMConfig, OpenAILLM
from hiyal.tools import Tools, tool
from test_connection_overrides import _close_sync, _make_model


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: City name.
    """
    return {}


@tool
def convert_currency(amount: float) -> float:
    """Convert dollars to lira.

    Args:
        amount: Amount in USD.
    """
    return amount


class ToolsShorthandConfigTests(TestCase):
    def test_list_becomes_tools_with_defaults(self):
        config = LLMConfig(tools=[get_weather])
        self.assertIsInstance(config.tools, Tools)
        self.assertEqual([spec.name for spec in config.tools.tool_list], ["get_weather"])
        self.assertEqual((config.tools.tool_choice, config.tools.parallel_tool_calls), ("auto", False))
        self.assertEqual(config.tools.model_fields_set, {"tool_list"})

    def test_settings_fold_into_tools(self):
        config = LLMConfig(tools=[get_weather], tool_choice="required", parallel_tool_calls=True)
        self.assertEqual((config.tools.tool_choice, config.tools.parallel_tool_calls), ("required", True))
        self.assertEqual((config.tool_choice, config.parallel_tool_calls), (None, None))

    def test_settings_override_a_tools_object_and_keep_its_other_fields(self):
        tools = Tools(tool_list=[get_weather], tool_choice="none", parallel_tool_calls=True)
        config = LLMConfig(tools=tools, tool_choice="required")
        self.assertEqual((config.tools.tool_choice, config.tools.parallel_tool_calls), ("required", True))
        self.assertEqual(tools.tool_choice, "none")  # the caller's object is unchanged

    def test_settings_without_tools_wait_for_tools(self):
        config = LLMConfig(tool_choice="required")
        self.assertIsNone(config.tools)
        self.assertEqual(config.tool_choice, "required")

    def test_required_still_needs_a_tool(self):
        with self.assertRaises(ValueError):
            LLMConfig(tools=[], tool_choice="required")


class ToolsShorthandRequestTests(TestCase):
    def request(self, kind=OpenAILLM, client_defaults=None, **call):
        llm, http, requests = _make_model(kind, **(client_defaults or {}))
        try:
            llm.query(user_prompt="Hello.", **call)
            return json.loads(requests[-1].content)
        finally:
            _close_sync(llm, http)

    def test_list_and_settings_per_call(self):
        payload = self.request(tools=[get_weather, convert_currency], tool_choice="required",
                               parallel_tool_calls=True)
        self.assertEqual([tool["name"] for tool in payload["tools"]], ["get_weather", "convert_currency"])
        self.assertEqual((payload["tool_choice"], payload["parallel_tool_calls"]), ("required", True))

    def test_plain_list_uses_default_settings(self):
        payload = self.request(tools=[get_weather])
        self.assertEqual((payload["tool_choice"], payload["parallel_tool_calls"]), ("auto", False))

    def test_per_call_setting_applies_to_client_tools(self):
        payload = self.request(client_defaults={"tools": [get_weather]}, tool_choice="required")
        self.assertEqual([tool["name"] for tool in payload["tools"]], ["get_weather"])
        self.assertEqual(payload["tool_choice"], "required")

    def test_client_settings_do_not_override_a_per_call_tools_object(self):
        payload = self.request(
            client_defaults={"tools": [get_weather], "tool_choice": "required"},
            tools=Tools(tool_list=[convert_currency], tool_choice="none"),
        )
        self.assertEqual([tool["name"] for tool in payload["tools"]], ["convert_currency"])
        self.assertEqual(payload["tool_choice"], "none")

    def test_settings_without_any_tools_raise_before_sending(self):
        llm, http, requests = _make_model(OpenAILLM)
        try:
            with self.assertRaisesRegex(ValueError, "need tools"):
                llm.query(user_prompt="Hello.", tool_choice="required")
            self.assertEqual(requests, [])
        finally:
            _close_sync(llm, http)

    def test_explicit_parallel_false_is_still_explicit_for_gemini(self):
        # Gemini cannot enforce a single call, so an explicit False must still raise.
        llm, http, requests = _make_model(GeminiLLM)
        try:
            with self.assertRaisesRegex(ValueError, "parallel_tool_calls"):
                llm.query(user_prompt="Hello.", tools=[get_weather], parallel_tool_calls=False)
            llm.query(user_prompt="Hello.", tools=[get_weather])  # default: allowed
        finally:
            _close_sync(llm, http)
