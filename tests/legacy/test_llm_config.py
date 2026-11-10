"""One schema, explicit-field merging, independent defaults, and all query modes."""

import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from hiyal.models import LLMConfig, OpenAILLM
from hiyal.models.schemas import GenerationConfig, ModelConfig
from hiyal.tools import OpenAIToolSpec, ToolParameters, Tools
from test_model_config import PROVIDERS, client


PROMPTS = {"system_prompt": "Be helpful.", "user_prompt": "Hello."}


class UnifiedConfigTests(TestCase):
    def test_stream_is_not_a_configuration_field(self):
        self.assertNotIn("stream", LLMConfig.model_fields)
        self.assertNotIn("stream", LLMConfig.model_json_schema()["properties"])
        self.assertNotIn("stream", LLMConfig().model_dump())
        with client(OpenAILLM, model="default") as (llm, _):
            llm._send = Mock()
            with self.assertRaisesRegex(ValueError, "query_stream"):
                llm.query(**PROMPTS, stream=True)
            llm._send.assert_not_called()

    def test_legacy_imports_are_aliases_of_the_single_schema(self):
        self.assertIs(GenerationConfig, LLMConfig)
        self.assertIs(ModelConfig, LLMConfig)

    def test_generation_defaults_are_not_passed_to_client_constructors(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                with client(kind, max_output_tokens=4096, temperature=0.4, **PROMPTS) as (llm, factories):
                    self.assertEqual(llm.config.max_output_tokens, 4096)
                    self.assertEqual(llm.config.temperature, 0.4)
                    called = next(factory for factory in factories.values() if factory.called)
                    for field in ("model", "max_output_tokens", "temperature", "system_prompt", "user_prompt", "others"):
                        self.assertNotIn(field, called.call_args.kwargs)

    def test_defaults_and_overrides_are_independent_for_all_sync_methods(self):
        for kind in PROVIDERS:
            for method, internal in (("query", "_query"), ("query_stream", "_query_stream")):
                with self.subTest(provider=kind.provider, method=method):
                    defaults = LLMConfig(api_key="test-key", model="default", max_output_tokens=4096, temperature=0.4, max_retries=4)
                    with client(kind, config=defaults) as (llm, _):
                        setattr(llm, internal, Mock(side_effect=lambda *, config: config))
                        query = getattr(llm, method)
                        one = query(config=LLMConfig(max_output_tokens=1024), **PROMPTS)
                        two = query(**PROMPTS)
                        three = query(config=LLMConfig(), **PROMPTS)
                        self.assertEqual([one.max_output_tokens, two.max_output_tokens, three.max_output_tokens], [1024, 4096, 4096])
                        for result in (one, two, three):
                            self.assertEqual(result.max_retries, 4)
                            self.assertEqual(result.model, "default")
                            self.assertEqual(result.temperature, 0.4)
                            self.assertEqual(result.system_prompt.text, "Be helpful.")
                            self.assertNotIn("stream", result.model_dump())
                        self.assertEqual(llm.config.max_output_tokens, 4096)
                        self.assertIsNone(defaults.system_prompt)

    def test_keywords_fill_unset_fields_but_config_wins_overlaps(self):
        defaults = LLMConfig(model="default", max_output_tokens=4096)
        with client(OpenAILLM, config=defaults, timeout=30) as (llm, _):
            llm._query = Mock(side_effect=lambda *, config: config)
            override = LLMConfig(max_output_tokens=1024, seed=0)
            result = llm.query(config=override, max_output_tokens=-1, seed=42, temperature=0, **PROMPTS)
            self.assertEqual(result.max_output_tokens, 1024)
            self.assertEqual(result.seed, 0)
            self.assertEqual(result.temperature, 0)
            self.assertEqual(llm.config.timeout, 30)
            self.assertIsNone(llm.config.system_prompt)
            self.assertEqual(override.model_fields_set, {"max_output_tokens", "seed"})

    def test_explicit_none_clears_optional_generation_defaults(self):
        with client(OpenAILLM, model="default", max_output_tokens=4096, temperature=0.4) as (llm, _):
            llm._query = Mock(side_effect=lambda *, config: config)
            result = llm.query(config=LLMConfig(max_output_tokens=None, temperature=None), **PROMPTS)
            self.assertIsNone(result.max_output_tokens)
            self.assertIsNone(result.temperature)
            self.assertNotIn("max_output_tokens", llm._generation_parameters(result))
            self.assertEqual(llm.query(**PROMPTS).max_output_tokens, 4096)

    def test_prompt_defaults_are_optional_at_initialization_but_required_per_call(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                with client(kind, model="default", **PROMPTS) as (llm, _):
                    llm._query = Mock(side_effect=lambda *, config: config)
                    result = llm.query(user_prompt="Another task.")
                    self.assertEqual(result.system_prompt.text, "Be helpful.")
                    self.assertEqual(result.user_prompt.text, "Another task.")
                    self.assertEqual(llm.query().user_prompt.text, "Hello.")
                    with self.assertRaises(ValidationError):
                        llm.query(user_prompt=None)

    def test_mutable_tools_and_extras_do_not_leak_between_calls(self):
        tools = Tools(tool_list=[OpenAIToolSpec(name="lookup", description="Look up.", parameters=ToolParameters())])
        original = LLMConfig(model="default", tools=tools, others={"metadata": {"tags": ["default"]}})
        with client(OpenAILLM, config=original) as (llm, _):
            llm._query = Mock(side_effect=lambda *, config: config)
            one = llm.query(**PROMPTS)
            one.tools.tool_list[0].description = "Changed."
            one.others["metadata"]["tags"].append("local")
            two = llm.query(**PROMPTS)
            self.assertEqual(two.tools.tool_list[0].description, "Look up.")
            self.assertEqual(two.others["metadata"]["tags"], ["default"])
            self.assertIsNone(llm.query(tools=None, **PROMPTS).tools)
            self.assertEqual(llm.query(others={}, **PROMPTS).others, {})
            self.assertEqual(original.tools.tool_list[0].description, "Look up.")

    def test_connection_options_are_fixed_and_never_enter_generation_payload(self):
        with client(OpenAILLM, model="default", client_options={"base_url": "https://example.invalid/v1"}) as (llm, _):
            local = llm._prepare_generation_config(config=None, parameters=PROMPTS)
            params = llm._generation_parameters(local)
            self.assertNotIn("client_options", params)
            self.assertNotIn("base_url", params)
            self.assertNotIn("asynchronous", params)
            for options in ({"asynchronous": True}, {"client_options": {}}):
                with self.assertRaisesRegex(ValueError, "client setting"):
                    llm.query(**PROMPTS, **options)


class AsyncUnifiedConfigTests(IsolatedAsyncioTestCase):
    async def test_all_async_modes_inherit_defaults_and_keep_overrides_local(self):
        for kind in PROVIDERS:
            for method, internal in (("query_async", "_query_async"), ("query_stream_async", "_query_stream_async")):
                with self.subTest(provider=kind.provider, method=method):
                    with client(kind, model="default", max_output_tokens=4096, asynchronous=True) as (llm, _):
                        setattr(llm, internal, AsyncMock(side_effect=lambda *, config: config))
                        query = getattr(llm, method)
                        first, second = await asyncio.gather(
                            query(config=LLMConfig(max_output_tokens=1024), **PROMPTS),
                            query(**PROMPTS),
                        )
                        self.assertEqual(first.max_output_tokens, 1024)
                        self.assertEqual(second.max_output_tokens, 4096)
                        self.assertNotIn("stream", first.model_dump())
                        self.assertEqual(llm.config.max_output_tokens, 4096)
