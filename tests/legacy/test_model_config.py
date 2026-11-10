"""Authoritative configs, keyword construction, and per-client model defaults."""

import asyncio
import threading
from contextlib import contextmanager, ExitStack
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from hiyal.models import AnthropicLLM, DeepSeekLLM, GeminiLLM, OllamaLLM, OpenAILLM
from hiyal.models.schemas import LLMConfig
from hiyal.prompts.messages import SystemPrompt, UserPrompt
from hiyal.tools import OpenAIToolSpec, ToolParameters, Tools


PROVIDERS = (OpenAILLM, AnthropicLLM, GeminiLLM, OllamaLLM, DeepSeekLLM)


def prompts(**updates):
    return {
        "system_prompt": SystemPrompt(text="Be helpful."),
        "user_prompt": UserPrompt(text="Hello."),
        **updates,
    }


@contextmanager
def client(kind, *, config=None, **parameters):
    with ExitStack() as stack:
        factories = {
            name: stack.enter_context(patch(name))
            for name in (
                "openai.OpenAI",
                "openai.AsyncOpenAI",
                "anthropic.Anthropic",
                "anthropic.AsyncAnthropic",
                "google.genai.Client",
                "ollama.Client",
                "ollama.AsyncClient",
            )
        }
        kwargs = {"api_key": "test-key", **parameters} if config is None else parameters
        yield kind(config=config, **kwargs), factories


class ModelConfigTests(TestCase):
    def test_keyword_fields_build_config_with_defaults_for_all_providers(self):
        for kind in PROVIDERS:
            with (
                self.subTest(provider=kind.provider),
                client(kind, model="default-model", timeout=12) as (llm, factories),
            ):
                self.assertEqual(llm.config.model, "default-model")
                self.assertEqual(llm.config.timeout, 12)
                self.assertEqual(llm.config.max_retries, 2)
                self.assertFalse(llm.config.asynchronous)
                self.assertEqual(llm.config.others, {})
                called = [factory for factory in factories.values() if factory.called]
                self.assertEqual(len(called), 1)
                self.assertNotIn("model", called[0].call_args.kwargs)
                self.assertNotIn("asynchronous", called[0].call_args.kwargs)

    def test_explicit_config_fields_win_over_conflicting_keywords(self):
        original = LLMConfig(
            api_key="config-key", model="config-model", timeout=7,
            max_retries=2, asynchronous=False, client_options={}, others={},
        )
        for kind in PROVIDERS:
            with (
                self.subTest(provider=kind.provider),
                client(
                    kind,
                    config=original,
                    model="ignored",
                    api_key="ignored-key",
                    timeout="invalid",
                    max_retries=-1,
                    asynchronous=True,
                    client_options=None,
                    others=None,
                    unknown_sdk_parameter="ignored",
                ) as (llm, factories),
            ):
                self.assertEqual(llm.config, original)
                self.assertIsNot(llm.config, original)
                self.assertEqual(llm.config.api_key.get_secret_value(), "config-key")
                called = [factory for factory in factories.values() if factory.called]
                self.assertEqual(len(called), 1)
                sent = called[0].call_args.kwargs
                self.assertNotIn("unknown_sdk_parameter", sent)
                self.assertNotIn("model", sent)

    def test_async_constructor_selection_uses_config_not_ignored_keywords(self):
        expected = {
            OpenAILLM: "openai.AsyncOpenAI",
            DeepSeekLLM: "openai.AsyncOpenAI",
            AnthropicLLM: "anthropic.AsyncAnthropic",
            GeminiLLM: "google.genai.Client",
            OllamaLLM: "ollama.AsyncClient",
        }
        for kind in PROVIDERS:
            config = LLMConfig(api_key="test-key", model="default", asynchronous=True)
            with client(kind, config=config, asynchronous=False) as (_, factories):
                factories[expected[kind]].assert_called_once()

    def test_config_and_keyword_extras_are_copied_not_shared(self):
        extras = {"headers": {"X-Test": "original"}}
        original = LLMConfig(client_options=extras)
        for supplied in (original, None):
            local = OpenAILLM._build_config(
                config=supplied, parameters={"client_options": extras}
            )
            local.client_options["headers"]["X-Test"] = "local"
            self.assertEqual(extras["headers"]["X-Test"], "original")
            self.assertEqual(original.client_options["headers"]["X-Test"], "original")

    def test_validation_applies_when_no_config_is_supplied(self):
        for values in ({"timeout": -1}, {"max_retries": -1}):
            with self.assertRaises(ValidationError):
                OpenAILLM._build_config(config=None, parameters=values)
        with self.assertRaises(ValidationError):
            OpenAILLM._build_config(config=None, parameters={"others": None})
        with self.assertRaises(TypeError):
            OpenAILLM._build_config(config=object(), parameters={})

    def test_opaque_sdk_resources_in_extras_are_preserved_not_deep_copied(self):
        resource = threading.RLock()
        extras = {"http_client": resource, "options": {"resources": [resource]}}
        for supplied in (None, LLMConfig(client_options=extras)):
            local = OpenAILLM._build_config(
                config=supplied, parameters={"client_options": extras}
            )
            self.assertIs(local.client_options["http_client"], resource)
            self.assertIs(local.client_options["options"]["resources"][0], resource)
            self.assertIsNot(local.client_options["options"], extras["options"])
            self.assertIsNot(
                local.client_options["options"]["resources"], extras["options"]["resources"]
            )

    def test_model_and_api_key_remain_optional_in_model_config(self):
        config = LLMConfig()
        self.assertIsNone(config.model)
        self.assertIsNone(config.api_key)


class GenerationConfigTests(TestCase):
    def test_string_prompts_are_normalized_in_generation_config(self):
        config = LLMConfig(system_prompt="Be helpful.", user_prompt="Hello.")
        self.assertIsInstance(config.system_prompt, SystemPrompt)
        self.assertIsInstance(config.user_prompt, UserPrompt)
        self.assertEqual(config.system_prompt.text, "Be helpful.")
        self.assertEqual(config.system_prompt.kind, "system")
        self.assertEqual(config.user_prompt.text, "Hello.")
        self.assertEqual(config.user_prompt.kind, "user")

    def test_string_and_object_prompts_work_in_sync_queries_for_every_provider(self):
        for kind in PROVIDERS:
            for method_name, target in (("query", "_query"), ("query_stream", "_query_stream")):
                with self.subTest(provider=kind.provider, method=method_name):
                    with client(kind, model="default") as (llm, _):
                        setattr(llm, target, Mock(side_effect=lambda *, config: config))
                        for parameters in (
                            {"system_prompt": "Be helpful.", "user_prompt": "Hello."},
                            {"system_prompt": "Be helpful.", "user_prompt": UserPrompt(text="Hello.")},
                            {"system_prompt": SystemPrompt(text="Be helpful."), "user_prompt": "Hello."},
                        ):
                            result = getattr(llm, method_name)(**parameters)
                            self.assertIsInstance(result.system_prompt, SystemPrompt)
                            self.assertIsInstance(result.user_prompt, UserPrompt)
                            self.assertEqual(result.system_prompt.text, "Be helpful.")
                            self.assertEqual(result.user_prompt.text, "Hello.")
                            self.assertEqual(result.model, "default")
                            self.assertNotIn("stream", result.model_dump())
                            expected = llm._generation_parameters(
                                LLMConfig(**prompts(model="default"))
                            )
                            self.assertEqual(llm._generation_parameters(result), expected)

    def test_provider_aliases_accept_string_prompts(self):
        config = OpenAILLM._build_generation_config(
            config=None, parameters={"instructions": "Be helpful.", "input": "Hello."}
        )
        self.assertIsInstance(config.system_prompt, SystemPrompt)
        self.assertIsInstance(config.user_prompt, UserPrompt)

    def test_invalid_prompt_types_raise_before_sending(self):
        with client(OpenAILLM, model="default") as (llm, _):
            llm._send = Mock()
            for field in ("system_prompt", "user_prompt"):
                # None is valid for the optional system prompt: it means "no system prompt".
                invalid = (42, [], b"not a string") if field == "system_prompt" else (None, 42, [], b"not a string")
                for value in invalid:
                    with self.subTest(field=field, value=value):
                        parameters = {"system_prompt": "Be helpful.", "user_prompt": "Hello."}
                        parameters[field] = value
                        with self.assertRaises(ValidationError):
                            llm.query(**parameters)
            llm._send.assert_not_called()

    def test_keywords_use_schema_defaults_and_provider_aliases(self):
        local = OpenAILLM._build_generation_config(
            config=None,
            parameters={
                "instructions": SystemPrompt(text="Instructions."),
                "input": UserPrompt(text="Question."),
                "model": "per-call",
                "temperature": 0.2,
                "others": {"metadata": {"tag": "example"}},
            },
        )
        self.assertEqual(local.system_prompt.text, "Instructions.")
        self.assertEqual(local.user_prompt.text, "Question.")
        self.assertEqual(local.model, "per-call")
        self.assertEqual(local.temperature, 0.2)
        self.assertNotIn("stream", local.model_dump())
        self.assertIsNone(local.tools)
        self.assertIsNone(local.max_output_tokens)

    def test_supplied_config_wins_only_for_explicit_fields(self):
        original = LLMConfig(**prompts(model="from-config", temperature=0.4, others={}))
        for kind in PROVIDERS:
            with client(kind, model="client-default") as (llm, _):
                llm._query = Mock(side_effect=lambda *, config: config)
                result = llm.query(
                    config=original,
                    model="ignored",
                    system_prompt=None,
                    user_prompt=None,
                    temperature=-1,
                    others=None,
                )
                self.assertEqual(result.model, original.model)
                self.assertEqual(result.temperature, original.temperature)
                self.assertEqual(result.system_prompt, original.system_prompt)
                self.assertEqual(result.api_key, llm.config.api_key)
                self.assertIsNot(result, original)
                self.assertNotIn("stream", result.model_dump())

    def test_missing_user_prompt_raises_before_sending(self):
        # The system prompt is optional; only the user prompt is required.
        for supplied in (
            {},
            {"system_prompt": SystemPrompt(text="System.")},
        ):
            with client(OpenAILLM, model="default") as (llm, _):
                llm._send = Mock()
                with self.assertRaises(ValidationError):
                    llm.query(**supplied)
                llm._send.assert_not_called()

    def test_model_precedence_and_fallback_for_every_provider(self):
        cases = [
            (None, prompts(model="per-call"), "per-call"),
            (None, prompts(), "client-default"),
            (None, prompts(model=None), "client-default"),
            (
                LLMConfig(**prompts(model="from-config")),
                {"model": "ignored"},
                "from-config",
            ),
            (LLMConfig(**prompts()), {"model": "keyword-model"}, "keyword-model"),
        ]
        for kind in PROVIDERS:
            with client(kind, model="client-default") as (llm, _):
                for original, parameters, expected in cases:
                    with self.subTest(provider=kind.provider, expected=expected):
                        local = llm._prepare_generation_config(
                            config=original, parameters=parameters
                        )
                        self.assertEqual(local.model, expected)
                        params = llm._generation_parameters(local)
                        self.assertEqual(params["model"], expected)
                        self.assertEqual(llm.config.model, "client-default")
                        if original is not None:
                            self.assertIsNot(local, original)

    def test_missing_model_raises_before_sdk_call_when_no_default_exists(self):
        for kind in PROVIDERS:
            with client(kind) as (llm, _):
                llm._send = Mock()
                for original, parameters in (
                    (None, prompts()),
                    (LLMConfig(**prompts()), {}),
                ):
                    with self.assertRaisesRegex(ValueError, "model is required"):
                        llm.query(config=original, **parameters)
                llm._send.assert_not_called()

    def test_model_less_config_is_not_mutated_when_reused_by_different_clients(self):
        original = LLMConfig(**prompts())
        with (
            client(OpenAILLM, model="first") as (first, _),
            client(OpenAILLM, model="second") as (second, _),
        ):
            one = first._prepare_generation_config(config=original, parameters={})
            two = second._prepare_generation_config(config=original, parameters={})
        self.assertEqual((one.model, two.model), ("first", "second"))
        self.assertIsNone(original.model)

    def test_stream_methods_do_not_add_delivery_settings_to_config(self):
        original = LLMConfig(**prompts())
        for kind in PROVIDERS:
            with client(kind, model="default") as (llm, _):
                llm._query_stream = Mock(side_effect=lambda *, config: config)
                result = llm.query_stream(
                    config=original
                )
                self.assertNotIn("stream", result.model_dump())
                self.assertEqual(result.model, "default")
                self.assertNotIn("stream", original.model_dump())
                self.assertIsNone(original.model)
                self.assertNotIn("stream", llm.query_stream(**prompts()).model_dump())

    def test_caller_tools_subclasses_fields_set_and_extras_survive_copying(self):
        spec = OpenAIToolSpec(
            name="lookup", description="Look up.", parameters=ToolParameters()
        )
        tools = Tools(tool_list=[spec])
        metadata = {"nested": ["original"]}
        for original, parameters in (
            (LLMConfig(**prompts(tools=tools, others=metadata)), {}),
            (None, prompts(tools=tools, others=metadata)),
        ):
            local = OpenAILLM._build_generation_config(
                config=original, parameters=parameters
            )
            self.assertIsInstance(local.tools.tool_list[0], OpenAIToolSpec)
            self.assertNotIn("parallel_tool_calls", local.tools.model_fields_set)
            local.tools.tool_list[0].description = "Local."
            local.others["nested"].append("local")
            self.assertEqual(tools.tool_list[0].description, "Look up.")
            self.assertEqual(metadata["nested"], ["original"])
            if original is not None:
                self.assertEqual(original.tools.tool_list[0].description, "Look up.")
                self.assertEqual(original.others["nested"], ["original"])


class AsyncGenerationConfigTests(IsolatedAsyncioTestCase):
    async def test_string_prompts_work_in_async_queries_for_every_provider(self):
        for kind in PROVIDERS:
            with client(kind, model="default", asynchronous=True) as (llm, _):
                for method_name, target in (
                    ("query_async", "_query_async"),
                    ("query_stream_async", "_query_stream_async"),
                ):
                    with self.subTest(provider=kind.provider, method=method_name):
                        setattr(llm, target, AsyncMock(side_effect=lambda *, config: config))
                        result = await getattr(llm, method_name)(
                            system_prompt="Be helpful.", user_prompt="Hello."
                        )
                        self.assertIsInstance(result.system_prompt, SystemPrompt)
                        self.assertIsInstance(result.user_prompt, UserPrompt)
                        self.assertEqual(result.system_prompt.text, "Be helpful.")
                        self.assertEqual(result.user_prompt.text, "Hello.")
                        self.assertEqual(result.model, "default")
                        self.assertNotIn("stream", result.model_dump())

    async def test_async_methods_match_precedence_fallback_and_stream_rules(self):
        original = LLMConfig(**prompts())
        for kind in PROVIDERS:
            with client(kind, model="default", asynchronous=True) as (llm, _):
                llm._query_async = AsyncMock(side_effect=lambda *, config: config)
                llm._query_stream_async = AsyncMock(
                    side_effect=lambda *, config: config
                )
                normal = await llm.query_async(
                    config=original, user_prompt=None
                )
                self.assertEqual(normal.model, "default")
                self.assertEqual(normal.user_prompt.text, "Hello.")
                per_call = await llm.query_async(**prompts(model="per-call"))
                self.assertEqual(per_call.model, "per-call")
                streamed = await llm.query_stream_async(
                    config=original
                )
                self.assertEqual(streamed.model, "default")
                self.assertNotIn("stream", streamed.model_dump())
                self.assertNotIn("stream", original.model_dump())
                self.assertIsNone(original.model)
                self.assertNotIn(
                    "stream", (await llm.query_stream_async(**prompts())).model_dump()
                )

    async def test_concurrent_calls_do_not_modify_shared_config_or_client_model(self):
        with client(OpenAILLM, model="default", asynchronous=True) as (llm, _):

            async def capture(*, config):
                await asyncio.sleep(0)
                return config

            llm._query_async = capture
            shared = LLMConfig(**prompts())
            first, second, third = await asyncio.gather(
                llm.query_async(config=shared),
                llm.query_async(**prompts(model="another")),
                llm.query_async(**prompts()),
            )
            self.assertEqual(
                [first.model, second.model, third.model],
                ["default", "another", "default"],
            )
            self.assertIsNone(shared.model)
            self.assertEqual(llm.config.model, "default")
