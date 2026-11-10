"""The generic entry point constructs the exact existing provider classes."""

import inspect
import json
from typing import get_args
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from hiyal.models import BaseLLM, LLM, LLMConfig
from hiyal.models.llm import _PROVIDERS
from hiyal.models.schemas import LLMResponse, Provider
from test_connection_overrides import PROMPTS, _close_async, _close_sync, _make_model
from test_model_config import PROVIDERS, client


class LLMFactoryTests(TestCase):
    def test_base_remains_abstract_and_all_providers_inherit_it(self):
        self.assertTrue(inspect.isabstract(BaseLLM))
        with self.assertRaises(TypeError):
            BaseLLM()
        for kind in PROVIDERS:
            self.assertTrue(issubclass(kind, BaseLLM))
        self.assertEqual(set(_PROVIDERS), set(get_args(Provider)))

    def test_factory_returns_exact_provider_type_and_initializes_sdk_once(self):
        for kind in PROVIDERS:
            for asynchronous in (False, True):
                with self.subTest(provider=kind.provider, asynchronous=asynchronous):
                    with client(
                        LLM, provider=kind.provider, model="test-model", asynchronous=asynchronous
                    ) as (llm, factories):
                        self.assertIs(type(llm), kind)
                        self.assertIsInstance(llm, BaseLLM)
                        self.assertEqual(llm.provider, kind.provider)
                        self.assertEqual(llm.config.model, "test-model")
                        called = [factory for factory in factories.values() if factory.called]
                        self.assertEqual(len(called), 1)
                        called[0].assert_called_once()
                        self.assertNotIn("provider", called[0].call_args.kwargs)
                        self.assertEqual(llm.config.client_options.get("provider"), None)
                        for name in ("query", "query_async", "query_stream", "query_stream_async"):
                            self.assertIs(getattr(llm, name).__func__, getattr(kind, name))

    def test_config_and_keyword_precedence_are_preserved(self):
        original = LLMConfig(api_key="test-key", model="default", max_output_tokens=1024)
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                with client(
                    LLM, provider=kind.provider, config=original,
                    max_output_tokens=4096, timeout=10,
                ) as (llm, _):
                    self.assertEqual(llm.config.max_output_tokens, 1024)
                    self.assertEqual(llm.config.timeout, 10)
                    self.assertIsNot(llm.config, original)
        self.assertIsNone(original.timeout)

    def test_unknown_provider_is_rejected_without_initializing_sdk(self):
        with patch("openai.OpenAI") as factory:
            for name in ("", "unknown", "OpenAI"):
                with self.subTest(provider=name), self.assertRaisesRegex(ValueError, "Supported providers"):
                    LLM(provider=name)
            for value in (None, 1, [], {}):
                with self.subTest(provider=value), self.assertRaisesRegex(TypeError, "provider"):
                    LLM(provider=value)
            factory.assert_not_called()

    def test_provider_is_required_and_invalid_config_still_fails(self):
        with self.assertRaises(TypeError):
            LLM(model="test-model")
        with self.assertRaisesRegex(TypeError, "LLMConfig"):
            LLM(provider="openai", config={})

    def test_query_and_stream_use_real_provider_paths_with_fake_http(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, constructor=LLM, max_output_tokens=4096)
                direct, direct_http, direct_requests = _make_model(kind, max_output_tokens=4096)
                try:
                    response = llm.query(**PROMPTS)
                    expected = direct.query(**PROMPTS)
                    self.assertIsInstance(response, LLMResponse)
                    self.assertEqual(response.provider, kind.provider)
                    self.assertEqual(response.text, expected.text)
                    self.assertEqual(response.status, expected.status)
                    self.assertEqual(json.loads(requests[0].content), json.loads(direct_requests[0].content))
                    with llm.query_stream(**PROMPTS) as stream:
                        events = list(stream)
                    self.assertEqual(events[-1].event_type, "response.completed")
                    self.assertEqual(events[-1].response.provider, kind.provider)
                    self.assertEqual(len(requests), 2)
                    for request in requests:
                        self.assertNotIn("provider", json.loads(request.content))
                finally:
                    _close_sync(llm, http)
                    _close_sync(direct, direct_http)


class AsyncLLMFactoryTests(IsolatedAsyncioTestCase):
    async def test_async_query_and_stream_use_the_exact_provider_implementation(self):
        for kind in PROVIDERS:
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, constructor=LLM, asynchronous=True)
                try:
                    self.assertIs(type(llm), kind)
                    response = await llm.query_async(**PROMPTS)
                    self.assertIsInstance(response, LLMResponse)
                    self.assertEqual(response.provider, kind.provider)
                    async with await llm.query_stream_async(**PROMPTS) as stream:
                        events = [event async for event in stream]
                    self.assertEqual(events[-1].event_type, "response.completed")
                    self.assertEqual(events[-1].response.provider, kind.provider)
                    self.assertEqual(len(requests), 2)
                finally:
                    await _close_async(llm, http)
