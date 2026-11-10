"""Common thinking controls map to native parameters without silent downgrades."""

import json
from copy import deepcopy
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock

from pydantic import ValidationError

from hiyal.models import (
    AnthropicLLM, DeepSeekLLM, GeminiLLM, LLM, LLMConfig, OllamaLLM, OpenAILLM, ThinkingConfig,
)
from hiyal.models._thinking import adapt_thinking, validate_ollama_think
from test_connection_overrides import PROMPTS, _close_async, _close_sync, _make_model
from test_model_config import client


MODELS = {
    OpenAILLM: "gpt-5.5", AnthropicLLM: "claude-sonnet-4-6",
    GeminiLLM: "gemini-3.8-flash", DeepSeekLLM: "deepseek-flash", OllamaLLM: "test-thinking",
}


def body(requests):
    return json.loads(requests[-1].content)


def check_effort(test, kind, payload, effort):
    if kind is OpenAILLM:
        test.assertEqual(payload["reasoning"]["effort"], effort)
    elif kind is AnthropicLLM:
        test.assertEqual(payload["output_config"]["effort"], effort)
        test.assertEqual(payload["thinking"]["type"], "adaptive")
    elif kind is GeminiLLM:
        test.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinking_level"].lower(), effort)
    elif kind is DeepSeekLLM:
        test.assertEqual(payload["reasoning_effort"], "high" if effort == "medium" else effort)
        test.assertEqual(payload["thinking"]["type"], "enabled")
    else:
        test.assertEqual(payload["think"], effort)


class ThinkingSchemaTests(TestCase):
    def test_structured_values_and_provider_specific_extensions(self):
        cfg = LLMConfig(thinking={"effort": "medium", "mode": "enabled"})
        self.assertEqual(cfg.thinking_effort, "medium")
        self.assertEqual(cfg.thinking_mode, "enabled")
        self.assertIsNone(cfg.thinking)
        self.assertEqual(cfg.model_fields_set, {"thinking_effort", "thinking_mode"})
        native = ThinkingConfig(effort="xhigh", mode="adaptive", others={"budget_tokens": 2048})
        self.assertEqual(native.effort, "xhigh")
        self.assertEqual(native.others["budget_tokens"], 2048)

    def test_old_level_and_switch_shorthands_are_preserved(self):
        self.assertEqual(LLMConfig(thinking="low").thinking_effort, "low")
        self.assertEqual(LLMConfig(thinking="enabled").thinking_mode, "enabled")
        self.assertEqual(LLMConfig(thinking="disabled").thinking_mode, "disabled")

    def test_invalid_shapes_and_typos_are_rejected(self):
        for value in (True, 100, {"effort": ""}, {"effort": "high effort"}, {"mode": False}, {"mod": "enabled"}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                LLMConfig(thinking=value)

    def test_partial_thinking_overrides_inherit_without_mutating_defaults(self):
        original = ThinkingConfig(effort="low", mode="enabled", others={"summary": "auto"})
        with client(OpenAILLM, model="gpt-5.5", thinking=original) as (llm, _):
            llm._query = Mock(side_effect=lambda *, config: config)
            result = llm.query(**PROMPTS, thinking={"effort": "high"})
            self.assertEqual(result.thinking_effort, "high")
            self.assertEqual(result.thinking_mode, "enabled")
            self.assertEqual(result.thinking.others, {"summary": "auto"})
            result.thinking.others["summary"] = "detailed"
            following = llm.query(**PROMPTS)
            self.assertEqual(following.thinking_effort, "low")
            self.assertEqual(following.thinking_mode, "enabled")
            self.assertEqual(following.thinking.others, original.others)
            cleared = llm.query(**PROMPTS, thinking=None)
            for name in ("thinking", "thinking_mode", "thinking_effort", "thinking_budget"):
                self.assertIsNone(getattr(cleared, name))

    def test_dictionary_controls_are_flattened_and_direct_fields_win(self):
        dictionary = {"mode": "enabled", "effort": "low", "budget": 2048, "others": {"summary": "auto"}}
        snapshot = deepcopy(dictionary)
        cfg = LLMConfig(thinking=dictionary, thinking_effort="high", thinking_budget=None)
        self.assertEqual(cfg.thinking_mode, "enabled")
        self.assertEqual(cfg.thinking_effort, "high")
        self.assertIsNone(cfg.thinking_budget)
        self.assertEqual(cfg.thinking.others, {"summary": "auto"})
        self.assertIsNone(cfg.thinking.mode)
        self.assertIsNone(cfg.thinking.effort)
        self.assertEqual(dictionary, snapshot)
        self.assertEqual(LLMConfig.model_validate(cfg.model_dump()), cfg)
        self.assertEqual(LLMConfig(thinking={"budget": 2048}).thinking_budget, 2048)

    def test_flat_fields_are_validated_and_exposed_in_json_schema(self):
        for field, invalid in (("thinking_mode", False), ("thinking_effort", ""),
                               ("thinking_effort", "high effort"), ("thinking_budget", -1),
                               ("thinking_budget", True), ("thinking_budget", 1.5),
                               ("thinking_budget", "2048")):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValidationError):
                LLMConfig(**{field: invalid})
        properties = LLMConfig.model_json_schema()["properties"]
        for name in ("thinking_mode", "thinking_effort", "thinking_budget", "thinking"):
            self.assertIn(name, properties)

    def test_config_precedence_is_independent_of_thinking_input_form(self):
        for config, keywords in (
            (LLMConfig(thinking={"effort": "low"}), {"thinking_effort": "high", "thinking_mode": "enabled"}),
            (LLMConfig(thinking_effort="low"), {"thinking": {"effort": "high", "mode": "enabled"}}),
            (LLMConfig(thinking_effort="low"), {"thinking": {"effort": "", "mode": "enabled"}}),
        ):
            with self.subTest(config=config), client(OpenAILLM, model="gpt-5.5") as (llm, _):
                llm._query = Mock(side_effect=lambda *, config: config)
                result = llm.query(config=config, **keywords, **PROMPTS)
                self.assertEqual(result.thinking_effort, "low")
                self.assertEqual(result.thinking_mode, "enabled")

    def test_budget_defaults_can_be_cleared_when_switching_to_adaptive_mode(self):
        with client(AnthropicLLM, model="claude-sonnet-4-6", max_output_tokens=8192,
                    thinking_mode="enabled", thinking_budget=2048) as (llm, _):
            local = llm._prepare_generation_config(
                config=LLMConfig(thinking_mode="adaptive", thinking_budget=None),
                parameters={**PROMPTS, "thinking_effort": "medium"},
            )
            self.assertEqual(llm._generation_parameters(local)["thinking"], {"type": "adaptive"})
            self.assertEqual(llm.config.thinking_budget, 2048)

    def test_per_call_dictionary_overrides_flat_defaults_and_the_reverse(self):
        for defaults, override in (
            ({"thinking_effort": "low", "thinking_mode": "enabled"}, {"thinking": {"effort": "high"}}),
            ({"thinking": {"effort": "low", "mode": "enabled"}}, {"thinking_effort": "high"}),
        ):
            with self.subTest(defaults=defaults), client(OpenAILLM, model="gpt-5.5", **defaults) as (llm, _):
                llm._query = Mock(side_effect=lambda *, config: config)
                self.assertEqual(llm.query(**PROMPTS, **override).thinking_effort, "high")
                self.assertEqual(llm.query(**PROMPTS).thinking_effort, "low")
                cleared = llm.query(**PROMPTS, thinking_effort=None)
                self.assertIsNone(cleared.thinking_effort)
                self.assertEqual(cleared.thinking_mode, "enabled")


class ThinkingAdapterTests(TestCase):
    def test_all_common_efforts_on_every_provider_through_real_sdks(self):
        for kind, model in MODELS.items():
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, model=model, constructor=LLM)
                try:
                    for effort in ("low", "medium", "high"):
                        for streaming in (False, True):
                            settings = {"thinking_effort": effort, "thinking_mode": "enabled"}
                            if streaming:
                                with llm.query_stream(**PROMPTS, **settings) as stream:
                                    self.assertTrue(list(stream))
                            else:
                                llm.query(**PROMPTS, **settings)
                            check_effort(self, kind, body(requests), effort)
                            self.assertNotIn("mode", body(requests))
                            for name in ("thinking_mode", "thinking_effort", "thinking_budget"):
                                self.assertNotIn(name, body(requests))
                finally:
                    _close_sync(llm, http)

    def test_disabled_mode_ignores_effort(self):
        for kind in MODELS:
            model = "gemini-2.5-flash" if kind is GeminiLLM else MODELS[kind]
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, model=model)
                try:
                    llm.query(**PROMPTS, thinking_mode="disabled", thinking_effort="high", thinking_budget=2048)
                    payload = body(requests)
                    if kind is OpenAILLM:
                        self.assertEqual(payload["reasoning"]["effort"], "none")
                    elif kind is AnthropicLLM:
                        self.assertEqual(payload["thinking"]["type"], "disabled")
                        self.assertNotIn("output_config", payload)
                    elif kind is GeminiLLM:
                        self.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinking_budget"], 0)
                    elif kind is DeepSeekLLM:
                        self.assertEqual(payload["thinking"]["type"], "disabled")
                        self.assertNotIn("reasoning_effort", payload)
                    else:
                        self.assertIs(payload["think"], False)
                finally:
                    _close_sync(llm, http)

    def test_known_always_on_models_raise_before_generation(self):
        cases = (
            (OpenAILLM, "gpt-6-astra"), (OpenAILLM, "gpt-6.1-sol"),
            (GeminiLLM, "models/gemini-3.8-flash"), (GeminiLLM, "gemini-2.5-pro"),
            (AnthropicLLM, "claude-opus-5-5"), (AnthropicLLM, "claude-sonnet-5-5"),
            (AnthropicLLM, "claude-mythos-5-1"),
        )
        for kind, model in cases:
            with self.subTest(provider=kind.provider, model=model):
                with client(kind, model=model) as (llm, _):
                    llm._send = Mock()
                    with self.assertRaisesRegex(ValueError, "cannot be.*disabled"):
                        llm.query(**PROMPTS, thinking={"mode": "disabled"})
                    llm._send.assert_not_called()

    def test_native_values_and_extra_options_are_preserved(self):
        openai = adapt_thinking("openai", {"reasoning": {"context": "current_turn"}},
                                ThinkingConfig(effort="xhigh", mode="pro", others={"summary": "auto"}), "gpt-5.6-sol")
        self.assertEqual(openai["reasoning"], {"context": "current_turn", "effort": "xhigh", "mode": "pro", "summary": "auto"})
        claude = adapt_thinking("anthropic", {}, ThinkingConfig(effort="max", mode="adaptive"), "claude-opus-5-5")
        self.assertEqual(claude["thinking"]["type"], "adaptive")
        self.assertEqual(claude["output_config"]["effort"], "max")
        gemini = adapt_thinking("gemini", {}, ThinkingConfig(effort="minimal", others={"include_thoughts": True}), "gemini-3.5-flash")
        self.assertEqual(gemini["thinking_config"], {"thinking_level": "minimal", "include_thoughts": True})
        seek = adapt_thinking("deepseek", {}, ThinkingConfig(effort="max"), "deepseek-flash")
        self.assertEqual(seek["reasoning_effort"], "max")

    def test_gemini_legacy_levels_and_dynamic_enable_use_budgets(self):
        for level, budget in (("low", 1024), ("medium", 8192), ("high", 24576)):
            mapped = adapt_thinking("gemini", {}, ThinkingConfig(effort=level), "gemini-2.5-flash")
            self.assertEqual(mapped["thinking_config"], {"thinking_budget": budget})
        mapped = adapt_thinking("gemini", {}, ThinkingConfig(mode="enabled"), "gemini-2.5-flash-lite")
        self.assertEqual(mapped["thinking_config"], {"thinking_budget": -1})

    def test_claude_manual_budget_requires_compatible_configuration(self):
        thinking = ThinkingConfig(mode="enabled", others={"budget_tokens": 2048})
        params = adapt_thinking("anthropic", {"max_output_tokens": 4096}, thinking, "claude-haiku-4-5")
        self.assertEqual(params["thinking"], {"type": "enabled", "budget_tokens": 2048})
        with self.assertRaisesRegex(ValueError, "budget_tokens"):
            adapt_thinking("anthropic", {}, ThinkingConfig(mode="enabled"), "claude-haiku-4-5")
        with self.assertRaisesRegex(ValueError, "manual thinking budgets"):
            adapt_thinking("anthropic", {"max_output_tokens": 4096}, thinking, "claude-opus-5-5")

    def test_flat_budgets_and_dictionary_budgets_generate_identical_native_requests(self):
        for kind, model in ((AnthropicLLM, "claude-sonnet-4-6"), (GeminiLLM, "gemini-2.5-flash")):
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, model=model, max_output_tokens=8192)
                try:
                    for streaming in (False, True):
                        for settings in (
                            {"thinking_mode": "enabled", "thinking_budget": 2048},
                            {"thinking": {"mode": "enabled", "budget": 2048}},
                        ):
                            if streaming:
                                with llm.query_stream(**PROMPTS, **settings) as stream:
                                    self.assertTrue(list(stream))
                            else:
                                llm.query(**PROMPTS, **settings)
                            payload = body(requests)
                            if kind is AnthropicLLM:
                                self.assertEqual(payload["thinking"], {"type": "enabled", "budget_tokens": 2048})
                            else:
                                self.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinking_budget"], 2048)
                            self.assertNotIn("thinking_budget", payload)
                    self.assertIsNone(llm.config.thinking_budget)
                finally:
                    _close_sync(llm, http)

    def test_budgets_raise_for_unsupported_providers_and_conflicting_controls(self):
        cases = (
            (OpenAILLM, "gpt-5.5", {}), (DeepSeekLLM, "deepseek-flash", {}), (OllamaLLM, "test-thinking", {}),
            (AnthropicLLM, "claude-sonnet-4-6", {"thinking_mode": "adaptive"}),
            (AnthropicLLM, "claude-opus-5-5", {"thinking_mode": "enabled"}),
            (GeminiLLM, "gemini-2.5-flash", {"thinking_effort": "low"}),
        )
        for kind, model, settings in cases:
            with self.subTest(provider=kind.provider, settings=settings), client(kind, model=model) as (llm, _):
                llm._send = Mock()
                with self.assertRaises(ValueError):
                    llm.query(**PROMPTS, thinking_budget=2048, max_output_tokens=8192, **settings)
                llm._send.assert_not_called()

    def test_provider_extensions_mix_with_flat_controls_and_direct_budget_wins(self):
        llm, http, requests = _make_model(OpenAILLM, model="gpt-5.5")
        try:
            llm.query(**PROMPTS, thinking_mode="enabled", thinking_effort="high", thinking={"others": {"summary": "auto"}})
            self.assertEqual(body(requests)["reasoning"], {"effort": "high", "summary": "auto"})
        finally:
            _close_sync(llm, http)
        params = AnthropicLLM._generation_parameters(LLMConfig(
            model="claude-sonnet-4-6", max_output_tokens=8192, **PROMPTS,
            thinking_budget=2048, thinking={"others": {"budget_tokens": 4096}},
        ))
        self.assertEqual(params["thinking"], {"type": "enabled", "budget_tokens": 2048})

    def test_gemini_zero_budget_respects_models_that_cannot_disable_thinking(self):
        for model in ("gemini-3.8-flash", "gemini-2.5-pro"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "cannot be disabled"):
                GeminiLLM._generation_parameters(LLMConfig(model=model, **PROMPTS, thinking_budget=0))
        params = GeminiLLM._generation_parameters(LLMConfig(
            model="gemini-2.5-flash", **PROMPTS, thinking_budget=0,
        ))
        self.assertEqual(params["config"]["thinking_config"]["thinking_budget"], 0)

    def test_native_options_and_caller_configs_are_not_mutated(self):
        params = {"reasoning": {"summary": "auto"}}
        snapshot = deepcopy(params)
        thinking = ThinkingConfig(effort="high", mode="enabled")
        adapt_thinking("openai", params, thinking, "gpt-5.5")
        self.assertEqual(params, snapshot)
        self.assertEqual(thinking.model_fields_set, {"effort", "mode"})

    def test_ollama_metadata_rejects_unsupported_values_instead_of_falling_back(self):
        info = {"thinking": {"values": ["low", "medium", "high"], "default": "medium"}}
        self.assertEqual(validate_ollama_think(True, info, "gpt-oss"), "medium")
        for value in (False, "xhigh", 1):
            with self.assertRaisesRegex(ValueError, "not supported|must be"):
                validate_ollama_think(value, info, "gpt-oss")
        with self.assertRaisesRegex(ValueError, "did not report"):
            validate_ollama_think(False, {}, "unknown")
        llm, http, requests = _make_model(OllamaLLM, model="gpt-oss", thinking_values=("low", "medium", "high"))
        try:
            with self.assertRaisesRegex(ValueError, "not supported"):
                llm.query(**PROMPTS, thinking={"mode": "disabled"})
            self.assertEqual([r.url.path for r in requests], ["/api/show"])
        finally:
            _close_sync(llm, http)

    def test_conflicting_or_unsupported_settings_raise(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            adapt_thinking("openai", {}, ThinkingConfig(mode="enabled", effort="none"), "gpt-5.5")
        with self.assertRaisesRegex(ValueError, "not a native mode"):
            adapt_thinking("gemini", {}, ThinkingConfig(mode="pro"), "gemini-3.8-flash")
        with self.assertRaisesRegex(ValueError, "not supported"):
            adapt_thinking("gemini", {}, ThinkingConfig(effort="minimal"), "gemini-3.8-flash")


class AsyncThinkingTests(IsolatedAsyncioTestCase):
    async def test_common_controls_work_for_async_and_async_streaming(self):
        for kind, model in MODELS.items():
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, model=model, asynchronous=True)
                try:
                    cfg = {"thinking_effort": "high", "thinking_mode": "enabled"}
                    await llm.query_async(**PROMPTS, **cfg)
                    check_effort(self, kind, body(requests), "high")
                    async with await llm.query_stream_async(**PROMPTS, **cfg) as stream:
                        self.assertTrue([event async for event in stream])
                    check_effort(self, kind, body(requests), "high")
                finally:
                    await _close_async(llm, http)

    async def test_budgets_work_for_async_calls_and_async_streams(self):
        for kind, model in ((AnthropicLLM, "claude-sonnet-4-6"), (GeminiLLM, "gemini-2.5-flash")):
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind, model=model, asynchronous=True, max_output_tokens=8192,
                                                  thinking_mode="enabled", thinking_budget=2048)
                try:
                    await llm.query_async(**PROMPTS, thinking={"budget": 4096})
                    payload = body(requests)
                    native = payload["thinking"]["budget_tokens"] if kind is AnthropicLLM else payload["generationConfig"]["thinkingConfig"]["thinking_budget"]
                    self.assertEqual(native, 4096)
                    async with await llm.query_stream_async(**PROMPTS) as stream:
                        self.assertTrue([event async for event in stream])
                    payload = body(requests)
                    native = payload["thinking"]["budget_tokens"] if kind is AnthropicLLM else payload["generationConfig"]["thinkingConfig"]["thinking_budget"]
                    self.assertEqual(native, 2048)
                    self.assertEqual(llm.config.thinking_budget, 2048)
                finally:
                    await _close_async(llm, http)

    async def test_ollama_async_disabled_fails_before_chat_on_a_levels_only_model(self):
        llm, http, requests = _make_model(OllamaLLM, model="gpt-oss", asynchronous=True, thinking_values=("low", "medium", "high"))
        try:
            with self.assertRaisesRegex(ValueError, "not supported"):
                await llm.query_async(**PROMPTS, thinking={"mode": "disabled"})
            self.assertEqual([r.url.path for r in requests], ["/api/show"])
        finally:
            await _close_async(llm, http)


class ResponseReasoningTests(TestCase):
    """LLMResponse.reasoning reads normalized reasoning messages for any provider."""

    @staticmethod
    def make(*contents):
        from datetime import datetime, timezone
        from hiyal.models.schemas.response import LLMResponse, ResponseContent, ResponseMessage

        return LLMResponse(
            text="answer",
            messages=[
                ResponseMessage(role="other" if kind == "reasoning" else "assistant",
                                content=ResponseContent(content_type=kind, text=text))
                for kind, text in contents
            ],
            provider="openai", id_="r", model="m",
            created_at=datetime.now(timezone.utc), status="completed", raw=None,
        )

    def test_joins_reasoning_parts_and_skips_text_and_empty_items(self):
        response = self.make(
            ("reasoning", "first"), ("reasoning", None), ("reasoning", "second"), ("text", "answer"),
        )
        self.assertEqual(response.reasoning, "first\n\nsecond")

    def test_none_when_no_readable_reasoning(self):
        self.assertIsNone(self.make(("text", "answer")).reasoning)
        self.assertIsNone(self.make(("reasoning", None), ("text", "answer")).reasoning)
