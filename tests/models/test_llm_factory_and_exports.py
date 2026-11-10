import importlib

import pytest
from pydantic import ValidationError

import hiyal
from hiyal import models, prompts, tools, utility
from hiyal.models import (
    LLM,
    AnthropicLLM,
    BaseLLM,
    DeepSeekLLM,
    GeminiLLM,
    LLMConfig,
    OllamaLLM,
    OpenAILLM,
)

PROVIDERS = {"openai": OpenAILLM, "anthropic": AnthropicLLM, "gemini": GeminiLLM, "ollama": OllamaLLM,
             "deepseek": DeepSeekLLM}


class TestFactory:
    @pytest.mark.parametrize("name, cls", PROVIDERS.items())
    def test_returns_the_exact_provider_type(self, name, cls):
        llm = LLM(provider=name, model="m", api_key="k")
        assert type(llm) is cls and isinstance(llm, BaseLLM) and llm.info.provider == name

    def test_config_and_keywords_are_passed_through(self):
        llm = LLM(provider="openai", config=LLMConfig(model="explicit", temperature=0.9), model="keyword",
                  temperature=0.1, top_p=0.5, api_key="k")
        assert (llm.config.model, llm.config.temperature, llm.config.top_p) == ("explicit", 0.9, 0.5)

    def test_unknown_provider_lists_the_supported_ones(self):
        with pytest.raises(ValueError, match="Unsupported provider 'mistral'. Supported providers: anthropic, "
                                             "deepseek, gemini, ollama, openai"):
            LLM(provider="mistral")

    def test_provider_must_be_a_string(self):
        with pytest.raises(TypeError, match="provider must be a supported provider name"):
            LLM(provider=OpenAILLM)

    def test_provider_is_required(self):
        with pytest.raises(TypeError):
            LLM(model="m")

    def test_invalid_settings_still_fail(self):
        with pytest.raises(ValidationError):
            LLM(provider="openai", temperature=-1, api_key="k")


class TestExports:
    def test_version(self):
        assert hiyal.__version__ == "0.0.1"

    @pytest.mark.parametrize("module_name, names", [
        ("hiyal.models", models.__all__),
        ("hiyal.prompts", prompts.__all__),
        ("hiyal.tools", tools.__all__),
        ("hiyal.utility", utility.__all__),
        ("hiyal.models.bases", importlib.import_module("hiyal.models.bases").__all__),
        ("hiyal.models.schemas", importlib.import_module("hiyal.models.schemas").__all__),
    ])
    def test_every_exported_name_exists(self, module_name, names):
        module = importlib.import_module(module_name)
        for name in names:
            assert getattr(module, name, None) is not None, f"{module_name}.{name}"

    @pytest.mark.parametrize("name, cls", PROVIDERS.items())
    def test_provider_packages_export_the_same_class(self, name, cls):
        module = importlib.import_module(f"hiyal.models.{name}")
        assert module.__dict__[cls.__name__] is cls and cls.__name__ in models.__all__

    def test_every_provider_declares_its_facts(self):
        for name, cls in PROVIDERS.items():
            assert cls.provider == name and cls.display_name
            assert cls.prompt_media <= {"images", "audio", "documents"}
