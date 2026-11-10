"""Short public imports preserve the existing module-level class identities."""

import importlib
from unittest import TestCase

from hiyal import models, prompts


class PublicImportTests(TestCase):
    def test_provider_exports_match_original_classes(self):
        for module_name, class_name in (
            ("anthropic", "AnthropicLLM"),
            ("deepseek", "DeepSeekLLM"),
            ("gemini", "GeminiLLM"),
            ("ollama", "OllamaLLM"),
            ("openai", "OpenAILLM"),
        ):
            with self.subTest(provider=module_name):
                module = importlib.import_module(f"hiyal.models.{module_name}")
                self.assertIs(getattr(models, class_name), getattr(module, class_name))
                self.assertIn(class_name, models.__all__)

    def test_prompt_exports_match_original_classes(self):
        module = importlib.import_module("hiyal.prompts.messages")
        for class_name in ("Prompt", "SystemPrompt", "UserPrompt"):
            with self.subTest(prompt=class_name):
                self.assertIs(getattr(prompts, class_name), getattr(module, class_name))
                self.assertIn(class_name, prompts.__all__)
