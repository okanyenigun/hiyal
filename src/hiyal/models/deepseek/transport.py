import os
from typing import Any
from ..openai import ChatCompletionsTransport


class DeepSeekTransport(ChatCompletionsTransport):
    """Sends Chat Completions requests to DeepSeek through the OpenAI SDK."""

    default_base_url = "https://api.deepseek.com"

    def create_client(self) -> Any:
        if self.config.asynchronous:
            from openai import AsyncOpenAI as OpenAI
        else:
            from openai import OpenAI

        params = self.client_kwargs()
        # Never let the OpenAI SDK fall back to OPENAI_API_KEY.
        params.setdefault("api_key", os.environ.get("DEEPSEEK_API_KEY"))
        if not params["api_key"]:
            raise ValueError(
                "DeepSeek needs an api_key or the DEEPSEEK_API_KEY environment variable"
            )
        others = dict(self.config.client_options)
        others.setdefault("base_url", self.default_base_url)
        return OpenAI(**others, **params)
