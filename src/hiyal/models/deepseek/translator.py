from typing import Any
from ..openai import ChatCompletionsTranslator
from ..schemas import LLMConfig, ThinkingConfig
from ...utility import Dicts


class DeepSeekTranslator(ChatCompletionsTranslator):
    """Translates for DeepSeek's OpenAI-compatible Chat Completions API.

    Responses and streams are parsed by ``ChatCompletionsTranslator``.
    """

    supports_thinking_budget = False

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        params = super().build_request(config)
        params.pop("system_prompt", None)
        params["messages"] = self.chat_messages(config)
        prompt = config.user_prompt
        if prompt is not None and prompt.has_media:
            # Image input works only on DeepSeek's vision model
            # (deepseek-v4-flash-vision-exp); other models reject the request.
            params["messages"][-1]["content"] = self.chat_user_content(prompt)
        if params.pop("top_k", None) is not None:
            raise ValueError("DeepSeek does not support top_k")
        self.require_text_only(params)
        return params

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        """Write DeepSeek's ``extra_body.thinking`` and ``reasoning_effort`` into ``params``."""
        body = Dicts.get(params, "extra_body", copy=True)
        options = {**Dicts.get(body, "thinking", copy=True), **thinking.others}
        if thinking.mode is not None:
            options["type"] = thinking.mode
        if thinking.mode == "disabled":
            params.pop("reasoning_effort", None)
        elif thinking.effort is not None:
            # DeepSeek documents medium as a compatibility alias for high.
            params["reasoning_effort"] = (
                "high" if thinking.effort == "medium" else thinking.effort
            )
        if options:
            body["thinking"] = options
            params["extra_body"] = body
