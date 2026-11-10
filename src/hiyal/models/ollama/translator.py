import base64
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Literal
from .request import ENDPOINT_KEY
from .thinking import apply_ollama_thinking
from ..bases import StreamAccumulator, Translator
from ..schemas import (
    FinishReason,
    LLMConfig,
    LLMResponse,
    LLMStreamEvent,
    ResponseContent,
    ResponseTerminalEvent,
    ResponseTiming,
    ThinkingConfig,
    TokenUsage,
    ToolCallContent,
)
from ...prompts.messages import UserPrompt
from ...tools.calls.ollama import OllamaToolSpec
from ...tools.calls.responses import ToolCall
from ...tools.calls.spec import Tools
from ...utility import Identifiers, Timing


# The server serializes only "stop" and "length"; anything else is preserved
# as unknown rather than guessed.
_FINISH_REASON: dict[str, FinishReason] = {"stop": "stop", "length": "length"}

_INCOMPLETE_DONE_REASONS = {"length"}

# Image-generation settings that go to generate() as top-level arguments.
_IMAGE_GENERATION_FIELDS = ("width", "height", "steps", "options", "keep_alive")

# Settings Ollama takes inside ``options`` rather than as top-level arguments.
OPTION_FIELDS = frozenset(
    {
        "numa",
        "num_ctx",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "low_vram",
        "f16_kv",
        "logits_all",
        "vocab_only",
        "use_mmap",
        "use_mlock",
        "embedding_only",
        "num_thread",
        "num_keep",
        "seed",
        "num_predict",
        "top_k",
        "top_p",
        "tfs_z",
        "typical_p",
        "repeat_last_n",
        "temperature",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "penalize_newline",
        "stop",
    }
)


def _image_mime_type(data: str) -> str:
    """MIME type of a base64 image from its leading bytes; PNG when unknown."""
    head = base64.b64decode(data[:16] + "=" * (-len(data[:16]) % 4))
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class OllamaTranslator(Translator):
    """Translates for Ollama's chat endpoint, and generate for image models."""

    supports_thinking_budget = False

    # Request ---------------------------------------------------------------

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        params = super().build_request(config)
        if "IMAGE" in (params.get("response_modalities") or []):
            return self._image_generation_request(params, config)
        params["messages"] = self.chat_messages(config)
        prompt = config.user_prompt
        if prompt is not None and prompt.images:
            params["messages"][-1]["images"] = self._image_data(prompt)
        params.pop("system", None)
        self.require_text_only(params)
        self._move_options(params)
        return params

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        apply_ollama_thinking(params, thinking, model)

    def tool_parameters(self, tools: Tools) -> dict[str, Any]:
        declarations = self.tool_definitions(tools, OllamaToolSpec)
        if tools.tool_choice == "required":
            raise ValueError(
                "Ollama does not support tool_choice='required'; use 'auto'."
            )
        if tools.tool_choice == "none":
            return {"tools": []}
        if (
            "parallel_tool_calls" in tools.model_fields_set
            and not tools.parallel_tool_calls
        ):
            raise ValueError("Ollama cannot enforce parallel_tool_calls=False.")
        return {"tools": declarations}

    def _move_options(self, params: dict[str, Any]) -> None:
        """Move model settings into Ollama's ``options`` dictionary, in place."""
        options = self.dict_param(params, "options", pop=True)
        for name in OPTION_FIELDS:
            if name in params:
                options[name] = params.pop(name)
        if options:
            params["options"] = options

    @staticmethod
    def _image_data(prompt: UserPrompt) -> list[str]:
        """Prompt images as base64; Ollama runs locally, so it needs the bytes."""
        if any(image.data is None for image in prompt.images):
            raise ValueError("Ollama needs images as inline data, not URLs or file IDs")
        return [image.base64 for image in prompt.images]

    def _image_generation_request(
        self, params: dict[str, Any], config: LLMConfig
    ) -> dict[str, Any]:
        """A generate() request for an image model (experimental in Ollama).

        ``width``, ``height`` and ``steps`` come from Ollama-specific settings.
        """
        unsupported = set(params.pop("response_modalities")) - {"TEXT", "IMAGE"}
        if unsupported:
            raise ValueError(f"Ollama cannot return {sorted(unsupported)}")
        if config.tools is not None:
            raise ValueError("Ollama image generation does not take tools")
        system, text = self.prompt_texts(config)
        request: dict[str, Any] = {
            ENDPOINT_KEY: "generate",
            "model": params.get("model"),
            "prompt": text,
        }
        if system is not None:
            request["system"] = system
        prompt = config.user_prompt
        if prompt is not None and prompt.images:
            request["images"] = self._image_data(prompt)
        self._move_options(params)
        request.update(
            {name: params[name] for name in _IMAGE_GENERATION_FIELDS if name in params}
        )
        return request

    # Response --------------------------------------------------------------

    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        contents = OllamaContents()
        if getattr(response, "message", None) is not None:
            contents.apply(response.message)
        else:
            # A generate() response: optional text, then the generated image.
            if response.response:
                contents.apply(
                    SimpleNamespace(
                        thinking=None, content=response.response, tool_calls=None
                    )
                )
            if response.image:
                contents.contents.append(
                    ResponseContent(
                        content_type="image",
                        data={
                            "mime_type": _image_mime_type(response.image),
                            "data": response.image,
                        },
                    )
                )
                contents.ids.append(Identifiers.generate_id("item"))
        return self.build_response(
            response=response,
            response_id=Identifiers.generate_id("ollama"),
            contents=contents,
            model=model,
            # Ollama reports its own creation time.
            created_at=None,
        )

    def stream_accumulator(self, config: LLMConfig) -> "OllamaStreamAccumulator":
        return OllamaStreamAccumulator(translator=self, model=config.model)

    def build_response(
        self,
        *,
        response: Any,
        response_id: str,
        contents: "OllamaContents",
        model: str | None,
        created_at: datetime | None,
    ) -> LLMResponse:
        """An ``LLMResponse`` from merged parts; used for full and streamed responses."""
        normalized = contents.final_contents(
            complete=bool(response.done)
            and response.done_reason not in _INCOMPLETE_DONE_REASONS,
        )
        # Only the final ``done`` chunk carries a reason.
        finish_reason = (
            self.finish_reason(
                response.done_reason or "", _FINISH_REASON, contents=normalized
            )
            if response.done
            else None
        )
        input_tokens = response.prompt_eval_count
        output_tokens = response.eval_count
        return LLMResponse(
            text=self.text(normalized),
            messages=self.messages(
                contents.ids, normalized, finish_reason=finish_reason
            ),
            provider=self.info.provider,
            id_=response_id,
            model=response.model or model or "",
            # A stream passes its first chunk's time; the last chunk's is later.
            created_at=created_at
            or Timing.parse_datetime(response.created_at)
            or datetime.now(timezone.utc),
            # A response without its ``done`` chunk was cut short.
            status=self.status(
                response.done_reason,
                _INCOMPLETE_DONE_REASONS,
                interrupted=not response.done,
            ),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                # Ollama reports no total; this one is input + output.
                total_tokens=None
                if input_tokens is None or output_tokens is None
                else input_tokens + output_tokens,
            ),
            timing=ResponseTiming(
                source="provider",
                total_seconds=Timing.nanoseconds_to_seconds(response.total_duration),
                load_seconds=Timing.nanoseconds_to_seconds(response.load_duration),
                input_evaluation_seconds=Timing.nanoseconds_to_seconds(
                    response.prompt_eval_duration
                ),
                output_evaluation_seconds=Timing.nanoseconds_to_seconds(
                    response.eval_duration
                ),
            )
            if response.total_duration is not None
            else None,
            raw=response,
        )


class OllamaContents:
    """Ordered response parts built from one message or many stream chunks.

    Ollama returns a single message holding thinking, text and tool calls.
    Each becomes its own ``ResponseMessage``, indexed by first appearance, so
    stream deltas and the final response agree on ``message_index``.
    """

    def __init__(self) -> None:
        self.contents: list[ResponseContent] = []
        # Ollama assigns no IDs; one is generated per part.
        self.ids: list[str] = []
        self._text_index: dict[str, int] = {}

    def apply(self, message: Any) -> list[tuple[int, ResponseContent]]:
        """Merge a message (or chunk) and return the new fragments."""
        fragments: list[tuple[int, ResponseContent]] = []
        parts: tuple[tuple[Literal["reasoning", "text"], str | None], ...] = (
            ("reasoning", message.thinking),
            ("text", message.content),
        )
        for content_type, text in parts:
            if not text:
                continue
            index = self._text_index.get(content_type)
            if index is None:
                index = self._text_index[content_type] = len(self.contents)
                self.contents.append(
                    ResponseContent(content_type=content_type, text="")
                )
                self.ids.append(Identifiers.generate_id("item"))
            current = self.contents[index]
            self.contents[index] = current.model_copy(
                update={"text": (current.text or "") + text}
            )
            fragments.append(
                (index, ResponseContent(content_type=content_type, text=text))
            )
        # Tool calls arrive whole, never as fragments.
        for call in message.tool_calls or []:
            content = ToolCallContent(
                data=ToolCall.normalize(
                    name=call.function.name,
                    arguments=call.function.arguments,
                    raw=call,
                    complete=False,
                ),
            )
            self.contents.append(content)
            self.ids.append(Identifiers.generate_id("item"))
            fragments.append((len(self.contents) - 1, content))
        return fragments

    def final_contents(self, *, complete: bool) -> list[ResponseContent]:
        """Contents with tool calls marked complete or incomplete for the whole response."""
        return [
            ToolCallContent(
                data=ToolCall.normalize(
                    name=content.data.name,
                    arguments=content.data.arguments,
                    raw=content.data.raw,
                    complete=complete,
                    issue=content.data.issue,
                )
            )
            if isinstance(content, ToolCallContent)
            else content
            for content in self.contents
        ]


class OllamaStreamAccumulator(StreamAccumulator[OllamaTranslator]):
    """Per-request state; the final ``done`` chunk carries reason and metrics."""

    def __init__(self, *, translator: OllamaTranslator, model: str | None) -> None:
        super().__init__(translator=translator, model=model)
        self.created_at: datetime | None = None
        self.contents = OllamaContents()
        # Ollama responses have no ID; one is generated per request.
        self.response_id = Identifiers.generate_id("ollama")

    def _adapt(self, event: Any) -> list[LLMStreamEvent]:
        self.created_at = self.created_at or Timing.parse_datetime(event.created_at)
        return [
            self._delta(
                message_index=index,
                message_id=self.contents.ids[index],
                content=content,
            )
            for index, content in self.contents.apply(event.message)
        ]

    def finalize(self) -> tuple[ResponseTerminalEvent, ...]:
        if self.raw is None:
            return ()
        # A stream that ends before a ``done`` chunk was cut short; the
        # response builder marks it incomplete.
        response = self.translator.build_response(
            response=self.raw,
            response_id=self.response_id or "",
            contents=self.contents,
            model=self.model,
            created_at=self.created_at,
        )
        return self._terminal(response)
