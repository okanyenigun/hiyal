from datetime import datetime, timezone
from typing import Any, Literal
from .chat_translator import ChatCompletionsTranslator
from ..bases import StreamAccumulator, Translator
from ..bases.stream_accumulator import TerminalEventType
from ..schemas import (
    ProviderInfo,
    ContentDeltaEvent,
    FinishReason,
    LLMConfig,
    LLMResponse,
    LLMStreamEvent,
    ResponseCancelledEvent,
    ResponseContent,
    ResponseError,
    ResponseMessage,
    ResponseStatus,
    ResponseTerminalEvent,
    ResponseTiming,
    StreamErrorEvent,
    ThinkingConfig,
    TokenUsage,
    ToolCallContent,
    ToolCallDeltaContent,
    UnsupportedThinkingError,
)
from ...prompts.messages import UserPrompt
from ...tools.calls.openai import OpenAIToolSpec
from ...tools.calls.responses import ToolCall, ToolCallDelta
from ...tools.calls.spec import Tools
from ...utility import Dicts


# Model families that reject reasoning.effort="none".
_ALWAYS_REASONING = ("gpt-6-astra", "gpt-6.1-sol", "o1", "o3", "o4")

_STATUS: dict[str, ResponseStatus] = {
    "queued": "pending",
    "in_progress": "in_progress",
    "completed": "completed",
    "incomplete": "incomplete",
    "failed": "failed",
    "cancelled": "cancelled",
}

# OpenAI's ``incomplete_details.reason`` is the only finish signal it gives.
_INCOMPLETE_FINISH_REASON: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}

# Chat Completions spoken output when the call gives no ``audio`` setting.
_DEFAULT_SPOKEN_OUTPUT = {"voice": "alloy", "format": "wav"}

_DELTA_TYPES: dict[str, Literal["text", "refusal", "reasoning"]] = {
    "response.output_text.delta": "text",
    "response.refusal.delta": "refusal",
    "response.reasoning_summary_text.delta": "reasoning",
    "response.reasoning_text.delta": "reasoning",
}
_TOOL_DELTA_TYPES = {
    "response.custom_tool_call_input.delta": "input",
    "response.mcp_call_arguments.delta": "arguments",
}
_TERMINAL_EVENTS: dict[str, TerminalEventType] = {
    "response.completed": "response.completed",
    "response.incomplete": "response.incomplete",
    "response.failed": "response.failed",
}


class OpenAITranslator(Translator):
    """Translates for the OpenAI Responses API.

    Audio in or out is not available there, so those requests go to Chat
    Completions (gpt-audio-* models); ``self.chat`` translates them.
    """

    supports_thinking_budget = False

    def __init__(self, info: ProviderInfo) -> None:
        super().__init__(info)
        self.chat = ChatCompletionsTranslator(info)

    # Request ---------------------------------------------------------------

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        params = super().build_request(config)
        system, user = self.prompt_texts(config)
        params.pop("instructions", None)
        if system is not None:
            params["instructions"] = system
        prompt = config.user_prompt
        params["input"] = (
            self._user_input(prompt)
            if prompt is not None and prompt.has_media
            else user
        )
        if self.uses_chat_completions(config):
            if "IMAGE" in (config.response_modalities or []):
                raise ValueError(
                    "OpenAI cannot return images together with audio, or from an audio prompt"
                )
            return self._chat_parameters(params, config)
        return self._image_output_parameters(params)

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        """Write OpenAI's ``reasoning`` settings (``effort`` plus native options) into ``params``."""
        options = {**Dicts.get(params, "reasoning", copy=True), **thinking.others}
        effort, mode = thinking.effort, thinking.mode
        if mode == "disabled":
            if model.startswith(_ALWAYS_REASONING):
                raise UnsupportedThinkingError(
                    "OpenAI", model, "thinking cannot be disabled"
                )
            options["effort"] = "none"
        else:
            if effort is not None:
                options["effort"] = effort
            if mode == "enabled":
                if options.get("effort") is None:
                    options["effort"] = "medium"
                if options["effort"] == "none":
                    raise UnsupportedThinkingError(
                        "OpenAI", model, "enabled mode conflicts with effort='none'"
                    )
            elif mode is not None:
                options["mode"] = mode
        if options.get("effort") == "none" and model.startswith(_ALWAYS_REASONING):
            raise UnsupportedThinkingError(
                "OpenAI", model, "thinking cannot be disabled"
            )
        if options:
            params["reasoning"] = options

    def tool_parameters(self, tools: Tools) -> dict[str, Any]:
        return {
            "tools": self.tool_definitions(tools, OpenAIToolSpec),
            "tool_choice": tools.tool_choice,
            "parallel_tool_calls": tools.parallel_tool_calls,
        }

    @staticmethod
    def uses_chat_completions(config: LLMConfig) -> bool:
        """Audio in or out needs Chat Completions; everything else uses Responses."""
        audio_prompt = config.user_prompt is not None and bool(config.user_prompt.audio)
        return audio_prompt or "AUDIO" in (config.response_modalities or [])

    @staticmethod
    def _image_output_parameters(params: dict[str, Any]) -> dict[str, Any]:
        """``IMAGE`` output adds the Responses ``image_generation`` tool.

        Its options (quality, size, output_format, ...) come from an OpenAI
        ``image_generation`` setting. The model decides whether to call it.
        """
        modalities = params.pop("response_modalities", None) or ["TEXT"]
        unsupported = set(modalities) - {"TEXT", "IMAGE"}
        if unsupported:
            raise ValueError(f"OpenAI Responses cannot return {sorted(unsupported)}")
        options = params.pop("image_generation", None)
        if "IMAGE" in modalities:
            tool = {**(options or {}), "type": "image_generation"}
            params["tools"] = [*params.get("tools", []), tool]
        elif options is not None:
            raise ValueError(
                "image_generation options need response_modalities to include 'IMAGE'"
            )
        return params

    def _chat_parameters(
        self, params: dict[str, Any], config: LLMConfig
    ) -> dict[str, Any]:
        """Turn Responses-style parameters into a Chat Completions request."""
        assert config.user_prompt is not None
        params.pop("input")
        system = params.pop("instructions", None)
        if params.keys() & {"tools", "tool_choice", "parallel_tool_calls"}:
            raise NotImplementedError(
                "OpenAI tools are not supported with audio prompts yet"
            )
        for responses_name, chat_name in (
            ("max_output_tokens", "max_completion_tokens"),
            ("stop_sequences", "stop"),
        ):
            if responses_name in params:
                params[chat_name] = params.pop(responses_name)
        reasoning = params.pop("reasoning", None)
        if reasoning:
            if reasoning.keys() - {"effort"}:
                raise ValueError(
                    "OpenAI Chat Completions only takes a reasoning effort; "
                    f"got {sorted(reasoning)} for an audio prompt"
                )
            params["reasoning_effort"] = reasoning["effort"]
        modalities = params.pop("response_modalities", None) or ["TEXT"]
        unsupported = set(modalities) - {"TEXT", "AUDIO"}
        if unsupported:
            raise ValueError(
                f"OpenAI Chat Completions cannot return {sorted(unsupported)}"
            )
        if "AUDIO" in modalities:
            # Spoken output always comes with its transcript as text.
            params["modalities"] = ["text", "audio"]
            params.setdefault("audio", dict(_DEFAULT_SPOKEN_OUTPUT))
        system_message = (
            [] if system is None else [{"role": "system", "content": system}]
        )
        params["messages"] = [
            *system_message,
            {
                "role": "user",
                "content": self.chat.chat_user_content(config.user_prompt),
            },
        ]
        return params

    @staticmethod
    def _user_input(prompt: UserPrompt) -> list[dict[str, Any]]:
        """A Responses user message: the text, then ``input_image`` and ``input_file`` parts."""
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt.text}]
        for image in prompt.images:
            part: dict[str, Any] = {"type": "input_image", "detail": "auto"}
            if image.file_id is not None:
                part["file_id"] = image.file_id
            else:
                part["image_url"] = image.url or image.data_url
            content.append(part)
        for document in prompt.documents:
            part = {"type": "input_file"}
            if document.file_id is not None:
                part["file_id"] = document.file_id
            elif document.url is not None:
                part["file_url"] = document.url
            else:
                part["filename"] = document.name
                part["file_data"] = document.data_url
            content.append(part)
        return [{"role": "user", "content": content}]

    # Response --------------------------------------------------------------

    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        if getattr(response, "object", None) == "chat.completion":
            return self.chat.parse_response(response, elapsed_seconds=elapsed_seconds)
        # Synchronous responses are final; ``status`` is only absent on those.
        usage = response.usage
        finish_reason = self._incomplete_finish_reason(response)
        messages = [
            self._parse_output_item(
                item,
                finish_reason=finish_reason,
                generation_completed=(response.status or "completed") == "completed",
            )
            for item in response.output
        ]
        return LLMResponse(
            text=response.output_text or "",
            provider=self.info.provider,
            id_=response.id,
            model=response.model,
            created_at=datetime.fromtimestamp(response.created_at, tz=timezone.utc),
            status=_STATUS[response.status or "completed"],
            messages=messages,
            usage=None
            if usage is None
            else TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                total_tokens=usage.total_tokens,
                cache_read_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
            ),
            timing=self._timing(response),
            error=None
            if response.error is None
            else ResponseError(
                code=response.error.code, message=response.error.message
            ),
            raw=response,
        )

    def stream_accumulator(self, config: LLMConfig) -> StreamAccumulator[Any]:
        if self.uses_chat_completions(config):
            return self.chat.stream_accumulator(config)
        return _OpenAIStreamAccumulator(translator=self)

    def parse_stream_event(
        self, event: Any, *, response_id: str | None
    ) -> LLMStreamEvent | None:
        """Normalize one SDK event; ``None`` means it carries nothing new."""
        event_type = event.type
        response = getattr(event, "response", None)
        common = {
            "provider": self.info.provider,
            "sequence_number": getattr(event, "sequence_number", None),
            "response_id": response_id,
            "raw": event,
        }

        if event_type in _DELTA_TYPES:
            return ContentDeltaEvent(
                event_type="content.delta",
                message_index=event.output_index,
                message_id=event.item_id,
                content_index=self._stream_content_index(event),
                delta=ResponseContent(
                    content_type=_DELTA_TYPES[event_type], text=event.delta
                ),
                **common,
            )

        # The first fragment carries the name. Tracking stays on message_id;
        # provider correlation metadata remains on the original event's item.
        if (
            event_type == "response.output_item.added"
            and event.item.type == "function_call"
        ):
            return ContentDeltaEvent(
                event_type="content.delta",
                message_index=event.output_index,
                message_id=event.item.id,
                content_index=0,
                delta=ToolCallDeltaContent(
                    data=ToolCallDelta(
                        name=event.item.name,
                        arguments_delta=event.item.arguments or None,
                    ),
                ),
                **common,
            )

        if event_type == "response.function_call_arguments.delta":
            return ContentDeltaEvent(
                event_type="content.delta",
                message_index=event.output_index,
                message_id=event.item_id,
                content_index=0,
                delta=ToolCallDeltaContent(
                    data=ToolCallDelta(arguments_delta=event.delta)
                ),
                **common,
            )

        if event_type in _TOOL_DELTA_TYPES:
            return ContentDeltaEvent(
                event_type="content.delta",
                message_index=event.output_index,
                message_id=event.item_id,
                delta=ResponseContent(
                    content_type="other",
                    data={_TOOL_DELTA_TYPES[event_type]: event.delta},
                ),
                **common,
            )

        if event_type in _TERMINAL_EVENTS:
            parsed_response = self.parse_response(response)
            return ResponseTerminalEvent(
                event_type=_TERMINAL_EVENTS[event_type],
                status=parsed_response.status,
                response=parsed_response,
                **common,
            )

        if event_type == "response.cancelled":
            return ResponseCancelledEvent(
                event_type="response.cancelled",
                response=None if response is None else self.parse_response(response),
                **common,
            )

        if event_type == "error":
            return StreamErrorEvent(
                event_type="error",
                error=ResponseError(
                    code=getattr(event, "code", None), message=event.message
                ),
                **common,
            )

        # Lifecycle and snapshot events repeat what deltas or the terminal
        # response already carry.
        return None

    @staticmethod
    def _incomplete_finish_reason(response: Any) -> FinishReason | None:
        """The finish reason of a Responses result; only incomplete ones carry one."""
        # A completed response carries no explicit reason, so none is invented.
        if response.status != "incomplete":
            return None
        details = response.incomplete_details
        if details is None or details.reason is None:
            return "unknown"
        return _INCOMPLETE_FINISH_REASON.get(details.reason, "other")

    @staticmethod
    def _timing(response: Any) -> ResponseTiming | None:
        if response.completed_at is None:
            return None
        # OpenAI reports no duration, only two timestamps (whole seconds).
        return ResponseTiming(
            source="calculated",
            total_seconds=response.completed_at - response.created_at,
        )

    def _parse_tool_call(self, item: Any, *, generation_completed: bool) -> ToolCall:
        """Normalize an invocation without executing or repairing its input."""
        item_status = getattr(item, "status", None)
        complete = item_status == "completed" or (
            item_status is None and generation_completed
        )
        return ToolCall.normalize(
            name=item.name,
            arguments=item.arguments,
            raw=item,
            complete=complete,
            json_encoded=True,
        )

    def _parse_output_item(
        self,
        item: Any,
        *,
        finish_reason: FinishReason | None,
        generation_completed: bool = True,
    ) -> ResponseMessage:
        if item.type == "message" and item.content:
            part = item.content[0]
            if part.type == "refusal":
                content = ResponseContent(content_type="refusal", text=part.refusal)
            else:
                content = ResponseContent(content_type="text", text=part.text)
            return ResponseMessage(
                id=item.id,
                role="assistant",
                content=content,
                finish_reason=finish_reason,
            )

        if item.type == "reasoning":
            texts = [part.text for part in item.summary]
            texts += [part.text for part in item.content or []]
            return ResponseMessage(
                id=item.id,
                role="other",
                content=ResponseContent(
                    content_type="reasoning", text="\n".join(texts) or None
                ),
            )

        if item.type == "image_generation_call" and item.result:
            return ResponseMessage(
                id=item.id,
                role="assistant",
                content=ResponseContent(
                    content_type="image",
                    data={
                        "mime_type": f"image/{item.output_format or 'png'}",
                        "data": item.result,
                    },
                    # The prompt the image tool actually used, when the model rewrote it.
                    text=getattr(item, "revised_prompt", None),
                ),
            )

        if item.type == "function_call":
            return ResponseMessage(
                id=item.id,
                role="assistant",
                content=ToolCallContent(
                    data=self._parse_tool_call(
                        item, generation_completed=generation_completed
                    ),
                ),
            )

        return ResponseMessage(
            id=getattr(item, "id", None),
            role="other",
            content=ResponseContent(content_type="other"),
        )

    @staticmethod
    def _stream_content_index(event: Any) -> int | None:
        return getattr(event, "content_index", getattr(event, "summary_index", None))


class _OpenAIStreamAccumulator(StreamAccumulator[OpenAITranslator]):
    """Per-request state; OpenAI numbers its events and streams its own
    terminal event, so only the response ID is tracked here."""

    def _adapt(self, event: Any) -> tuple[LLMStreamEvent, ...]:
        # Deltas lack the response ID, so learn it from skipped events too.
        response = getattr(event, "response", None)
        self.response_id = getattr(response, "id", None) or self.response_id
        normalized = self.translator.parse_stream_event(
            event, response_id=self.response_id
        )
        return () if normalized is None else (normalized,)
