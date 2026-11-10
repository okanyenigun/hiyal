from datetime import datetime, timezone
from typing import Any
from .thinking import apply_claude_thinking
from ..bases import StreamAccumulator, Translator
from ..schemas import (
    ContentDeltaEvent,
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
    ToolCallDeltaContent,
)
from ...prompts.media import Media
from ...prompts.messages import UserPrompt
from ...tools.calls.anthropic import AnthropicToolSpec
from ...tools.calls.responses import ToolCall, ToolCallDelta, ToolCallIssue
from ...tools.calls.spec import Tools
from ...utility import Identifiers


_FINISH_REASON: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_call",
    "pause_turn": "pause",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_limit",
}

_INCOMPLETE_STOP_REASONS = {"max_tokens", "model_context_window_exceeded"}

# Inline image types the Messages API accepts.
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class AnthropicTranslator(Translator):
    """Translates for the Anthropic Messages API."""

    # The Messages API requires max_tokens; used when the call sets none.
    default_max_tokens = 1024

    # Request ---------------------------------------------------------------

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        params = super().build_request(config)
        params.setdefault("max_tokens", self.default_max_tokens)
        system, user = self.prompt_texts(config)
        params.pop("system", None)
        if system is not None:
            params["system"] = system
        prompt = config.user_prompt
        content = (
            self._user_content(prompt)
            if prompt is not None and prompt.has_media
            else user
        )
        params["messages"] = [{"role": "user", "content": content}]
        # Claude writes text only; image or audio output raises.
        self.require_text_only(params)
        return params

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        apply_claude_thinking(params, thinking, model)

    def tool_parameters(self, tools: Tools) -> dict[str, Any]:
        choice: dict[str, Any] = {
            "type": {"required": "any"}.get(tools.tool_choice, tools.tool_choice)
        }
        if tools.tool_choice != "none":
            choice["disable_parallel_tool_use"] = not tools.parallel_tool_calls
        return {
            "tools": self.tool_definitions(tools, AnthropicToolSpec),
            "tool_choice": choice,
        }

    @staticmethod
    def _source(media: Media, *, inline_type: str) -> dict[str, Any]:
        """Anthropic ``source``: an uploaded file, a URL, or inline base64."""
        if media.file_id is not None:
            return {"type": "file", "file_id": media.file_id}
        if media.url is not None:
            return {"type": "url", "url": media.url}
        return {
            "type": inline_type,
            "media_type": media.mime_type,
            "data": media.base64,
        }

    def _user_content(self, prompt: UserPrompt) -> list[dict[str, Any]]:
        """Images and documents first (Anthropic's recommended order), then the text."""
        content: list[dict[str, Any]] = []
        for image in prompt.images:
            if image.data is not None and image.mime_type not in _IMAGE_TYPES:
                raise ValueError(
                    f"Anthropic images must be JPEG, PNG, GIF or WebP, got {image.mime_type!r}"
                )
            content.append(
                {"type": "image", "source": self._source(image, inline_type="base64")}
            )
        for document in prompt.documents:
            if document.mime_type == "text/plain" and document.data is not None:
                source = {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": document.data.decode(),
                }
            elif (
                document.mime_type == "application/pdf" or document.file_id is not None
            ):
                source = self._source(document, inline_type="base64")
            else:
                raise ValueError(
                    f"Anthropic documents must be PDF or plain text, got {document.mime_type!r}; "
                    "read other text formats into the prompt text"
                )
            block: dict[str, Any] = {"type": "document", "source": source}
            if document.filename:
                block["title"] = document.filename
            content.append(block)
        content.append({"type": "text", "text": prompt.text})
        return content

    # Response --------------------------------------------------------------

    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        return self.build_response(
            message_id=response.id,
            model=response.model,
            contents=[
                self.parse_block(
                    block,
                    complete=response.stop_reason not in _INCOMPLETE_STOP_REASONS,
                )
                for block in response.content
            ],
            content_ids=[Identifiers.generate_id("item") for _ in response.content],
            stop_reason=response.stop_reason,
            usage=response.usage.model_dump(),
            created_at=created_at or datetime.now(timezone.utc),
            # Anthropic reports no timing, so this is the client-side wall time.
            timing=self.calculated_timing(elapsed_seconds),
            raw=response,
        )

    def stream_accumulator(self, config: LLMConfig) -> "AnthropicStreamAccumulator":
        return AnthropicStreamAccumulator(
            translator=self,
            model=config.model,
            created_at=datetime.now(timezone.utc),
        )

    def build_response(
        self,
        *,
        message_id: str,
        model: str,
        contents: list[ResponseContent],
        content_ids: list[str],
        stop_reason: str | None,
        usage: dict[str, Any] | None,
        created_at: datetime,
        interrupted: bool = False,
        timing: ResponseTiming | None = None,
        raw: Any,
    ) -> LLMResponse:
        """An ``LLMResponse`` from parsed blocks; used for full and streamed responses."""
        finish_reason = self.finish_reason(stop_reason, _FINISH_REASON)
        # Anthropic returns one message of typed blocks; each block becomes a
        # message, matching OpenAI's one-message-per-output-item shape.
        return LLMResponse(
            text=self.text(contents),
            messages=self.messages(content_ids, contents, finish_reason=finish_reason),
            provider=self.info.provider,
            id_=message_id,
            model=model,
            # Anthropic reports no creation time; this is the request time.
            created_at=created_at,
            status=self.status(
                stop_reason, _INCOMPLETE_STOP_REASONS, interrupted=interrupted
            ),
            usage=self._parse_usage(usage),
            timing=timing,
            raw=raw,
        )

    def parse_block(self, block: Any, *, complete: bool = True) -> ResponseContent:
        """One Messages content block as response content."""
        if block.type == "text":
            return ResponseContent(content_type="text", text=block.text)
        if block.type == "thinking":
            return ResponseContent(content_type="reasoning", text=block.thinking)
        if block.type == "redacted_thinking":
            return ResponseContent(content_type="reasoning")
        if block.type == "tool_use":
            return ToolCallContent(
                data=ToolCall.normalize(
                    name=block.name,
                    arguments=block.input,
                    raw=block,
                    complete=complete,
                )
            )
        if block.type == "server_tool_use":
            return ResponseContent(content_type="other")
        if block.type.endswith("_tool_result"):
            return ResponseContent(
                content_type="tool_result",
                data={"id": block.tool_use_id, "result": block.content},
            )
        return ResponseContent(content_type="other")

    @staticmethod
    def _parse_usage(usage: dict[str, Any] | None) -> TokenUsage | None:
        if usage is None:
            return None
        details = usage.get("output_tokens_details") or {}
        return TokenUsage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=details.get("thinking_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
            cache_write_tokens=usage.get("cache_creation_input_tokens"),
        )


class AnthropicStreamAccumulator(StreamAccumulator[AnthropicTranslator]):
    """Per-request state built from raw ``create(stream=True)`` events."""

    def __init__(
        self,
        *,
        translator: AnthropicTranslator,
        model: str | None,
        created_at: datetime,
    ) -> None:
        super().__init__(translator=translator, model=model or "")
        self.created_at = created_at
        self.contents: dict[int, ResponseContent] = {}
        # Anthropic content blocks have no IDs; one is generated per block.
        self.block_ids: dict[int, str] = {}
        self.json_buffers: dict[int, str] = {}
        self.tool_blocks: dict[int, Any] = {}
        self.closed_blocks: set[int] = set()
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.stopped = False

    def _adapt(self, event: Any) -> list[LLMStreamEvent]:
        if event.type == "message_start":
            self.response_id = event.message.id
            self.model = event.message.model or self.model
            self.usage = event.message.usage.model_dump()
            return []

        if event.type == "message_delta":
            self.stop_reason = event.delta.stop_reason or self.stop_reason
            # Delta usage is cumulative; absent fields keep their start values.
            self.usage.update(
                {
                    name: value
                    for name, value in event.usage.model_dump().items()
                    if value is not None
                }
            )
            return []

        if event.type == "message_stop":
            self.stopped = True
            return []

        if event.type == "content_block_start":
            content = self.translator.parse_block(event.content_block, complete=False)
            if content.content_type == "tool_call":
                self.tool_blocks[event.index] = event.content_block
                self.json_buffers[event.index] = ""
            self.contents[event.index] = content
            self.block_ids[event.index] = Identifiers.generate_id("item")
            if content.content_type in {"text", "reasoning"} and not content.text:
                return []
            return [self._block_delta(event.index, content)]

        if event.type == "content_block_delta":
            return self._apply_delta(event.index, event.delta)

        if event.type == "content_block_stop":
            self.closed_blocks.add(event.index)
            if event.index in self.tool_blocks:
                self._finish_tool(event.index)
            return []

        return []

    def _apply_delta(self, index: int, delta: Any) -> list[LLMStreamEvent]:
        content = self.contents[index]
        if delta.type in {"text_delta", "thinking_delta"}:
            text = delta.text if delta.type == "text_delta" else delta.thinking
            self.contents[index] = content.model_copy(
                update={"text": (content.text or "") + text}
            )
            return [
                self._block_delta(
                    index, ResponseContent(content_type=content.content_type, text=text)
                )
            ]
        if delta.type == "input_json_delta":
            if not delta.partial_json or index not in self.tool_blocks:
                return []
            self.json_buffers[index] += delta.partial_json
            return [
                self._block_delta(
                    index,
                    ToolCallDeltaContent(
                        data=ToolCallDelta(arguments_delta=delta.partial_json)
                    ),
                )
            ]
        # Signatures and citations annotate a block; they add no new content.
        return []

    def _finish_tool(self, index: int) -> None:
        block = self.tool_blocks[index]
        text = self.json_buffers[index]
        issue = None
        if text and block.input:
            issue = ToolCallIssue(
                code="conflicting_arguments",
                message="Both initial input and streamed JSON were supplied; cannot safely merge them.",
            )
        self.contents[index] = ToolCallContent(
            data=ToolCall.normalize(
                name=block.name,
                arguments=text if text else block.input,
                raw=block,
                complete=(
                    index in self.closed_blocks
                    and self.stop_reason not in _INCOMPLETE_STOP_REASONS
                ),
                json_encoded=bool(text),
                issue=issue,
            )
        )

    def _block_delta(self, index: int, content: ResponseContent) -> ContentDeltaEvent:
        return self._delta(
            message_index=index, message_id=self.block_ids[index], content=content
        )

    def finalize(self) -> tuple[ResponseTerminalEvent]:
        for index in self.tool_blocks:
            self._finish_tool(index)
        response = self.translator.build_response(
            message_id=self.response_id or "",
            model=self.model or "",
            contents=[self.contents[index] for index in sorted(self.contents)],
            content_ids=[self.block_ids[index] for index in sorted(self.contents)],
            stop_reason=self.stop_reason,
            usage=self.usage or None,
            created_at=self.created_at,
            # A stream that ends before ``message_stop`` was cut short.
            interrupted=not self.stopped,
            raw=self.raw,
        )
        return self._terminal(response)
