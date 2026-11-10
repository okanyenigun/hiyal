from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast
from .thinking import apply_gemini_thinking
from ..bases import StreamAccumulator, Translator
from ..schemas import (
    MessageRole,
    FinishReason,
    LLMConfig,
    LLMResponse,
    LLMStreamEvent,
    ResponseContent,
    ResponseMessage,
    ResponseStatus,
    ResponseTerminalEvent,
    ThinkingConfig,
    TokenUsage,
    ToolCallContent,
)
from ...prompts.messages import UserPrompt
from ...tools.calls.gemini import GeminiToolSpec
from ...tools.calls.responses import ToolCall, ToolCallIssue
from ...tools.calls.spec import Tools
from ...utility import Identifiers


_FINISH_REASON: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "ESCALATION": "content_filter",
    "RECITATION": "recitation",
    "IMAGE_RECITATION": "recitation",
    "LANGUAGE": "unsupported_language",
    "MALFORMED_FUNCTION_CALL": "invalid_output",
    "UNEXPECTED_TOOL_CALL": "invalid_output",
    "TOO_MANY_TOOL_CALLS": "invalid_output",
    "MISSING_THOUGHT_SIGNATURE": "invalid_output",
    "MALFORMED_RESPONSE": "invalid_output",
    "NO_IMAGE": "invalid_output",
    "PUP_LIMITED_DISABLED": "account_restricted",
}

_TOOL_CHOICE_MODES = {"auto": "AUTO", "required": "ANY", "none": "NONE"}


def _enum_value(value: Any) -> str | None:
    return getattr(value, "value", value)


def _terminal_reason(value: Any) -> str | None:
    reason = _enum_value(value)
    return None if reason in {None, "", "FINISH_REASON_UNSPECIFIED"} else reason


class GeminiTranslator(Translator):
    """Translates for Gemini ``generate_content``.

    Generation settings travel nested in the request's ``config``.
    """

    # Request ---------------------------------------------------------------

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        params = super().build_request(config)
        system, user = self.prompt_texts(config)
        prompt = config.user_prompt
        params["contents"] = (
            self._user_contents(prompt)
            if prompt is not None and prompt.has_media
            else user
        )
        params.pop("system_instruction", None)
        if system is not None:
            params["system_instruction"] = system
        params.pop("stream", None)

        model = params.pop("model", None)
        contents = params.pop("contents")
        nested_config = self.dict_param(params, "config", pop=True)
        nested_config.update(params)
        # The SDK enables AFC by default, even without tools, and warns about
        # using that execution loop through Models instead of Chats.
        if not nested_config.get("tools"):
            nested_config.setdefault("automatic_function_calling", {"disable": True})
        request: dict[str, Any] = {"model": model, "contents": contents}
        if nested_config:
            request["config"] = nested_config
        return request

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        apply_gemini_thinking(params, thinking, model)

    def tool_parameters(self, tools: Tools) -> dict[str, Any]:
        if (
            tools.tool_choice != "none"
            and not tools.parallel_tool_calls
            and "parallel_tool_calls" in tools.model_fields_set
        ):
            raise ValueError(
                "Gemini cannot enforce parallel_tool_calls=False; omit it or allow parallel calls"
            )
        if any(tool.result_behavior is not None for tool in tools.tool_list):
            raise ValueError(
                "Gemini generate_content does not support Live-only result_behavior"
            )
        return {
            "tools": self.tool_definitions(tools, GeminiToolSpec),
            "tool_config": {
                "function_calling_config": {
                    "mode": _TOOL_CHOICE_MODES[tools.tool_choice]
                }
            },
            "automatic_function_calling": {"disable": True},
        }

    @staticmethod
    def _user_contents(prompt: UserPrompt) -> list[Any]:
        """Media parts first (Gemini's recommended order), then the text.

        Inline media becomes ``inline_data``; a Files API URI (from
        ``upload_file``) or other URL becomes ``file_data``.
        """
        from google.genai import types

        parts: list[Any] = []
        for media in (*prompt.images, *prompt.audio, *prompt.documents):
            if media.file_id is not None:
                raise ValueError(
                    "Gemini references uploaded files by URI, not file_id; use the media upload_file returns"
                )
            if media.url is not None:
                parts.append(
                    types.Part.from_uri(file_uri=media.url, mime_type=media.mime_type)
                )
            else:
                parts.append(
                    types.Part.from_bytes(
                        data=media.data or b"", mime_type=media.mime_type
                    )
                )
        parts.append(types.Part.from_text(text=prompt.text))
        return [types.Content(role="user", parts=parts)]

    # Response --------------------------------------------------------------

    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        buffers = []
        for candidate in (response.candidates or [])[:1]:
            # Parts merge exactly as stream deltas do, so both agree on messages.
            buffer = CandidateBuffer(
                role=self.role(candidate),
                finish_reason=_terminal_reason(candidate.finish_reason),
            )
            for part in self.parts(candidate):
                buffer.append(
                    self.parse_part(
                        part, complete=candidate.finish_reason != "MAX_TOKENS"
                    )
                )
            buffers.append(buffer)
        return LLMResponse(
            text=self.text(buffers[0].contents) if buffers else "",
            messages=[message for buffer in buffers for message in buffer.messages()],
            provider=self.info.provider,
            id_=response.response_id or "",
            model=response.model_version or model or "",
            created_at=response.create_time or created_at or datetime.now(timezone.utc),
            status="incomplete"
            if any(buffer.finish_reason == "MAX_TOKENS" for buffer in buffers)
            else "completed",
            usage=self.parse_usage(response.usage_metadata),
            # Gemini reports no timing, so this is the client-side wall time.
            timing=self.calculated_timing(elapsed_seconds),
            raw=response,
        )

    def stream_accumulator(self, config: LLMConfig) -> "GeminiStreamAccumulator":
        return GeminiStreamAccumulator(translator=self, model=config.model)

    @staticmethod
    def parts(candidate: Any) -> list[Any]:
        if candidate.content is None:
            return []
        return candidate.content.parts or []

    @staticmethod
    def role(candidate: Any) -> MessageRole:
        role = getattr(candidate.content, "role", None)
        if role in {None, "model", "assistant"}:
            return "assistant"
        return "tool" if role == "tool" else "other"

    def parse_part(self, part: Any, *, complete: bool = True) -> ResponseContent:
        """One Gemini part as response content."""
        if part.text is not None:
            return ResponseContent(
                content_type="reasoning" if part.thought else "text", text=part.text
            )
        if part.function_call is not None:
            call = part.function_call
            partial = call.will_continue is True or call.partial_args is not None
            issue = (
                ToolCallIssue(
                    code="unsupported_partial_arguments",
                    message="This backend supplied incremental function fields that are not assembled yet.",
                )
                if partial
                else None
            )
            return ToolCallContent(
                data=ToolCall.normalize(
                    name=call.name,
                    arguments=call.args,
                    raw=part,
                    complete=complete and not partial,
                    issue=issue,
                )
            )
        if part.function_response is not None:
            result = part.function_response
            return ResponseContent(
                content_type="tool_result",
                data={"id": result.id, "name": result.name, "result": result.response},
            )
        media = part.inline_data or part.file_data
        if media is not None:
            mime_type = media.mime_type or ""
            content_type = (
                "image"
                if mime_type.startswith("image/")
                else "audio"
                if mime_type.startswith("audio/")
                else "other"
            )
            data = {"mime_type": media.mime_type}
            if part.inline_data is not None:
                data["data"] = media.data
            else:
                data["uri"] = media.file_uri
            return ResponseContent(content_type=content_type, data=data)
        return ResponseContent(content_type="other")

    @staticmethod
    def parse_usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        return TokenUsage(
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            reasoning_tokens=usage.thoughts_token_count,
            total_tokens=usage.total_token_count,
            cache_read_tokens=usage.cached_content_token_count,
        )


@dataclass
class CandidateBuffer:
    """One candidate's parts; each part becomes its own ``ResponseMessage``."""

    role: MessageRole = "assistant"
    contents: list[ResponseContent] = field(default_factory=list)
    # Gemini parts have no IDs; one is generated per part.
    ids: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    tool_indexes: dict[str, int] = field(default_factory=dict)
    pending_tool: int | None = None

    def append(self, content: ResponseContent) -> int | None:
        """Add a part and return its index, or ``None`` when it is skipped."""
        # Empty text/thought-signature-only parts are metadata, not appendable
        # content. Their original values remain in raw.
        if content.content_type == "other" or (
            content.content_type in {"text", "reasoning"} and not content.text
        ):
            return None
        if isinstance(content, ToolCallContent):
            native = content.data.raw.function_call
            index = self.tool_indexes.get(native.id) if native.id else self.pending_tool
            if index is not None:
                previous = cast(ToolCall, self.contents[index].data)
                if native.id or content.data.name in {None, previous.name}:
                    issue = previous.issue or content.data.issue
                    updates = {
                        "raw": previous.raw,
                        "name": content.data.name or previous.name,
                    }
                    if (
                        issue is not None
                        and issue.code == "unsupported_partial_arguments"
                    ):
                        updates.update(status="incomplete", issue=issue)
                    self.contents[index] = ToolCallContent(
                        data=content.data.model_copy(update=updates)
                    )
                    self.pending_tool = index if native.will_continue else None
                    return index
        # Gemini has no cross-chunk content-part IDs. Consecutive text/thought
        # fragments of the same kind belong to one logical content block.
        if (
            self.contents
            and content.content_type in {"text", "reasoning"}
            and self.contents[-1].content_type == content.content_type
        ):
            previous = self.contents[-1]
            self.contents[-1] = previous.model_copy(
                update={"text": (previous.text or "") + (content.text or "")}
            )
        elif self.contents and self._joins_audio(self.contents[-1], content):
            previous = self.contents[-1]
            before, after = (
                cast(dict[str, Any], previous.data),
                cast(dict[str, Any], content.data),
            )
            self.contents[-1] = previous.model_copy(
                update={"data": {**before, "data": before["data"] + after["data"]}}
            )
        else:
            self.contents.append(content)
            self.ids.append(Identifiers.generate_id("item"))
            if isinstance(content, ToolCallContent):
                native = content.data.raw.function_call
                if native.id:
                    self.tool_indexes[native.id] = len(self.contents) - 1
                self.pending_tool = (
                    len(self.contents) - 1 if native.will_continue else None
                )
        return len(self.contents) - 1

    @staticmethod
    def _joins_audio(previous: ResponseContent, content: ResponseContent) -> bool:
        """Whether inline audio ``content`` continues ``previous`` (same MIME type)."""
        if previous.content_type != "audio" or content.content_type != "audio":
            return False
        before, after = previous.data or {}, content.data or {}
        return (
            isinstance(after.get("data"), bytes)
            and isinstance(before.get("data"), bytes)
            and after["mime_type"] == before["mime_type"]
        )

    def messages(self) -> list[ResponseMessage]:
        for index, content in enumerate(self.contents):
            if isinstance(content, ToolCallContent):
                call = content.data
                unsupported = (
                    call.issue is not None
                    and call.issue.code == "unsupported_partial_arguments"
                )
                complete = (
                    not unsupported
                    and self.finish_reason != "MAX_TOKENS"
                    and (self.finish_reason is not None or call.status == "complete")
                )
                self.contents[index] = ToolCallContent(
                    data=ToolCall.normalize(
                        name=call.name,
                        arguments=call.arguments,
                        raw=call.raw,
                        complete=complete,
                        issue=call.issue,
                    )
                )
        reason = (
            Translator.finish_reason(
                self.finish_reason,
                _FINISH_REASON,
                default="other",
                contents=self.contents,
            )
            # A candidate still generating has no finish reason yet.
            or "unknown"
        )
        if not self.contents:
            # Keep the finish reason of a candidate that returned no parts.
            return [
                ResponseMessage(
                    id=Identifiers.generate_id("item"),
                    role=self.role,
                    content=ResponseContent(content_type="other"),
                    finish_reason=reason,
                )
            ]
        return Translator.messages(
            self.ids, self.contents, finish_reason=reason, role=self.role
        )


class GeminiStreamAccumulator(StreamAccumulator[GeminiTranslator]):
    """Per-request state; cumulative usage is replaced, never added."""

    def __init__(self, *, translator: GeminiTranslator, model: str | None) -> None:
        super().__init__(translator=translator, model=model or "")
        self.created_at = datetime.now(timezone.utc)
        self.usage: TokenUsage | None = None
        self.candidates: dict[int, CandidateBuffer] = {}
        self.blocked = False

    def _adapt(self, event: Any) -> list[LLMStreamEvent]:
        chunk = event
        self.response_id = chunk.response_id or self.response_id
        self.model = chunk.model_version or self.model
        self.created_at = chunk.create_time or self.created_at
        if chunk.usage_metadata is not None:
            self.usage = self.translator.parse_usage(chunk.usage_metadata)
        feedback = chunk.prompt_feedback
        if feedback is not None:
            reason = _enum_value(feedback.block_reason)
            self.blocked = reason not in {None, "", "BLOCKED_REASON_UNSPECIFIED"}

        events: list[LLMStreamEvent] = []
        for position, candidate in enumerate(chunk.candidates or []):
            index = candidate.index if candidate.index is not None else position
            # Alternative candidates remain accessible on each chunk's raw
            # object; never mix their text into the primary output stream.
            if index != 0:
                continue
            buffer = self.candidates.setdefault(index, CandidateBuffer())
            buffer.role = self.translator.role(candidate)
            for part in self.translator.parts(candidate):
                content = self.translator.parse_part(part, complete=False)
                part_index = buffer.append(content)
                if part_index is None:
                    continue
                events.append(
                    self._delta(
                        message_index=part_index,
                        message_id=buffer.ids[part_index],
                        content=content,
                    )
                )
            # The finish reason surfaces on the terminal response, not as its own event.
            reason = _terminal_reason(candidate.finish_reason)
            if reason is not None:
                buffer.finish_reason = reason
        return events

    def finalize(self) -> tuple[ResponseTerminalEvent]:
        # Wait for exhaustion: a finish reason may precede a usage-only chunk
        # or an alternative candidate's final chunk.
        buffers = [self.candidates[index] for index in sorted(self.candidates)]
        complete = self.blocked or (
            bool(buffers)
            and all(buffer.finish_reason is not None for buffer in buffers)
        )
        status: ResponseStatus = "completed" if complete else "incomplete"
        if any(buffer.finish_reason == "MAX_TOKENS" for buffer in buffers):
            status = "incomplete"
        response = LLMResponse(
            text=self.translator.text(buffers[0].contents) if buffers else "",
            messages=[message for buffer in buffers for message in buffer.messages()],
            provider=self.translator.info.provider,
            id_=self.response_id or "",
            model=self.model or "",
            created_at=self.created_at,
            status=status,
            usage=self.usage,
            raw=self.raw,
        )
        return self._terminal(response)
