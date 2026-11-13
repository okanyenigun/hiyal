import base64
import json
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime
from typing import Any, Literal, TypeVar
from .usage import ResponseTiming, TokenUsage
from .types import FinishReason, MessageRole, Provider, ResponseStatus
from ...prompts.media import Audio, Image, Media
from ...tools.calls.responses import ToolCall, ToolCallDelta
from ...utility import Raw


MediaT = TypeVar("MediaT", bound=Media)


class ResponseError(BaseModel):
    code: str | None = Field(
        default=None,
        description="Error code from the provider.",
        examples=["invalid_request_error"],
    )
    message: str | None = Field(
        default=None,
        description="Error message from the provider.",
        examples=["The request was invalid."],
    )
    retryable: bool | None = Field(
        default=None,
        description="Whether the error is retryable.",
        examples=[True],
    )


class ResponseContent(BaseModel):
    # Image and audio output keep raw bytes in ``data``; in JSON they are
    # written as base64 so a response can be stored and reloaded.
    model_config = ConfigDict(ser_json_bytes="base64")

    content_type: Literal[
        "text",
        "reasoning",
        "image",
        "audio",
        "tool_call",
        "tool_result",
        "refusal",
        "other",
    ] = Field(
        description="Type of content returned by the model.",
        examples=["text"],
    )
    text: str | None = Field(
        default=None,
        description="Text content returned by the model.",
        examples=["Hello, world!"],
    )
    data: Any | None = Field(
        default=None,
        description="Binary or structured data returned by the model.",
        examples=[{"image_url": "https://example.com/image.png"}],
    )


class ToolCallContent(ResponseContent):
    """Typed application-function content used by migrated provider adapters."""

    model_config = ConfigDict(extra="forbid")

    content_type: Literal["tool_call"] = Field(
        default="tool_call",
        description="An application-function invocation.",
        examples=["tool_call"],
    )
    data: ToolCall = Field(
        description="Normalized complete, incomplete, or invalid invocation."
    )


class ToolCallDeltaContent(ResponseContent):
    """A function-call fragment; the final invocation appears on the response."""

    model_config = ConfigDict(extra="forbid")

    content_type: Literal["tool_call"] = Field(
        default="tool_call",
        description="An application-function stream fragment.",
        examples=["tool_call"],
    )
    data: ToolCallDelta = Field(
        description="New function-call metadata or argument fragment."
    )


class ResponseMessage(BaseModel):
    id: str | None = Field(
        default=None,
        description=(
            "Message ID; equals message_id on this message's stream deltas. "
            "OpenAI's own item ID, generated ('item_<uuid>') for other providers."
        ),
        examples=["msg_123", "item_2b872af4ec12479ea60df39c419d0e5b"],
    )
    role: MessageRole = Field(
        description="Role of the message sender.",
        examples=["assistant"],
    )
    content: ResponseContent = Field(
        description="Content of this normalized response item."
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Reason for stopping generation.",
        examples=["stop"],
    )

    @field_validator("content", mode="before")
    @classmethod
    def _rebuild_tool_calls(cls, value: Any) -> Any:
        """Rebuild tool-call content loaded as plain data as ``ToolCallContent``.

        ``content`` is declared as the general ``ResponseContent``, whose
        ``data`` accepts anything, so a tool call read from JSON would stay a
        dict and ``LLMResponse.tool_calls`` would miss it.
        """
        if isinstance(value, dict) and value.get("content_type") == "tool_call":
            return ToolCallContent.model_validate(value)
        return value


class LLMResponse(BaseModel):
    # ser_json_bytes: image and audio output bytes in message content are
    # written as base64 in JSON; untyped bytes follow the outermost model.
    model_config = ConfigDict(arbitrary_types_allowed=True, ser_json_bytes="base64")
    text: str = Field(
        description="Normalized text output from the model.",
        examples=["Hello, world!"],
    )
    messages: list[ResponseMessage] = Field(
        description="Normalized response items in provider order."
    )
    provider: Provider = Field(description="Provider name.", examples=["openai"])
    id_: str = Field(
        min_length=1,
        description="Response ID; the provider's, or generated when it gives none. Never empty.",
        examples=["resp-123"],
    )
    model: str = Field(description="Model ID.", examples=["gpt-5.5"])
    created_at: datetime = Field(
        description="Response creation timestamp.", examples=["2024-01-01T12:00:00Z"]
    )
    status: ResponseStatus = Field(
        description="Response status.", examples=["completed"]
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Normalized token accounting.",
    )
    timing: ResponseTiming | None = Field(
        default=None,
        description="Normalized timing information, when the provider reports it.",
    )
    error: ResponseError | None = Field(
        default=None,
        description="Error details when status is 'failed'.",
    )
    raw: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "The provider's original response as a JSON-compatible dict, converted "
            "from the SDK object with model_dump(mode='json'). Excluded from "
            "serialization, so None when the response is loaded back from JSON."
        ),
    )

    @field_validator("raw", mode="before")
    @classmethod
    def _raw_as_dict(cls, value: Any) -> dict[str, Any] | None:
        return Raw.to_dict(value)

    @property
    def images(self) -> list[Image]:
        """Images the model returned, in message order (e.g. Gemini image output)."""
        return self._media("image", Image)

    @property
    def audio(self) -> list[Audio]:
        """Audio the model returned, in message order (e.g. speech output)."""
        return self._media("audio", Audio)

    def _media(self, content_type: str, media_class: type[MediaT]) -> list[MediaT]:
        items: list[MediaT] = []
        for message in self.messages:
            content = message.content
            if content.content_type != content_type:
                continue
            if isinstance(content.data, media_class):
                items.append(content.data)
            elif isinstance(content.data, dict) and content.data.get("mime_type"):
                # Adapters store {"mime_type", "data" (bytes or base64) | "uri"}.
                data = content.data.get("data")
                if isinstance(data, str):
                    # Standard or URL-safe base64; a reloaded response holds
                    # the URL-safe form Pydantic writes for bytes.
                    data = base64.b64decode(data.replace("-", "+").replace("_", "/"))
                item = media_class(
                    mime_type=content.data["mime_type"],
                    data=data,
                    url=content.data.get("uri"),
                )
                if isinstance(item, Audio) and content.data.get("transcript"):
                    item.transcript = content.data["transcript"]
                items.append(item)
        return items

    @property
    def reasoning(self) -> str | None:
        """Readable reasoning text, provider-agnostic; None when none was returned.

        Joins every reasoning message in order. Reasoning items without text
        (OpenAI items without a summary, Claude redacted thinking) are skipped,
        so None does not mean the model skipped reasoning; see usage.reasoning_tokens.
        """
        texts = [
            message.content.text
            for message in self.messages
            if message.content.content_type == "reasoning" and message.content.text
        ]
        return "\n\n".join(texts) or None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Typed application calls in message order, including invalid calls."""
        return [
            message.content.data
            for message in self.messages
            if message.content.content_type == "tool_call"
            and isinstance(message.content.data, ToolCall)
        ]

    @property
    def has_tool_calls(self) -> bool:
        """Whether this response contains normalized application-function calls."""
        return bool(self.tool_calls)

    def to_text(self) -> str:
        """The response as plain text for a model to read, e.g. in a summary prompt.

        The visible text, then each tool call with its arguments, then a
        placeholder for each image or audio output. Reasoning is left out:
        it is the model's scratch work, not part of the conversation.
        """
        lines = [f"Assistant: {self.text}"] if self.text else []
        for call in self.tool_calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False) if call.arguments is not None else ""
            lines.append(f"Tool call: {call.name or '(unnamed)'}({arguments})")
        lines += [media.placeholder() for media in [*self.images, *self.audio]]
        return "\n".join(lines) or "Assistant: (no output)"
