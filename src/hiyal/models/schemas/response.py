import base64
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Any, Literal, TypeVar
from .usage import ResponseTiming, TokenUsage
from .types import FinishReason, MessageRole, Provider, ResponseStatus
from ...prompts.media import Audio, Image, Media
from ...tools.calls.responses import ToolCall, ToolCallDelta


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


class LLMResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str = Field(
        description="Normalized text output from the model.",
        examples=["Hello, world!"],
    )
    messages: list[ResponseMessage] = Field(
        description="Normalized response items in provider order."
    )
    provider: Provider = Field(description="Provider name.", examples=["openai"])
    id_: str = Field(description="Response ID.", examples=["resp-123"])
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
    raw: Any = Field(
        exclude=True,
        repr=False,
        description="Original SDK response object; excluded from serialization.",
    )

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
                    data = base64.b64decode(data)
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
