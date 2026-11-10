from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Any, Literal
from .response import LLMResponse, ResponseContent, ResponseError
from .types import Provider, ResponseStatus


class StreamEventBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Provider = Field(
        description="Provider that emitted the event.",
        examples=["openai"],
    )
    sequence_number: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Arrival order of the raw provider event this came from (0, 1, 2, ...). "
            "Says nothing about structure. OpenAI numbers every event, including "
            "skipped lifecycle events, so gaps are normal; for Gemini and Anthropic "
            "this library counts raw chunks, and events from one chunk share it."
        ),
        examples=[6],
    )
    response_id: str | None = Field(
        default=None,
        description="Provider response identifier when known.",
        examples=["resp_123"],
    )
    raw: Any = Field(
        exclude=True,
        repr=False,
        description="Original SDK event object; excluded from serialization.",
    )


class ContentDeltaEvent(StreamEventBase):
    event_type: Literal["content.delta"] = Field(
        description="Newly generated content.", examples=["content.delta"]
    )
    message_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Which item of the response the delta belongs to, i.e. its position in "
            "LLMResponse.messages. OpenAI: output item (with a reasoning model, "
            "0 is the reasoning item and 1 the visible message). Gemini: part of the "
            "first candidate (e.g. 0 thought, 1 text). Anthropic: content block "
            "(e.g. 0 thinking, 1 text)."
        ),
        examples=[1],
    )
    message_id: str | None = Field(
        default=None,
        description=(
            "ID of the message the delta belongs to; equals that message's id in "
            "the final LLMResponse.messages. Generated when the provider has none."
        ),
        examples=["msg_123"],
    )
    content_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Which part inside that message the delta belongs to. Deltas with the "
            "same message_index and content_index are fragments of one part and "
            "concatenate. Usually 0; Anthropic blocks always have one part (0)."
        ),
        examples=[0],
    )
    delta: ResponseContent = Field(
        description=(
            "Content fragment to append. Text fragments concatenate; a tool call "
            "carries name/argument fragments in ToolCallDelta. Its tracking ID "
            "is message_id, and provider metadata remains on raw."
        ),
    )


class ResponseTerminalEvent(StreamEventBase):
    event_type: Literal[
        "response.completed",
        "response.incomplete",
        "response.failed",
    ] = Field(description="Final event of the stream.", examples=["response.completed"])
    status: ResponseStatus = Field(
        description="Final response status.", examples=["completed"]
    )
    response: LLMResponse = Field(
        description="Complete response, including finish reasons and usage."
    )


class ResponseCancelledEvent(StreamEventBase):
    event_type: Literal["response.cancelled"] = Field(
        description="The response was cancelled.", examples=["response.cancelled"]
    )
    status: Literal["cancelled"] = Field(
        default="cancelled", description="Final response status."
    )
    response: LLMResponse | None = Field(
        default=None, description="Partial response, when the provider returns one."
    )


class StreamErrorEvent(StreamEventBase):
    event_type: Literal["error"] = Field(
        description="The provider reported an error mid-stream.", examples=["error"]
    )
    error: ResponseError = Field(description="Error details.")


LLMStreamEvent = Annotated[
    ContentDeltaEvent
    | ResponseTerminalEvent
    | ResponseCancelledEvent
    | StreamErrorEvent,
    Field(discriminator="event_type"),
]
