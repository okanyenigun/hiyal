"""Provider-neutral descriptions of model-requested function calls."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from typing import Any, Literal, Self

# Decodes a JSON object whose values are JSON, rejecting NaN and non-object input.
_ARGUMENTS = TypeAdapter(
    dict[str, JsonValue], config=ConfigDict(strict=True, allow_inf_nan=False)
)
# Pydantic error types -> ToolCallIssue codes; anything else is "invalid_arguments".
_ARGUMENT_ISSUE_CODES = {
    "json_invalid": "invalid_json",
    "dict_type": "arguments_not_object",
}
ToolCallStatus = Literal["complete", "incomplete", "invalid"]


class ToolCallIssue(BaseModel):
    """An assembly or parsing problem affecting one invocation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Local assembly or parsing issue, not a provider request error.",
        examples=["invalid_json", "arguments_not_object", "missing_name"],
    )
    message: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Explanation of why this invocation could not be fully normalized.",
        examples=["Function arguments were truncated before the object closed."],
    )

    @classmethod
    def from_validation_error(cls, error: ValidationError) -> Self:
        """The first validation error as an issue, coded by its kind."""
        detail = error.errors(include_url=False)[0]
        code = _ARGUMENT_ISSUE_CODES.get(detail["type"], "invalid_arguments")
        return cls(code=code, message=detail["msg"])


class ToolCall(BaseModel):
    """A normalized invocation for inspection, not an execution result.

    Producers set ``status`` after assembling provider output. A complete call
    has a name and decoded JSON-object arguments, but has not necessarily been
    validated against its tool definition, authorized, or executed.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    name: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
        description=(
            "Requested function name. None preserves an unnamed invalid or "
            "incomplete invocation for inspection."
        ),
        examples=["get_weather"],
    )
    arguments: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Decoded JSON-object arguments. None means unavailable; an explicit "
            "empty dictionary is an actual empty input. JSON strings and partial "
            "argument fragments must be parsed or assembled by the producer."
        ),
        examples=[{"city": "Istanbul"}, {}],
    )
    status: ToolCallStatus = Field(
        description=(
            "State of the generated invocation, not function execution. Complete "
            "means the producer confirmed generation ended and supplied a name "
            "and argument object; it does not imply tool-schema validation."
        ),
        examples=["complete", "incomplete", "invalid"],
    )
    issue: ToolCallIssue | None = Field(
        default=None,
        description=(
            "Local assembly or parsing problem. Required for invalid invocations "
            "and absent for complete ones."
        ),
        examples=[
            {"code": "missing_arguments", "message": "No arguments were returned."}
        ],
    )
    raw: Any = Field(
        exclude=True,
        repr=False,
        description=(
            "Original provider tool-call item or enclosing part, preserved unchanged. "
            "Provider correlation IDs and other native metadata can be read from "
            "this object. Excluded from serialization; tracking belongs to the "
            "containing ResponseMessage.id."
        ),
    )

    @model_validator(mode="after")
    def validate_call_state(self) -> Self:
        if self.status == "complete":
            if self.name is None:
                raise ValueError("complete tool calls require a name")
            if self.arguments is None:
                raise ValueError("complete tool calls require an argument object")
            if self.issue is not None:
                raise ValueError("complete tool calls cannot have an issue")
        elif self.status == "invalid" and self.issue is None:
            raise ValueError("invalid tool calls require an issue")
        return self

    @classmethod
    def normalize(
        cls,
        *,
        name: str | None,
        arguments: Any,
        raw: Any,
        complete: bool,
        json_encoded: bool = False,
        issue: ToolCallIssue | None = None,
    ) -> Self:
        """Build a call from provider output, recording problems instead of raising.

        Inspects the output without executing or repairing it. ``complete`` says
        whether the provider finished generating the call; with
        ``json_encoded``, ``arguments`` is a JSON string to decode. A missing or
        whitespace-containing name, missing or malformed arguments, or a
        non-object result becomes an ``issue`` and an ``invalid`` status (or
        ``incomplete`` when generation had not finished). An ``issue`` passed in
        takes precedence over one found here.
        """
        if not name or any(character.isspace() for character in name):
            issue = issue or ToolCallIssue(
                code="missing_name" if not name else "invalid_name",
                message="The function call did not supply a valid function name.",
            )
            name = None
        if arguments is None or (json_encoded and not arguments.strip()):
            arguments = None
            issue = issue or ToolCallIssue(
                code="missing_arguments",
                message="No function arguments were returned.",
            )
        elif json_encoded:
            try:
                arguments = _ARGUMENTS.validate_json(arguments)
            except ValidationError as error:
                issue = issue or ToolCallIssue.from_validation_error(error)
                arguments = None
        try:
            return cls(
                name=name,
                arguments=arguments,
                raw=raw,
                issue=issue,
                status=cls._status(complete=complete, issue=issue),
            )
        except ValidationError as error:
            issue = issue or ToolCallIssue.from_validation_error(error)
            return cls(
                name=name,
                arguments=None,
                raw=raw,
                issue=issue,
                status=cls._status(complete=complete, issue=issue),
            )

    @staticmethod
    def _status(*, complete: bool, issue: ToolCallIssue | None) -> ToolCallStatus:
        """``incomplete`` until generation ends, then ``invalid`` if an issue was found."""
        if not complete:
            return "incomplete"
        return "invalid" if issue is not None else "complete"


class ToolCallDelta(BaseModel):
    """Function metadata, argument text, or an object snapshot; never executable."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    name: str | None = Field(
        default=None,
        description="Function name when supplied by this stream fragment.",
        examples=["get_weather"],
    )
    arguments_delta: str | None = Field(
        default=None,
        description="Literal JSON argument text to append for this message's call.",
        examples=['{"city":', '"Istanbul"}'],
    )
    arguments: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Argument-object snapshot from a dictionary-based provider. Replace "
            "the previous object; do not append it as text. Not proof of call completion."
        ),
        examples=[{"city": "Istanbul"}],
    )

    @model_validator(mode="after")
    def validate_argument_update(self) -> Self:
        if self.arguments_delta is not None and self.arguments is not None:
            raise ValueError(
                "a tool update cannot append and replace arguments together"
            )
        return self
