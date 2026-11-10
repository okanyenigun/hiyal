from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
)
from typing import Any, ClassVar, Literal
from .spec import ToolSpec, ToolParameters


class AnthropicCacheControl(BaseModel):
    """Tool-level prompt-cache breakpoint settings."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ephemeral"] = Field(
        default="ephemeral",
        description="Cache-control discriminator; ephemeral is the supported value.",
        examples=["ephemeral"],
    )
    ttl: Literal["5m", "1h"] = Field(
        default="5m",
        description="Cache breakpoint lifetime; the provider default is five minutes.",
        examples=["5m", "1h"],
    )


class AnthropicToolSpec(ToolSpec):
    """Anthropic Messages client tool, not an OpenAI free-form custom tool."""

    _provider: ClassVar[str] = "anthropic"

    parameters: ToolParameters = Field(
        validation_alias=AliasChoices("parameters", "input_schema"),
        serialization_alias="input_schema",
        description="Normalized tool parameters; mapped to Anthropic's input_schema field.",
        examples=[
            {
                "properties": [
                    {
                        "name": "city",
                        "type": "string",
                        "description": "City to look up.",
                        "required": True,
                    }
                ],
            }
        ],
    )
    type: Literal["custom"] | None = Field(
        default=None,
        description="Optional Anthropic custom-tool discriminator. Omit when unspecified.",
        examples=["custom"],
    )
    allowed_callers: (
        list[
            Literal[
                "direct",
                "code_execution_20250825",
                "code_execution_20260120",
                "code_execution_20260521",
            ]
        ]
        | None
    ) = Field(
        default=None,
        min_length=1,
        description="Permitted direct or versioned code-execution callers from this SDK revision; these are not OpenAI caller identifiers.",
        examples=[["direct"], ["direct", "code_execution_20260521"]],
    )
    cache_control: AnthropicCacheControl | None = Field(
        default=None,
        description="Optional prompt-cache breakpoint on this tool definition.",
        examples=[{"type": "ephemeral", "ttl": "5m"}],
    )
    eager_input_streaming: bool | None = Field(
        default=None,
        strict=True,
        description="Stream tool inputs incrementally. When omitted/None, behavior follows the request's beta headers.",
        examples=[True, False],
    )
    input_examples: list[dict[str, Any]] | None = Field(
        default=None,
        description="Example complete argument objects shown to the model; not actual tool invocations.",
        examples=[[{"city": "Istanbul"}, {"city": "Paris"}]],
    )
