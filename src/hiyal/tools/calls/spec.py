from __future__ import annotations
import json
import inspect
from collections.abc import Callable, Sequence
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    SerializeAsAny,
)
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self


class ArgumentType(StrEnum):
    """Supported JSON Schema argument types; values serialize as strings."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


class ToolArgument(BaseModel):
    """One named argument, including whether the argument is mandatory."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(
        min_length=1,
        pattern=r"^\S+$",
        description="Unique argument name within its containing object.",
        examples=["city"],
    )
    type_: ArgumentType | list[ArgumentType] = Field(
        alias="type",
        description="Argument type, or distinct types for a nullable union; independent of the root object type.",
        examples=["string", ["string", "null"]],
    )
    description: str = Field(
        min_length=1,
        pattern=r"\S",
        description="What this argument means and how the model should populate it.",
        examples=["City to look up."],
    )
    required: bool = Field(
        default=False,
        strict=True,
        description="Whether the argument must be supplied. Used to generate the containing JSON Schema required list.",
        examples=[True, False],
    )
    enum: list[Any] | None = Field(
        default=None,
        min_length=1,
        description="Optional nonempty list of allowed argument values.",
        examples=[["outdoors", "indoors"]],
    )
    items: dict[str, Any] | bool | None = Field(
        default=None,
        description="Explicit JSON Schema fragment for array elements; required for array arguments.",
        examples=[{"type": "string"}],
    )
    properties: list[ToolArgument] | None = Field(
        default=None,
        description=(
            "Use only for an object/dictionary argument. Defines the named keys inside "
            "that dictionary, not additional function arguments. For example, a location "
            "argument can contain city and country keys; each key has its own type, "
            "description, and required flag. Omit for non-object arguments."
        ),
        examples=[
            [
                {
                    "name": "city",
                    "type": "string",
                    "description": "City to look up.",
                    "required": True,
                },
                {
                    "name": "country",
                    "type": "string",
                    "description": "Country containing the city.",
                    "required": True,
                },
            ]
        ],
    )
    additionalProperties: bool | None = Field(
        default=None,
        strict=True,
        description="Whether a dictionary argument accepts keys not declared in properties. Omit to use False when serialized.",
        examples=[False, True],
    )

    @model_validator(mode="after")
    def validate_argument_shape(self) -> Self:
        types = [self.type_] if isinstance(self.type_, str) else self.type_
        if not types or len(types) != len(set(types)):
            raise ValueError("type must contain at least one unique type")
        if ArgumentType.ARRAY in types and self.items is None:
            raise ValueError("array arguments need an items schema")
        if self.items is not None and ArgumentType.ARRAY not in types:
            raise ValueError("items is only valid for an array argument")
        if (
            self.properties is not None or self.additionalProperties is not None
        ) and ArgumentType.OBJECT not in types:
            raise ValueError(
                "properties and additionalProperties require an object argument"
            )
        names = [prop.name for prop in self.properties or []]
        if len(names) != len(set(names)):
            raise ValueError("nested property names must be unique")
        return self

    def to_json_schema(self) -> dict[str, Any]:
        """Remove config-only name/required and serialize nested object fields."""
        schema = self.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
            exclude={"name", "required", "properties"},
        )
        types = [self.type_] if isinstance(self.type_, str) else self.type_
        if ArgumentType.OBJECT in types:
            properties = self.properties or []
            schema["properties"] = {
                prop.name: prop.to_json_schema() for prop in properties
            }
            schema["required"] = [prop.name for prop in properties if prop.required]
            schema.setdefault("additionalProperties", False)
        return schema


class ToolParameters(BaseModel):
    """Normalized object schema with requiredness stored on each property.

    Undeclared fields are rejected. Array item schemas belong to individual
    ToolArgument definitions, not this object root.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,  # TODO: Reconsider accepting both field names and aliases.
    )

    type_: Literal["object"] = Field(
        default="object",
        alias="type",
        description="Root argument type, fixed to object; property types are independent.",
        examples=["object"],
    )
    properties: list[ToolArgument] = Field(
        default_factory=list,
        description="Named argument definitions; each includes its own type, description, and required flag.",
        examples=[
            [
                {
                    "name": "city",
                    "type": "string",
                    "description": "City to look up.",
                    "required": True,
                }
            ]
        ],
    )
    additionalProperties: bool = Field(
        default=False,
        strict=True,
        description="Whether undeclared function arguments are allowed. Our default is False; OpenAI strict mode requires False.",
        examples=[False, True],
    )

    @model_validator(mode="after")
    def validate_property_names(self) -> Self:
        names = [prop.name for prop in self.properties]
        if len(names) != len(set(names)):
            raise ValueError("property names must be unique")
        return self

    def to_json_schema(self) -> dict[str, Any]:
        """Build the standard properties dictionary and required-name list."""
        schema = self.model_dump(
            mode="json", by_alias=True, exclude_none=True, exclude={"properties"}
        )
        schema["properties"] = {
            prop.name: prop.to_json_schema() for prop in self.properties
        }
        schema["required"] = [prop.name for prop in self.properties if prop.required]
        return schema


class ResultBehavior(StrEnum):
    """Whether model generation waits for an application tool's result."""

    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


class ToolSpec(BaseModel):
    """Common function metadata and requested execution policies."""

    _provider: ClassVar[str | None] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", populate_by_name=True
    )

    name: str = Field(
        min_length=1,
        pattern=r"^\S+$",
        description="Function identifier without whitespace; prefer snake_case for portability.",
        examples=["get_weather"],
    )
    description: str = Field(
        min_length=1,
        pattern=r"\S",
        description="What the function does and when the model should request it.",
        examples=["Get the weather for a city."],
    )
    parameters: ToolParameters = Field(
        description="Object schema describing arguments, not argument values or an executable function.",
        examples=[
            {
                "type": "object",
                "properties": [
                    {
                        "name": "city",
                        "type": "string",
                        "description": "City to look up.",
                        "required": True,
                    }
                ],
                "additionalProperties": False,
            }
        ],
    )
    # TODO: Enforce strict=True in the tool-dispatch layer, after arguments arrive
    # and BEFORE executing the application function: validate against parameters,
    # reject invalid calls, and optionally request a bounded model retry. Gemini
    # and Ollama need this hook; native OpenAI/Anthropic strict can also be checked
    # locally. This is acceptance validation, not constrained model generation.
    # Gemini's request-wide VALIDATED mode is separate and must not silently
    # replace per-tool intent. No runtime validation/retry hook is implemented yet.
    strict: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Request schema-valid tool arguments before execution. OpenAI and Anthropic send "
            "their native strict flag; Gemini and Ollama retain this intent only in our config "
            "and omit the unsupported per-tool flag. Their local validation/retry hook is TODO, "
            "so this field does not yet enforce that policy there. False does not request strict "
            "enforcement; None leaves provider behavior unchanged. OpenAI overrides our default to True."
        ),
        examples=[True, False, None],
    )
    result_behavior: ResultBehavior | None = Field(
        default=None,
        description="Whether the model must wait for this tool's result. None preserves provider behavior; non_blocking requires endpoint/model support. Does not control Python async execution.",
        examples=[None, "blocking", "non_blocking"],
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema describing what the application function returns, not the model's final answer. "
            "OpenAI sends output_schema; Gemini sends response_json_schema; Anthropic and Ollama "
            "append it to the tool description instead of sending a separate field. "
            "None omits it. Does not execute the function or validate its returned value."
        ),
        examples=[
            {
                "type": "object",
                "properties": {"temperature_c": {"type": "number"}},
                "required": ["temperature_c"],
                "additionalProperties": False,
            }
        ],
    )

    defer_loading: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Load this tool's definition on demand through tool search. OpenAI and Anthropic "
            "send this flag unchanged and require a compatible tool-search workflow when True. "
            "Gemini and Ollama reject True and omit False/None. None leaves provider behavior unchanged."
        ),
        examples=[True, False, None],
    )

    def to_sdk_tool(self) -> dict[str, Any]:
        """Return a declaration; provider subclasses adapt names and wrappers."""
        payload = self.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
            exclude={"parameters", "result_behavior"},
        )
        if self.parameters is not None:
            field = type(self).model_fields["parameters"]
            schema_key = field.serialization_alias or field.alias or "parameters"
            payload[schema_key] = self.parameters.to_json_schema()
        payload.update(
            self._adapt_result_behavior(self._provider, self.result_behavior)
        )
        if self._provider in ("gemini", "ollama"):
            if self.defer_loading is True:
                raise ValueError(
                    f"{self._provider} does not support defer_loading=True"
                )
            payload.pop("defer_loading", None)
            # TODO: The dispatch hook above must read self.strict before execution.
            # Removing this unsupported SDK field does NOT enforce strictness.
            payload.pop("strict", None)
        if self._provider in ("anthropic", "ollama") and self.output_schema is not None:
            output_schema = payload.pop("output_schema")
            payload["description"] += (
                "\n\nTool output JSON Schema (describes the function's return value):\n"
                + json.dumps(output_schema, ensure_ascii=False, sort_keys=True)
            )
        return payload

    @staticmethod
    def _adapt_result_behavior(
        provider: str | None,
        behavior: ResultBehavior | None,
    ) -> dict[str, Any]:
        """Map provider fields; request code must check endpoint/model support.

        Gemini's explicit behavior field is for the Live API only. Leave behavior
        unset for generate_content, whose normal tool-result loop already waits.
        """
        if behavior is None:
            return {}
        if provider is None:
            raise ValueError(
                "result_behavior needs a provider-specific ToolSpec adapter"
            )
        non_blocking = behavior == ResultBehavior.NON_BLOCKING
        if provider in ("anthropic", "ollama"):
            if non_blocking:
                raise ValueError(
                    f"{provider} does not support non-blocking tool results"
                )
            return {}  # Ordinary tool-result continuation is already blocking.
        if provider == "openai":
            return {"async": non_blocking}
        if provider == "gemini":
            return {"behavior": "NON_BLOCKING" if non_blocking else "BLOCKING"}
        raise ValueError(f"Unknown tool provider: {provider}")


class Tools(BaseModel):
    """Normalized request controls, separate from individual tool definitions.

    Accepts manual specs and plain functions. Decorated functions reuse their
    attached spec; other functions are inspected during construction. Stored
    entries are copied ToolSpec objects, never executable functions.
    """

    model_config = ConfigDict(extra="forbid")

    def __init__(
        self,
        *,
        tool_list: Sequence[ToolSpec | Callable[..., Any] | dict[str, Any]],
        **data: Any,
    ) -> None:
        super().__init__(tool_list=tool_list, **data)

    tool_list: list[SerializeAsAny[ToolSpec]] = Field(
        description=(
            "Manual specs or plain functions on input; normalized to independent "
            "ToolSpec objects. Decorated functions reuse .tool_spec metadata; "
            "undecorated functions are parsed at construction. Provider-subclass options survive."
        ),
        examples=[
            [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "properties": [
                            {
                                "name": "city",
                                "type": "string",
                                "description": "City to look up.",
                                "required": True,
                            }
                        ],
                    },
                }
            ]
        ],
    )
    tool_choice: Literal["auto", "required", "none"] = Field(
        default="auto",
        description="Normalized selection policy: optional calls, at least one call, or no calls. Requires provider-specific translation.",
        examples=["auto", "required", "none"],
    )
    parallel_tool_calls: bool = Field(
        default=False,
        strict=True,
        description=(
            "Request at most one call when False and supported. The default is False; "
            "Gemini and Ollama leave their native behavior unchanged when omitted, "
            "but reject an explicitly supplied False because they cannot enforce it. "
            "True permits multiple calls; it does not execute Python concurrently."
        ),
        examples=[False, True],
    )

    @model_validator(mode="after")
    def validate_required_tool_choice(self) -> Self:
        if self.tool_choice == "required" and not self.tool_list:
            raise ValueError("tool_choice='required' needs at least one tool")
        names = [spec.name for spec in self.tool_list]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique within Tools.tool_list")
        return self

    @field_validator("tool_list", mode="before")
    @classmethod
    def parse_functions(cls, value: Any) -> Any:
        # Import here to keep declaration models independent of parser imports.
        from ..introspection import parse_function

        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            return value
        normalized = []
        for item in value:
            if inspect.isfunction(item):
                if hasattr(item, "tool_spec"):
                    if not isinstance(item.tool_spec, ToolSpec):
                        raise ValueError("function.tool_spec must be a ToolSpec")
                    item = item.tool_spec
                else:
                    item = parse_function(item)
            elif callable(item) and not isinstance(item, ToolSpec):
                raise ValueError(
                    "tools must be plain functions; methods, partials, "
                    "and callable objects are not supported"
                )
            normalized.append(item)
        return normalized

    @field_validator("tool_list")
    @classmethod
    def copy_specs(cls, value: list[ToolSpec]) -> list[ToolSpec]:
        return [spec.model_copy(deep=True) for spec in value]
