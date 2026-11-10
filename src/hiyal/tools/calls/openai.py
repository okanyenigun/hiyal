from pydantic import (
    Field,
    model_validator,
)
from typing import Any, ClassVar, Literal, Self
from .spec import ToolSpec, ResultBehavior


class OpenAIToolSpec(ToolSpec):
    """OpenAI Responses function; result_behavior maps to the wire async flag."""

    _provider: ClassVar[str] = "openai"

    type: Literal["function"] = Field(
        default="function",
        description="OpenAI function-tool discriminator; hosted and free-form tools use different definitions.",
        examples=["function"],
    )
    strict: bool | None = Field(
        default=True,
        strict=True,
        description="Our default is True: enforce a strict-compatible argument schema. None sends OpenAI's explicit null value.",
        examples=[True, False, None],
    )
    allowed_callers: list[Literal["direct", "programmatic"]] | None = Field(
        default=None,
        min_length=1,
        description="Permitted invocation routes. Omit for ordinary direct calls; programmatic calls require the hosted programmatic tool.",
        examples=[["direct"], ["direct", "programmatic"]],
    )

    @model_validator(mode="after")
    def validate_strict_input(self) -> Self:
        if self.strict is True:
            self._validate_openai_strict_objects(self.parameters.to_json_schema())
        if (
            self.result_behavior == ResultBehavior.NON_BLOCKING
            and self.allowed_callers
            and "programmatic" in self.allowed_callers
        ):
            raise ValueError(
                "OpenAI non-blocking tools require direct, not programmatic, invocation"
            )
        return self

    def to_sdk_tool(self) -> dict[str, Any]:
        payload = super().to_sdk_tool()
        payload["strict"] = self.strict  # This required SDK key is nullable.
        return payload

    @staticmethod
    def _validate_openai_strict_objects(schema: Any, path: str = "parameters") -> None:
        """Check strict object structure, not the entire OpenAI JSON Schema subset."""
        if not isinstance(schema, dict):
            return
        schema_type = schema.get("type")
        is_object = schema_type == "object" or (
            isinstance(schema_type, list) and "object" in schema_type
        )
        if is_object or "properties" in schema:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if (
                not isinstance(properties, dict)
                or not isinstance(required, list)
                or not all(isinstance(name, str) for name in required)
            ):
                raise ValueError(
                    f"{path}: properties must be an object and required a list"
                )
            if schema.get("additionalProperties") is not False:
                raise ValueError(
                    f"{path}: strict mode requires additionalProperties=False"
                )
            if len(required) != len(properties) or set(required) != set(properties):
                raise ValueError(
                    f"{path}: strict mode requires every property in required"
                )
        for keyword in ("properties", "$defs", "definitions", "patternProperties"):
            children = schema.get(keyword, {})
            if not isinstance(children, dict):
                raise ValueError(f"{path}.{keyword}: must be an object")
            for name, child in children.items():
                OpenAIToolSpec._validate_openai_strict_objects(
                    child, f"{path}.{keyword}.{name}"
                )
        for keyword in (
            "items",
            "additionalProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
        ):
            OpenAIToolSpec._validate_openai_strict_objects(
                schema.get(keyword), f"{path}.{keyword}"
            )
        for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
            children = schema.get(keyword, [])
            if not isinstance(children, list):
                raise ValueError(f"{path}.{keyword}: must be a list")
            for index, child in enumerate(children):
                OpenAIToolSpec._validate_openai_strict_objects(
                    child, f"{path}.{keyword}[{index}]"
                )
