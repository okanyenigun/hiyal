from pydantic import AliasChoices, Field
from typing import Any, ClassVar
from .spec import ToolSpec, ToolParameters


class GeminiToolSpec(ToolSpec):
    """One Gemini function declaration, serialized inside a Tool wrapper."""

    _provider: ClassVar[str] = "gemini"

    name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]*$",
        description="Gemini function name; start with a letter/underscore and use letters, digits, underscores, dots, colons, or dashes.",
        examples=["get_weather", "travel.get_weather"],
    )
    parameters: ToolParameters | None = Field(
        default=None,
        validation_alias=AliasChoices("parameters", "parameters_json_schema"),
        serialization_alias="parameters_json_schema",
        description="Normalized tool parameters, mapped to Gemini parameters_json_schema. Omit for a zero-argument function; native typed Schema input is not exposed.",
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
            }
        ],
    )

    def to_sdk_tool(self) -> dict[str, Any]:
        """Explicit result_behavior emits a Live-only field, not for generate_content.

        Endpoint/model compatibility must be checked where the request is sent.
        """
        payload = super().to_sdk_tool()
        if "output_schema" in payload:
            payload["response_json_schema"] = payload.pop("output_schema")
        return {"function_declarations": [payload]}
