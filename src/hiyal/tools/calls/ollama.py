from pydantic import Field
from typing import Any, ClassVar, Literal
from .spec import ToolSpec


class OllamaToolSpec(ToolSpec):
    """Normalized Ollama function; the SDK has no additional execution flags.

    Ollama's typed schema serialization can drop unsupported schema keywords.
    This model preserves the source schema; it does not promise enforcement.
    """

    _provider: ClassVar[str] = "ollama"

    type: Literal["function"] = Field(
        default="function",
        description="Function-tool discriminator. Name, description, and parameters are wrapped inside function for the SDK.",
        examples=["function"],
    )

    def to_sdk_tool(self) -> dict[str, Any]:
        payload = super().to_sdk_tool()
        return {
            "type": payload["type"],
            "function": {
                "name": payload["name"],
                "description": payload["description"],
                "parameters": payload["parameters"],
            },
        }
