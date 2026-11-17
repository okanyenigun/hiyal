import json
from typing import Any


class Parsing:
    """Helpers for parsing provider payloads."""

    @staticmethod
    def parse_json(value: str) -> Any:
        """Parse a JSON string.

        Empty input gives ``{}``; invalid JSON is returned unchanged.
        """
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def unfenced(text: str) -> str:
        """``text`` stripped, without a surrounding Markdown code fence such as ```json ... ```."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            return "\n".join(stripped.splitlines()[1:-1])
        return stripped
