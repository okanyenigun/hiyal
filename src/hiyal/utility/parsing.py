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
