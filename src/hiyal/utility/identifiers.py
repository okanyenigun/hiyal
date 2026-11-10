from uuid import uuid4


class Identifiers:
    """Helpers for creating identifiers."""

    @staticmethod
    def generate_id(prefix: str) -> str:
        """Return a unique ID for values a provider leaves unidentified."""
        return f"{prefix}_{uuid4().hex}"
