from datetime import datetime


class Timing:
    """Helpers for converting provider times and durations."""

    @staticmethod
    def nanoseconds_to_seconds(nanoseconds: int | None) -> float | None:
        """Convert a nanosecond duration to seconds; ``None`` stays ``None``."""
        return None if nanoseconds is None else nanoseconds / 1_000_000_000

    @staticmethod
    def parse_datetime(value: str | None) -> datetime | None:
        """Parse an ISO 8601 timestamp; empty or invalid input gives ``None``."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
