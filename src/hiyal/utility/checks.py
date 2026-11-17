class Checks:
    """Helpers for checking arguments."""

    @staticmethod
    def non_empty_string(value: object, name: str) -> str:
        """``value`` when it is a non-empty string; raises ``TypeError`` or ``ValueError`` otherwise.

        ``name`` is the argument's name, used in the error message.
        """
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value

    @staticmethod
    def positive_int(value: object, name: str) -> int:
        """``value`` when it is a positive integer (not a bool); raises ``ValueError`` otherwise.

        ``name`` is the argument's name, used in the error message.
        """
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value
