from typing import Any, cast

from pydantic_core import PydanticSerializationError, to_jsonable_python


class Raw:
    """Helpers for the provider data kept in ``raw`` fields."""

    @staticmethod
    def to_dict(value: Any) -> dict[str, Any] | None:
        """A provider SDK object as a JSON-compatible dict; dicts and ``None`` pass through.

        Every provider SDK object is a Pydantic model, converted with
        ``model_dump(mode="json")``: bytes become base64 text and datetimes ISO
        strings, so the result holds only JSON types. A model whose bytes
        cannot be written that way (no base64 setting of its own) is dumped
        to Python first and then converted, with bytes as URL-safe base64,
        the same text Pydantic's base64 setting gives. Type-mismatch warnings
        are silenced: a provider may send fields its SDK does not know yet, and
        this is a conversion, not a validation. Anything else raises
        ``TypeError``, so no provider object is ever kept as it is.
        """
        if value is None or isinstance(value, dict):
            return value
        dump = getattr(value, "model_dump", None)
        if not callable(dump):
            raise TypeError(f"raw must be a provider SDK model, a dict or None, got {type(value).__name__}")
        try:
            return cast(dict[str, Any], dump(mode="json", warnings=False))
        except (PydanticSerializationError, UnicodeDecodeError):
            return cast(dict[str, Any], to_jsonable_python(dump(warnings=False), bytes_mode="base64"))
