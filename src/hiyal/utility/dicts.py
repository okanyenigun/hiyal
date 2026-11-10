from typing import Any


class Dicts:
    """Helpers for dictionary-valued options inside parameter dictionaries."""

    @staticmethod
    def get(
        params: dict[str, Any],
        key: str,
        *,
        pop: bool = False,
        copy: bool = False,
        owner: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Return ``params[key]`` (default ``{}``), raising ``TypeError`` unless it is a dict.

        ``pop`` removes the key from ``params``; ``copy`` returns a new dict
        instead of the stored one. ``owner`` and ``label`` only shape the
        error message: ``"<owner> '<label or key>' must be a dictionary"``.
        """
        value = params.pop(key, {}) if pop else params.get(key, {})
        if not isinstance(value, dict):
            name = label or key
            raise TypeError(
                f"{owner} '{name}' must be a dictionary"
                if owner
                else f"{name} must be a dictionary"
            )
        return dict(value) if copy else value
