"""The errors short-term memory raises for its own reasons."""


class NoNotesSettingsError(ValueError):
    """Notes were asked for (an update, a section set), but the manager was made without notes settings.

    The manager needs ``notes_llm`` and ``notes_schema`` for notes. A
    ``ValueError``, so code that catches those catches it too.
    """

    def __init__(self) -> None:
        super().__init__(
            "this manager has no notes settings (notes_llm and notes_schema)"
        )
