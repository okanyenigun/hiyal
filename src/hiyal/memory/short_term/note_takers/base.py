"""The interface every working-note taker implements."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, JsonValue
from ..entry import Entry
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt
from ....prompts.hub import hub


@dataclass(frozen=True)
class NotesUpdate:
    """What a note taker returns: the new notes, and the LLM's response (for its token usage)."""

    notes: dict[str, JsonValue]
    response: LLMResponse


class BaseNoteTaker(ABC):
    """Writes the new version of a task's working notes, using an LLM.

    A worker: it gets plain data (the current notes and the entries since
    they were written) and returns a ``NotesUpdate``. It knows no history
    and keeps no state. Deciding *when* to update is the caller's job; the
    ``ShortTermMemoryManager`` does it with ``notes_every`` or
    ``refresh_notes()``.

    ``schema`` is a Pydantic model describing the sections: each field is a
    section, and its type and description tell the LLM what belongs there.
    It is used only for generating notes, to describe them and to check the
    LLM's answer; the notes themselves are a plain dict.
    """

    def __init__(
        self,
        llm: BaseLLM,
        schema: type[BaseModel],
        instructions: str | None = None,
        prompt_name: str = "memory/notes",
        prompt_version: str = "v1",
    ) -> None:
        """Use ``llm`` to fill in notes shaped like ``schema``, with the instructions from the prompt hub.

        The instructions are prompt ``prompt_name`` at ``prompt_version`` in
        the hub, so another version can be tried by passing it. Text given
        as ``instructions`` is used instead of the hub's.

        Raises ``TypeError`` if ``schema`` is not a Pydantic model class.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(f"schema must be a Pydantic model class, got {schema!r}")
        self.llm = llm
        self.schema = schema
        self.instructions = (
            instructions
            if instructions is not None
            else hub.get(prompt_name, prompt_version)
        )

    @property
    def system_prompt(self) -> SystemPrompt:
        """The instructions, as the system prompt of each update call."""
        return SystemPrompt(text=self.instructions)

    @abstractmethod
    def update(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        """The new notes: ``notes`` (``None`` for none yet) updated with what ``entries`` say.

        The result is checked against ``schema`` and returned as a plain
        dict. Raises ``ValueError`` when ``entries`` is empty, or when the
        answer does not fit ``schema``.
        """

    async def update_async(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        """``update`` for async code; by default it runs ``update`` in a worker thread.

        A note taker with an async LLM call overrides it (``NoteTaker`` does).
        """
        return await asyncio.to_thread(self.update, notes, entries)
