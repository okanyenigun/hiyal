"""Note taker that asks an LLM for the complete updated notes."""

import json
from pydantic import JsonValue, ValidationError
from .base import BaseNoteTaker, NotesUpdate
from ..entry import Entry
from ....models.schemas import LLMResponse
from ....prompts import UserPrompt
from ....utility import Parsing


class NoteTaker(BaseNoteTaker):
    """Sends the current notes, the schema and the new entries; the LLM answers with the complete notes.

    The prompt holds the current notes as JSON (``{}`` when there are none
    yet, so the LLM writes the first ones), the JSON schema of ``schema``
    (field names, types and descriptions), and each entry's ``to_text()``.
    The answer must be the complete notes as JSON; a reply wrapped in a
    Markdown code fence is accepted. An answer that does not fit ``schema``
    raises ``ValueError``.
    """

    def update(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        response = self.llm.query(
            system_prompt=self.system_prompt, user_prompt=self._prompt(notes, entries)
        )
        return self._checked(response)

    async def update_async(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        """``update``, with ``llm.query_async``."""
        response = await self.llm.query_async(
            system_prompt=self.system_prompt, user_prompt=self._prompt(notes, entries)
        )
        return self._checked(response)

    def _prompt(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> UserPrompt:
        """The current notes, the schema and the conversation; raises ``ValueError`` for no entries."""
        if not entries:
            raise ValueError("there are no entries to update the notes with")
        conversation = "\n\n".join(entry.to_text() for entry in entries)
        return UserPrompt(
            text=f"Current notes:\n{json.dumps(notes or {}, indent=2, ensure_ascii=False)}\n\n"
            f"Schema:\n{json.dumps(self.schema.model_json_schema(), indent=2)}\n\n"
            f"Conversation since the last update:\n{conversation}"
        )

    def _checked(self, response: LLMResponse) -> NotesUpdate:
        """The notes in ``response``, checked against ``schema``; raises ``ValueError`` if they do not fit."""
        try:
            checked = self.schema.model_validate_json(Parsing.unfenced(response.text))
        except ValidationError as error:
            raise ValueError(
                f"the model's notes do not fit {self.schema.__name__}: {error}"
            ) from error
        return NotesUpdate(checked.model_dump(mode="json"), response)
