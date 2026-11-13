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

    An answer that **leaves out** a section that had content also raises
    ``ValueError``: with defaults in the schema it would still fit, and the
    section would be wiped silently (an answer of ``{}`` would wipe them
    all). A section emptied on purpose is accepted: the answer names it
    with an empty value (``"open_questions": []``).
    """

    def update(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        response = self.llm.query(
            system_prompt=self.system_prompt, user_prompt=self._prompt(notes, entries)
        )
        return self._checked(response, notes)

    async def update_async(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        """``update``, with ``llm.query_async``."""
        response = await self.llm.query_async(
            system_prompt=self.system_prompt, user_prompt=self._prompt(notes, entries)
        )
        return self._checked(response, notes)

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

    def _checked(
        self, response: LLMResponse, notes: dict[str, JsonValue] | None
    ) -> NotesUpdate:
        """The notes in ``response``, checked against ``schema`` and against losing ``notes``' sections.

        Raises ``ValueError`` if they do not fit ``schema``, or if they leave
        out a section that has content in ``notes``.
        """
        text = Parsing.unfenced(response.text)
        try:
            checked = self.schema.model_validate_json(text)
        except ValidationError as error:
            raise ValueError(
                f"the model's notes do not fit {self.schema.__name__}: {error}"
            ) from error
        answered = json.loads(text)  # valid JSON, as the schema accepted it
        left_out = [
            name
            for name, value in (notes or {}).items()
            if self._has_content(value) and name not in answered
        ]
        if left_out:
            raise ValueError(
                f"the model's notes leave out sections that had content: {', '.join(left_out)}; "
                "the notes stay as they were"
            )
        return NotesUpdate(checked.model_dump(mode="json"), response)

    @staticmethod
    def _has_content(value: JsonValue) -> bool:
        """Whether a section's value holds anything: not ``None``, ``""``, ``[]`` or ``{}``."""
        return value not in (None, "", [], {})
