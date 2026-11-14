"""Note taker that asks an LLM for the complete updated notes."""

import json
from pydantic import BaseModel, JsonValue, ValidationError
from .base import BaseNoteTaker, NotesUpdate
from ..entry import Entry, WorkingNotesEntry
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt, UserPrompt
from ....prompts.hub import hub
from ....utility import Checks, Parsing, Tokens
from ....utility.tokens import DEFAULT_ENCODING


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

    **A size limit** (``max_tokens``): the notes lead every prompt and no
    policy drops them, so their lists can grow without bound. When the
    current notes pass ``max_tokens`` (counted as they read in ``active``),
    the update is also asked to condense them, with no extra call: merge
    related items, drop duplicates and what is done, keep everything
    distinct. A condensed answer that empties a section that had content
    is refused (condensing shrinks, it does not delete). If the answer is
    still over the limit, it is kept, and the next update asks again.
    """

    def __init__(
        self,
        llm: BaseLLM,
        schema: type[BaseModel],
        instructions: str | None = None,
        prompt_name: str = "memory/notes",
        prompt_version: str = "v1",
        *,
        max_tokens: int | None = None,
        condense_version: str = "v1",
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """As ``BaseNoteTaker``, plus the size limit.

        - ``max_tokens``: when the current notes pass it, the update also
          condenses them; ``None`` (the default) means no limit.
        - ``condense_version``: the version of the hub prompt
          ``memory/condense_notes``, added to the instructions then.
        - ``encoding``: the tiktoken encoding the notes are counted with.

        Raises ``ValueError`` if ``max_tokens`` is given but not a positive
        integer, or ``encoding`` is unknown.
        """
        super().__init__(llm, schema, instructions, prompt_name, prompt_version)
        self.max_tokens = (
            Checks.positive_int(max_tokens, "max_tokens")
            if max_tokens is not None
            else None
        )
        self.condense_instructions = hub.get("memory/condense_notes", condense_version)
        self.encoding = Tokens.require_encoding(encoding)

    def condenses(self, notes: dict[str, JsonValue] | None) -> bool:
        """Whether updating ``notes`` also condenses them: they pass ``max_tokens``, as they read in ``active``."""
        if self.max_tokens is None or not notes:
            return False
        text = WorkingNotesEntry(notes=notes).to_text()
        return Tokens.count(text, self.encoding) > self.max_tokens

    def update(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        condense = self.condenses(notes)
        response = self.llm.query(
            system_prompt=self._system(condense),
            user_prompt=self._prompt(notes, entries),
        )
        return self._checked(response, notes, condense)

    async def update_async(
        self, notes: dict[str, JsonValue] | None, entries: list[Entry]
    ) -> NotesUpdate:
        """``update``, with ``llm.query_async``."""
        condense = self.condenses(notes)
        response = await self.llm.query_async(
            system_prompt=self._system(condense),
            user_prompt=self._prompt(notes, entries),
        )
        return self._checked(response, notes, condense)

    def _system(self, condense: bool) -> SystemPrompt:
        """The instructions; when condensing, followed by the condense instructions and the size aim."""
        if not condense:
            return self.system_prompt
        assert self.max_tokens is not None
        aim = (
            f"Aim for about {Tokens.to_words(self.max_tokens)} words "
            f"(at most {self.max_tokens} tokens)."
        )
        return SystemPrompt(
            text=f"{self.instructions} {self.condense_instructions} {aim}"
        )

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
        self,
        response: LLMResponse,
        notes: dict[str, JsonValue] | None,
        condense: bool = False,
    ) -> NotesUpdate:
        """The notes in ``response``, checked against ``schema`` and against losing ``notes``' sections.

        Raises ``ValueError`` if they do not fit ``schema``, if they leave
        out a section that has content in ``notes``, or, when condensing,
        if they empty one.
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
        emptied = [
            name
            for name, value in (notes or {}).items()
            if condense
            and self._has_content(value)
            and not self._has_content(answered.get(name))
        ]
        if emptied:
            raise ValueError(
                f"the model's condensed notes empty sections that had content: {', '.join(emptied)}; "
                "condensing must shrink them, not delete them, so the notes stay as they were"
            )
        return NotesUpdate(checked.model_dump(mode="json"), response)

    @staticmethod
    def _has_content(value: JsonValue) -> bool:
        """Whether a section's value holds anything: not ``None``, ``""``, ``[]`` or ``{}``."""
        return value not in (None, "", [], {})
