"""The manager's working notes: the task's current state, kept by an LLM from the entries it has not read yet."""

import json
from typing import Literal
from pydantic import BaseModel, JsonValue, ValidationError
from ..errors import NoNotesSettingsError
from ..schemas import MANAGED_KINDS, Entry, EntryKind, Notes
from ..storage import BaseStorage
from ....models.bases import BaseLLM
from ....models.schemas import LLMResponse
from ....utility import Extraction


class NotesKeeper:
    """Keeps a session's working notes for the ``ShortTermMemoryManager``: the current version, and what the LLM has not read yet.

    Each version is a notes entry in the session; the latest is the current
    one. A version made by the LLM (``update``) reads the entries the app
    added since the LLM's last version; one set by the app (``set_section``) changes
    one section and reads nothing, so those entries still wait. The manager
    decides *when* (an add, a call); this class does the work.

    For the manager's async side, each method that writes or reads entries
    has an ``_async`` twin, using the storage's async twins and the LLM's
    async client.
    """

    def __init__(
        self,
        storage: BaseStorage,
        session_id: str,
        llm: BaseLLM | None,
        schema: type[BaseModel] | None,
        every: int | None,
        instructions: str,
    ) -> None:
        """Keep the notes of ``session_id`` in ``storage``; none until ``read_back``, ``update`` or ``set``.

        ``llm`` writes the notes into ``schema``, following ``instructions``;
        without them (``None``), there are no notes. ``every``: an update is
        due after this many finished turns; ``None`` means only on demand.
        """
        self.storage = storage
        self.session_id = session_id
        self.llm = llm
        self.schema = schema
        self.every = every
        self.instructions = instructions
        self._notes_id: str | None = None  # the current version
        self._since: list[
            str
        ] = []  # the IDs the app added since the LLM's last version
        self._turns = 0  # how many turns those finish

    def read_back(self, index: list[tuple[str, EntryKind]]) -> None:
        """Rebuild the notes from the session's ``index``: the latest version, and the app's entries after the LLM's latest.

        For a session opened again, or rewound. Only the notes versions down
        to the LLM's latest, and (with ``every``) the entries since it, are
        read.
        """
        notes_ids = [entry_id for entry_id, kind in index if kind is EntryKind.NOTES]
        self._notes_id = notes_ids[-1] if notes_ids else None
        start = 0
        for notes_id in reversed(notes_ids):
            version = self.storage.get(self.session_id, [notes_id])[0]
            if isinstance(version.payload, Notes) and version.payload.source == "llm":
                start = (
                    next(i for i, (stored, _) in enumerate(index) if stored == notes_id)
                    + 1
                )
                break
        self._since = [
            entry_id for entry_id, kind in index[start:] if kind not in MANAGED_KINDS
        ]
        self._turns = (
            sum(
                map(self._finishes_turn, self.storage.get(self.session_id, self._since))
            )
            if self.every is not None
            else 0
        )

    def added(self, entry: Entry) -> None:
        """Note that the app added ``entry``: it waits for the LLM's next update."""
        self._since.append(entry.id_)
        self._turns += self._finishes_turn(entry)

    def forget(self, entry: Entry) -> None:
        """Forget the removed ``entry``: it is no longer sent, and its finished turn no longer counts."""
        if entry.id_ in self._since:
            self._since.remove(entry.id_)
            self._turns -= self._finishes_turn(entry)

    @property
    def due(self) -> bool:
        """Whether an update is due: ``every`` turns finished since the LLM's last version."""
        return self.every is not None and self._turns >= self.every

    def update(self) -> Entry:
        """Have the LLM update the notes from the entries since its last version; returns the new version.

        The LLM gets the current notes (``{}`` before the first version) and
        those entries, and answers with the complete notes, checked against
        ``schema`` (``Extraction``). Raises ``NoNotesSettingsError``
        without notes settings, and ``ValueError`` when nothing arrived since
        the last update or for an answer that is not the notes; the notes
        then stay as they were.
        """
        llm, schema = self._ready()
        since = self.storage.get(self.session_id, self._since)
        text = self._text(self.current, since)
        notes = Extraction.extract(llm, self.instructions, text, schema)
        entry = self._store(notes, source="llm")
        self._since, self._turns = [], 0
        return entry

    async def update_async(self) -> Entry:
        """``update``, with the storage's async twins and the LLM's async client."""
        llm, schema = self._ready()
        since = await self.storage.get_async(self.session_id, self._since)
        text = self._text(await self.current_async(), since)
        notes = await Extraction.extract_async(llm, self.instructions, text, schema)
        entry = await self._store_async(notes, source="llm")
        self._since, self._turns = [], 0
        return entry

    def set_section(self, section: str, value: JsonValue) -> Entry:
        """Set one section to ``value``, without an LLM call; returns the new version.

        The new version is the current notes (none before the first) with
        ``section`` set, checked against the whole ``schema`` and stored as
        the schema writes it: every section, by field name. The entries
        waiting for the LLM still wait. Raises, changing nothing,
        ``NoNotesSettingsError`` without notes settings, and ``ValueError``
        when ``section`` is not a field of ``schema`` or the notes with
        ``value`` do not fit it.
        """
        schema = self._schema_with(section)
        notes = self._fit(schema, self.current, section, value)
        return self._store(notes, source="app")

    async def set_section_async(self, section: str, value: JsonValue) -> Entry:
        """``set_section``, with the storage's async twins."""
        schema = self._schema_with(section)
        notes = self._fit(schema, await self.current_async(), section, value)
        return await self._store_async(notes, source="app")

    @property
    def current(self) -> Entry | None:
        """The current version, or ``None`` before the first; only that entry is read."""
        if self._notes_id is None:
            return None
        return self.storage.get(self.session_id, [self._notes_id])[0]

    async def current_async(self) -> Entry | None:
        """``current``, with the storage's async twins."""
        if self._notes_id is None:
            return None
        return (await self.storage.get_async(self.session_id, [self._notes_id]))[0]

    def _ready(self) -> tuple[BaseLLM, type[BaseModel]]:
        """The LLM and the schema, for an update; raises ``NoNotesSettingsError`` without them, ``ValueError`` when nothing arrived since the last update."""
        if self.llm is None or self.schema is None:
            raise NoNotesSettingsError()
        if not self._since:
            raise ValueError("nothing arrived since the last notes update")
        return self.llm, self.schema

    def _schema_with(self, section: str) -> type[BaseModel]:
        """The schema, when ``section`` is one of its fields; raises ``ValueError`` otherwise, and ``NoNotesSettingsError`` without notes settings."""
        schema = self.schema
        if schema is None:
            raise NoNotesSettingsError()
        if section not in schema.model_fields:
            raise ValueError(
                f"{section!r} is not a section of {schema.__name__}: {', '.join(schema.model_fields)}"
            )
        return schema

    @classmethod
    def _fit(
        cls,
        schema: type[BaseModel],
        current: Entry | None,
        section: str,
        value: JsonValue,
    ) -> BaseModel:
        """The ``current`` notes with ``section`` set to ``value``, checked against ``schema``; raises ``ValueError`` when they do not fit."""
        try:
            return schema.model_validate(
                {**cls._sections(current), section: value}, by_name=True
            )
        except ValidationError as error:
            raise ValueError(
                f"{value!r} does not fit section {section!r} of {schema.__name__}:\n{error}"
            ) from error

    @classmethod
    def _text(cls, current: Entry | None, since: list[Entry]) -> str:
        """What the LLM reads for an update: the ``current`` notes, then the entries ``since`` its last version."""
        return (
            f"Current notes:\n{json.dumps(cls._sections(current), indent=2, ensure_ascii=False)}\n\n"
            "Conversation since the last update:\n"
            + "\n\n".join(entry.to_text() for entry in since)
        )

    @staticmethod
    def _sections(current: Entry | None) -> dict[str, JsonValue]:
        """The sections of the ``current`` notes; none before the first version."""
        if current is not None and isinstance(current.payload, Notes):
            return current.payload.sections
        return {}

    def _store(self, notes: BaseModel, source: Literal["llm", "app"]) -> Entry:
        """Store ``notes`` as a new version made by ``source``, as the schema writes them, and make it the current one."""
        entry = Entry.of(Notes(sections=notes.model_dump(mode="json"), source=source))
        self.storage.add(self.session_id, entry)
        self._notes_id = entry.id_
        return entry

    async def _store_async(
        self, notes: BaseModel, source: Literal["llm", "app"]
    ) -> Entry:
        """``_store``, with the storage's async twins."""
        entry = Entry.of(Notes(sections=notes.model_dump(mode="json"), source=source))
        await self.storage.add_async(self.session_id, entry)
        self._notes_id = entry.id_
        return entry

    @staticmethod
    def _finishes_turn(entry: Entry) -> bool:
        """Whether ``entry`` is the model's answer that ends a turn: a response with no tool calls."""
        return isinstance(entry.payload, LLMResponse) and not entry.payload.tool_calls
