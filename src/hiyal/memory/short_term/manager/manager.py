"""The short-term memory manager: the one thing the app talks to; it coordinates the other components."""

import asyncio
from typing import Any, Self
from uuid import uuid4
from pydantic import BaseModel, JsonValue
from .dump import DumpKeeper
from .notes import NotesKeeper
from .pins import PinKeeper
from .summary import SummaryKeeper
from ..schemas import MANAGED_KINDS, Entry, EntryKind
from ..policies import Policy, Selection, text_tokens
from ..storage import BaseStorage
from ..summarizers import SummaryWorker
from ....models.bases import BaseLLM
from ....prompts.hub import hub
from ....utility import Checks


class ShortTermMemoryManager:
    """Manages one session's short-term memory, coordinating the components that do the work.

    The app talks only to the manager; the manager hands the work over:

    ==============  =========================================================
    Component       Its job here
    ==============  =========================================================
    storage         keeps the session's entries (``BaseStorage``)
    policy          chooses the window (a chain of ``Policy`` steps)
    summarizer      summarizes what the window drops as too old (a chain of
                    ``SummaryWorker`` steps)
    notes           keep the task's current state, extracted by an LLM into
                    ``notes_schema`` (``Extraction``)
    dump storage    an archive the entries are copied to
    ==============  =========================================================

    One manager serves one session, from one thread.

    What goes to the model (``active``): the latest system prompt, the
    summary, the notes, the pinned entries and the window, in that order.
    System prompts are instructions, not conversation: the policy never
    sees them, so they are never dropped or summarized, and only the latest
    one is sent (older ones stay in the history).

    **Async.** For async code, open the manager with ``await
    ShortTermMemoryManager.open_async(storage, ...)`` and use the ``_async``
    twin of each method and read (``await manager.add_async(entry)``,
    ``await manager.window_async()``, ...). The twins do the same work
    through the components' async twins, so the LLMs need async clients.
    Await each call before the next: one call at a time, like one thread.
    """

    def __init__(
        self,
        storage: BaseStorage,
        *,
        session_id: str | None = None,
        policy: Policy | None = None,
        summarizer: SummaryWorker | None = None,
        notes_llm: BaseLLM | None = None,
        notes_schema: type[BaseModel] | None = None,
        notes_every: int | None = None,
        notes_prompt_name: str = "memory/notes",
        notes_prompt_version: str = "v1",
        dump_storage: BaseStorage | None = None,
        dump_every: int | None = None,
    ) -> None:
        """Manage the session ``session_id`` in ``storage``; ``None`` starts a new one."""
        # Session
        self.storage = storage
        self._session_id = str(uuid4()) if session_id is None else session_id
        # Settings
        self._policy = policy
        self.summarizer = summarizer
        self.notes_llm = notes_llm
        self.notes_schema = notes_schema
        self.notes_every = notes_every
        self.dump_storage = dump_storage
        self.dump_every = dump_every
        self._validate()
        # Raises ValueError for a prompt or version the hub does not have.
        self.notes_instructions = hub.get(notes_prompt_name, notes_prompt_version)
        # State: each keeper keeps its part, and _resume rebuilds them all from the storage
        # (here, and on a rewind). The manager hands its work over to them.
        self._summary_keeper = SummaryKeeper(
            self.storage, self._session_id, self.summarizer
        )  # the summary
        self._notes_keeper = NotesKeeper(
            self.storage,
            self._session_id,
            self.notes_llm,
            self.notes_schema,
            self.notes_every,
            self.notes_instructions,
        )  # the working notes
        self._pin_keeper = PinKeeper(
            self.storage, self._session_id
        )  # the pinned entries
        self._dump_keeper = DumpKeeper(
            self.storage, self._session_id, self.dump_storage, self.dump_every
        )  # the dump
        self._system_id: str | None = None  # the latest system prompt
        self._window_ids: list[str] = []  # the window, oldest first
        self._resume()

    def _validate(self) -> None:
        """Check the session and the settings, each on its own and how they fit together."""
        # Each on its own
        if not isinstance(self.storage, BaseStorage):
            raise TypeError(
                f"storage must be a BaseStorage, got {type(self.storage).__name__}"
            )
        Checks.non_empty_string(self._session_id, "session_id")
        for name, value, kind in (
            ("policy", self._policy, Policy),
            ("summarizer", self.summarizer, SummaryWorker),
            ("dump_storage", self.dump_storage, BaseStorage),
        ):
            if value is not None and not isinstance(value, kind):
                raise TypeError(
                    f"{name} must be a {kind.__name__}, got {type(value).__name__}"
                )
        if self.notes_schema is not None and not (
            isinstance(self.notes_schema, type)
            and issubclass(self.notes_schema, BaseModel)
        ):
            raise TypeError(
                f"notes_schema must be a Pydantic model class, got {self.notes_schema!r}"
            )
        for name, value in (
            ("notes_every", self.notes_every),
            ("dump_every", self.dump_every),
        ):
            if value is not None:
                Checks.positive_int(value, name)
        # Together
        if (self.notes_llm is None) != (self.notes_schema is None):
            raise ValueError(
                "notes_llm and notes_schema go together: give both or neither"
            )
        if self.notes_every is not None and self.notes_schema is None:
            raise ValueError("notes_every needs notes_llm and notes_schema")
        if self.dump_every is not None and self.dump_storage is None:
            raise ValueError("dump_every needs a dump_storage")

    # ------------------------------------------------------------------ session

    @property
    def session_id(self) -> str:
        """The ID of the session this manager serves: the one given, or a new UUID."""
        return self._session_id

    @classmethod
    async def open_async(cls, storage: BaseStorage, **settings: Any) -> Self:
        """The manager for async code: takes the constructor's arguments, and runs it in a worker thread.

        Opening reads the session back (``_resume``), and a constructor
        cannot await. Reading back needs only the storage and the policy,
        whose sync methods always work (only an LLM client can be async
        only), so the sync read-back runs in a worker thread, off the event
        loop.
        """
        return await asyncio.to_thread(cls, storage, **settings)

    def _resume(self) -> None:
        """Rebuild all the state from the storage: for a session opened again, or rewound."""
        index = self.storage.index(self._session_id)
        self._summary_keeper.read_back(index)
        self._pin_keeper.read_back(index)
        self._notes_keeper.read_back(index)
        self._dump_keeper.read_back(index)
        self._system_id = self._latest_system(index)
        # The window.
        self._choose()

    def _position(
        self, index: list[tuple[str, EntryKind]], entry_id: str, method: str
    ) -> int:
        """Where the app's entry ``entry_id`` is in ``index``; raises ``ValueError`` if it is not stored, or the manager made it."""
        position = next(
            (i for i, (stored, _) in enumerate(index) if stored == entry_id), None
        )
        if position is None:
            raise ValueError(
                f"entry {entry_id!r} is not stored in session {self._session_id!r}"
            )
        kind = index[position][1]
        if kind in MANAGED_KINDS:
            raise ValueError(
                f"{kind.value} entries come from the manager; {method}() takes the app's entries"
            )
        return position

    # ------------------------------------------------------------------ writes

    def add(self, entry: Entry, *, pinned: bool = False) -> bool:
        """Store ``entry`` and update the memory: the window, the summary if entries leave it, the notes and the dump if due."""
        self._check_entry(entry, pinned)
        # The storage refuses a different entry with a stored ID, and reports a repeat as False.
        if not self.storage.add(self._session_id, entry):
            return False
        if entry.kind is EntryKind.SYSTEM_PROMPT:
            self._system_id = entry.id_
        if pinned:
            self._pin_keeper.pin(entry.id_)
        self._update_window()
        self._notes_keeper.added(entry)
        if self._notes_keeper.due:
            self._notes_keeper.update()
        # The dump comes last, so it copies what this add made.
        self._dump_keeper.added()
        if self._dump_keeper.due:
            self._dump_keeper.dump()
        return True

    async def add_async(self, entry: Entry, *, pinned: bool = False) -> bool:
        """``add``, through the components' async twins."""
        self._check_entry(entry, pinned)
        if not await self.storage.add_async(self._session_id, entry):
            return False
        if entry.kind is EntryKind.SYSTEM_PROMPT:
            self._system_id = entry.id_
        if pinned:
            await self._pin_keeper.pin_async(entry.id_)
        await self._update_window_async()
        self._notes_keeper.added(entry)
        if self._notes_keeper.due:
            await self._notes_keeper.update_async()
        self._dump_keeper.added()
        if self._dump_keeper.due:
            await self._dump_keeper.dump_async()
        return True

    def remove(self, entry_id: str) -> None:
        """Remove the entry ``entry_id`` from the session, and every piece of state that points to it."""
        index = self.storage.index(self._session_id)
        self._position(index, entry_id, "remove")
        removed = self.storage.get(self._session_id, [entry_id])[0]
        self.storage.remove(self._session_id, entry_id)
        gone = [entry_id]
        if self._pin_keeper.is_pinned(entry_id):
            gone.append(self._pin_keeper.unpin(entry_id))  # its pin record goes too
        self._forget(index, removed, gone)
        self._choose()

    async def remove_async(self, entry_id: str) -> None:
        """``remove``, through the components' async twins."""
        index = await self.storage.index_async(self._session_id)
        self._position(index, entry_id, "remove")
        removed = (await self.storage.get_async(self._session_id, [entry_id]))[0]
        await self.storage.remove_async(self._session_id, entry_id)
        gone = [entry_id]
        if self._pin_keeper.is_pinned(entry_id):
            gone.append(await self._pin_keeper.unpin_async(entry_id))
        self._forget(index, removed, gone)
        await self._choose_async()

    def unpin(self, entry_id: str) -> None:
        """Unpin the entry ``entry_id``: it goes back to the pool, where the policy chooses it like any other."""
        index = self.storage.index(self._session_id)
        record_id = self._pin_keeper.unpin(entry_id)
        self._dump_keeper.forget([self._index_of(index, record_id)])
        self._update_window()

    async def unpin_async(self, entry_id: str) -> None:
        """``unpin``, through the components' async twins."""
        index = await self.storage.index_async(self._session_id)
        record_id = await self._pin_keeper.unpin_async(entry_id)
        self._dump_keeper.forget([self._index_of(index, record_id)])
        await self._update_window_async()

    def compact(self, max_tokens: int | None = None) -> Entry | None:
        """Move the oldest window entries into the summary now: all of them, or until the window fits ``max_tokens``."""
        self._check_compact(max_tokens)
        moving = self._to_compact(self.window, max_tokens)
        if not moving:
            return None
        summary = self._summary_keeper.summarize(moving)
        self._choose()
        return summary

    async def compact_async(self, max_tokens: int | None = None) -> Entry | None:
        """``compact``, through the components' async twins."""
        self._check_compact(max_tokens)
        moving = self._to_compact(await self.window_async(), max_tokens)
        if not moving:
            return None
        summary = await self._summary_keeper.summarize_async(moving)
        await self._choose_async()
        return summary

    def update_notes(self) -> Entry:
        """Update the working notes now from the entries since the last update; returns the new notes entry."""
        return self._notes_keeper.update()

    async def update_notes_async(self) -> Entry:
        """``update_notes``, through the components' async twins."""
        return await self._notes_keeper.update_async()

    def set_notes(self, section: str, value: JsonValue) -> Entry:
        """Set one notes section directly, without an LLM call; returns the new notes entry."""
        return self._notes_keeper.set_section(section, value)

    async def set_notes_async(self, section: str, value: JsonValue) -> Entry:
        """``set_notes``, through the components' async twins."""
        return await self._notes_keeper.set_section_async(section, value)

    def rewind(self, entry_id: str) -> None:
        """Take the session back to just after ``entry_id``: what came after no longer counts."""
        index = self.storage.index(self._session_id)
        for later_id in self._after(index, entry_id):
            self.storage.remove(self._session_id, later_id)
        self._resume()

    async def rewind_async(self, entry_id: str) -> None:
        """``rewind``, through the storage's async twins; the state is read back in a worker thread (see ``open_async``)."""
        index = await self.storage.index_async(self._session_id)
        for later_id in self._after(index, entry_id):
            await self.storage.remove_async(self._session_id, later_id)
        await asyncio.to_thread(self._resume)

    def set_policy(self, policy: Policy | None) -> None:
        """Choose the window with ``policy`` from now on; ``None`` keeps every entry."""
        self._policy = self._check_policy(policy)
        self._update_window()

    async def set_policy_async(self, policy: Policy | None) -> None:
        """``set_policy``, through the components' async twins."""
        self._policy = self._check_policy(policy)
        await self._update_window_async()

    def dump(self) -> list[Entry]:
        """Copy the entries not yet dumped to ``dump_storage``, in order; returns them."""
        return self._dump_keeper.dump()

    async def dump_async(self) -> list[Entry]:
        """``dump``, through the components' async twins."""
        return await self._dump_keeper.dump_async()

    # ------------------------------------------------------------------ the window

    def _choose(self) -> Selection:
        """Choose the window again and keep its IDs; returns the whole choice, with what left it."""
        index = self.storage.index(self._session_id)
        pool = self.storage.get(self._session_id, self._pool_ids(index))
        selection = (
            Selection(kept=pool, too_old=[], unwanted=[])
            if self._policy is None
            else self._policy.choose(pool)
        )
        self._window_ids = [entry.id_ for entry in selection.kept]
        return selection

    async def _choose_async(self) -> Selection:
        """``_choose``, through the async twins of the storage and the policy."""
        index = await self.storage.index_async(self._session_id)
        pool = await self.storage.get_async(self._session_id, self._pool_ids(index))
        selection = (
            Selection(kept=pool, too_old=[], unwanted=[])
            if self._policy is None
            else await self._policy.choose_async(pool)
        )
        self._window_ids = [entry.id_ for entry in selection.kept]
        return selection

    def _update_window(self) -> None:
        """Choose the window again, and summarize what it dropped as too old (with a ``summarizer``)."""
        leaving = self._choose().too_old
        if leaving and self.summarizer is not None:
            self._summary_keeper.summarize(leaving)

    async def _update_window_async(self) -> None:
        """``_update_window``, through the components' async twins."""
        leaving = (await self._choose_async()).too_old
        if leaving and self.summarizer is not None:
            await self._summary_keeper.summarize_async(leaving)

    def _pool_ids(self, index: list[tuple[str, EntryKind]]) -> list[str]:
        """The IDs the policy chooses from, in the session's ``index``: the app's entries but system prompts, not pinned or summarized."""
        return [
            entry_id
            for entry_id, kind in index
            if kind not in MANAGED_KINDS
            and kind is not EntryKind.SYSTEM_PROMPT
            and not self._summary_keeper.covers(entry_id)
            and not self._pin_keeper.is_pinned(entry_id)
        ]

    # ------------------------------------------------------------------ shared by both sides

    @staticmethod
    def _check_entry(entry: Entry, pinned: bool) -> None:
        """Raise unless ``entry`` is an ``Entry`` the app may add (not a kind only the manager makes), and may be ``pinned``."""
        if not isinstance(entry, Entry):
            raise TypeError(f"add takes an Entry, not a {type(entry).__name__}")
        if entry.kind in MANAGED_KINDS:
            raise ValueError(
                f"{entry.kind.value} entries come from the manager, not from add()"
            )
        if pinned and entry.kind is EntryKind.SYSTEM_PROMPT:
            raise ValueError(
                "system prompts are kept by rule (the latest one always leads active), so they are not pinned"
            )

    def _forget(
        self, index: list[tuple[str, EntryKind]], removed: Entry, gone: list[str]
    ) -> None:
        """Fix the state after ``removed`` left the session, with the IDs ``gone`` with it (its pin record), placed by the ``index`` from before."""
        self._notes_keeper.forget(removed)
        self._summary_keeper.forget(removed.id_)
        if removed.id_ == self._system_id:  # the one before it is the latest now
            self._system_id = self._latest_system(
                [(stored, kind) for stored, kind in index if stored != removed.id_]
            )
        positions = {stored: i for i, (stored, _) in enumerate(index)}
        self._dump_keeper.forget([positions[stored] for stored in gone])

    @staticmethod
    def _latest_system(index: list[tuple[str, EntryKind]]) -> str | None:
        """The ID of the latest system prompt in ``index``, or ``None`` when there is none."""
        return next(
            (
                stored
                for stored, kind in reversed(index)
                if kind is EntryKind.SYSTEM_PROMPT
            ),
            None,
        )

    @staticmethod
    def _index_of(index: list[tuple[str, EntryKind]], entry_id: str) -> int:
        """Where ``entry_id`` is in ``index``."""
        return next(i for i, (stored, _) in enumerate(index) if stored == entry_id)

    def _check_compact(self, max_tokens: int | None) -> None:
        """Raise without a summarizer, or for a ``max_tokens`` that is not a positive integer."""
        if self.summarizer is None:
            raise ValueError("this manager has no summarizer")
        if max_tokens is not None:
            Checks.positive_int(max_tokens, "max_tokens")

    @staticmethod
    def _to_compact(window: list[Entry], max_tokens: int | None) -> list[Entry]:
        """The oldest ``window`` entries to move into the summary: all of them, or as many as needed to fit ``max_tokens``."""
        start = len(window)  # the window keeps window[start:]
        if max_tokens is not None:
            total = 0
            for entry in reversed(window):
                total += text_tokens(entry)
                if total > max_tokens:
                    break
                start -= 1
        return window[:start]

    def _after(self, index: list[tuple[str, EntryKind]], entry_id: str) -> list[str]:
        """The IDs a rewind to ``entry_id`` removes: everything after it, but the manager's entries right after it."""
        end = self._position(index, entry_id, "rewind") + 1
        while end < len(index) and index[end][1] in MANAGED_KINDS:
            end += 1
        return [later_id for later_id, _ in index[end:]]

    @staticmethod
    def _check_policy(policy: Policy | None) -> Policy | None:
        """``policy``, when it is a ``Policy`` or ``None``; raises ``TypeError`` otherwise."""
        if policy is not None and not isinstance(policy, Policy):
            raise TypeError(f"policy must be a Policy, got {type(policy).__name__}")
        return policy

    # ------------------------------------------------------------------ reads

    @property
    def policy(self) -> Policy | None:
        """The policy choosing the window; ``None`` keeps every entry."""
        return self._policy

    @property
    def active(self) -> list[Entry]:
        """What goes to the model: the latest system prompt, the summary, the notes, the pinned entries, then the window."""
        leading = [
            entry
            for entry in (self.system_prompt, self.summary, self.notes)
            if entry is not None
        ]
        return [*leading, *self.pinned, *self.window]

    async def active_async(self) -> list[Entry]:
        """``active``, through the components' async twins."""
        leading = [
            entry
            for entry in (
                await self.system_prompt_async(),
                await self.summary_async(),
                await self.notes_async(),
            )
            if entry is not None
        ]
        return [*leading, *await self.pinned_async(), *await self.window_async()]

    @property
    def window(self) -> list[Entry]:
        """The entries the policy chose, oldest first; only those entries are read."""
        return self.storage.get(self._session_id, self._window_ids)

    async def window_async(self) -> list[Entry]:
        """``window``, through the storage's async twins."""
        return await self.storage.get_async(self._session_id, self._window_ids)

    @property
    def pinned(self) -> list[Entry]:
        """The pinned entries, oldest first; only those entries are read."""
        return self._pin_keeper.entries()

    async def pinned_async(self) -> list[Entry]:
        """``pinned``, through the storage's async twins."""
        return await self._pin_keeper.entries_async()

    @property
    def system_prompt(self) -> Entry | None:
        """The latest system prompt entry, or ``None`` before the first one."""
        if self._system_id is None:
            return None
        return self.storage.get(self._session_id, [self._system_id])[0]

    async def system_prompt_async(self) -> Entry | None:
        """``system_prompt``, through the storage's async twins."""
        if self._system_id is None:
            return None
        return (await self.storage.get_async(self._session_id, [self._system_id]))[0]

    @property
    def summary(self) -> Entry | None:
        """The current summary entry (the latest made), or ``None`` before the first one."""
        return self._summary_keeper.current

    async def summary_async(self) -> Entry | None:
        """``summary``, through the storage's async twins."""
        return await self._summary_keeper.current_async()

    @property
    def notes(self) -> Entry | None:
        """The current notes entry (the latest version), or ``None`` before the first one."""
        return self._notes_keeper.current

    async def notes_async(self) -> Entry | None:
        """``notes``, through the storage's async twins."""
        return await self._notes_keeper.current_async()

    @property
    def entries(self) -> list[Entry]:
        """The session's history, oldest first: what the app added, without the summary, notes and pin entries."""
        return [
            entry
            for entry in self.storage.entries(self._session_id)
            if entry.kind not in MANAGED_KINDS
        ]

    async def entries_async(self) -> list[Entry]:
        """``entries``, through the storage's async twins."""
        return [
            entry
            for entry in await self.storage.entries_async(self._session_id)
            if entry.kind not in MANAGED_KINDS
        ]
