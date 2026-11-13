"""The component that manages a session's short-term memory."""

import asyncio
from dataclasses import dataclass
from typing import Self
from uuid import uuid4
from pydantic import JsonValue
from .compaction import BaseCompaction
from .entry import Entry, SummaryEntry, WorkingNotesEntry, finishes_turn, is_derived
from .note_takers import BaseNoteTaker, NotesUpdate
from .policies import BasePolicy, FullPolicy
from .storage import BaseStorage
from ...models.schemas import LLMResponse
from ...prompts import SystemPrompt
from ...utility import Checks, Tokens
from ...utility.tokens import DEFAULT_ENCODING


@dataclass(frozen=True)
class SummaryUpdate:
    """A new summary, and the summarizer's response (for its token usage)."""

    summary: SummaryEntry
    response: LLMResponse


class ShortTermMemoryManager:
    """Manages one session's short-term memory, kept in two storages.

    - ``storage``, the **primary**: the actual history. Everything the
      manager reads comes from here; nothing is ever removed from it.
    - ``dump_storage``, the **secondary** (optional): an archive the history
      is dumped into. It only receives entries; the manager never reads the
      history back from it.

    A dump is append-only by ID: ``dump()`` writes only the entries whose ID
    is not in the dump yet, so dumping again never duplicates anything. With
    ``dump_every``, it runs by itself after every ``dump_every`` adds.

    ``active`` is what goes to the model: **the summary, then the working
    notes, then the window**. The manager keeps only IDs, as state; the
    entries themselves live in the primary, which is read for just those
    IDs.

    - **The window** is the policy's choice plus the pinned entries. The
      policy runs on each ``add``, on the current window plus the new entry,
      and chooses among the unpinned ones; pinned entries do not count
      toward its window (a ``SlidingWindowPolicy`` still jumps by ``step``).
    - **Pinned** entries are never removed from the window. Pinning changes
      nothing about the order: a pinned entry stays where it is, and becomes
      the first one once everything older has left. Unpinning removes
      nothing either: the policy decides from the next ``add`` on.
    - **The system prompt:** adding a ``SystemPrompt`` pins it, and unpins the
      previous one after that add, so exactly one system prompt is pinned
      once there is one. The previous one stays until the policy drops it.
      Choosing the window again from the history (a new ``policy``, a
      continued session) considers only the latest system prompt: an old one
      that already left never comes back, and one still in the window
      leaves then.
    - **The summary** stands for the entries that left the window. With a
      ``compaction``, whatever the policy's window drops is summarized in the
      same ``add``: the compaction gets the current summary and only the
      dropped entries, and writes the text of the summary that replaces it.
      So every entry is in the window, in the summary, or excluded by a
      filter, and the policy sets the rhythm (a ``SlidingWindowPolicy``
      gives one summary per jump of ``step`` entries; ``LastNPolicy`` one
      per ``add``). The manager records what each summary stands for
      (everything up to ``covers_through``, except ``covers_except``), so
      nothing is summarized twice. There is one summary at most, always
      first in ``active``; it is never in the window, so the policy cannot
      drop it.
      Old system prompts that leave the window are instructions, not
      conversation: they are not summarized. Entries the policy's
      **filters** exclude (``TypeFilterPolicy``, alone or first in a
      ``ChainPolicy``) are not summarized either: a filter means "not
      wanted", a window means "too old". Without a ``compaction``, what the
      policy drops is forgotten.
    - **Compacting on demand:** ``compact()`` moves the oldest window
      entries into the summary when a caller asks: all of them, or until
      the window fits ``max_tokens``. It is the capability a caller that
      manages the context (a token budget, a ``/compact`` command) uses;
      deciding the budget is that caller's job, not the manager's.
    - **The working notes** are the current state of the task, kept up to
      date by a ``note_taker`` as entries *arrive* (the summary is about
      the past, the notes about the present). Once ``notes_every`` turns
      have finished since the last version (or on ``refresh_notes()``), the
      updater gets the current notes and only the entries added since, and
      returns the next version. A turn finishes with the model's answer: an
      ``LLMResponse`` with no tool calls (one that calls a tool is the middle
      of a turn). So an update never catches a question without its answer;
      the notes can still lag up to ``notes_every - 1`` finished turns
      behind. An app that never adds the model's answers finishes no turns:
      its notes update only on ``refresh_notes()``.
      There is one version at most in ``active``, right after the summary;
      it is never in the window, so the policy cannot drop it.

    The window is the policy's choice plus the pinned entries: a cut can
    leave an answer without its question, or a tool result without its call.

    **Async:** every operation has an ``_async`` twin (``add_async``,
    ``compact_async``, ``refresh_notes_async``, ``dump_async``,
    ``set_policy_async``), and every read an ``_async`` method
    (``active_async()``, ``window_async()``, ...), using the storages',
    compaction's and note taker's async methods. A manager for async code is
    made with ``await ShortTermMemoryManager.open_async(...)``, since reading
    a continued session (and maybe summarizing it) needs I/O. A lock runs one
    async operation at a time, so concurrent ``add_async`` calls do not
    interleave; it does not guard the sync methods, so do not call those
    while async ones are running on the same manager.
    """

    def __init__(
        self,
        storage: BaseStorage,
        dump_storage: BaseStorage | None = None,
        dump_every: int | None = None,
        policy: BasePolicy | None = None,
        compaction: BaseCompaction | None = None,
        note_taker: BaseNoteTaker | None = None,
        notes_every: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """Manage a session kept in ``storage``, dumped into ``dump_storage`` if one is given.

        ``session_id`` names the session in both storages, which may serve
        other sessions too. ``None`` starts a new session with a generated
        UUID; the ID of an earlier session continues it. Raises
        ``TypeError`` or ``ValueError`` if it is given but not a non-empty
        string.

        ``dump_every`` dumps by itself after every ``dump_every`` adds
        (summaries and notes are not adds, but each dump includes them);
        ``None`` means only ``dump()`` does.

        ``policy`` chooses the window; ``None`` means ``FullPolicy()``, so
        everything stays in it and nothing is ever summarized.

        ``compaction`` summarizes whatever the window leaves out, as it
        leaves; ``None`` means what leaves is forgotten. What the policy's
        filters exclude is never summarized. Each time entries leave is one
        compaction call, so pick a policy that drops them in batches:
        ``LastNPolicy`` drops one entry per add once full, which means an
        LLM call on every add and a new summary that breaks the provider's
        prompt cache each time; ``SlidingWindowPolicy`` with a ``step`` of
        several entries summarizes once every ``step`` adds.

        ``note_taker`` writes the working notes: automatically once
        ``notes_every`` turns have finished since the last version, or
        only through ``refresh_notes()`` when ``notes_every`` is ``None``.
        Raises ``ValueError`` if ``dump_every`` is given without a
        ``dump_storage``, or ``notes_every`` without a ``note_taker``, or
        if either is not a positive integer.

        For a continued session, the stored entries are read once, here: the
        latest system prompt is pinned again (pins live only in memory), the
        latest summary is the current one, and the window is chosen from the
        entries it does not cover, considering only the latest system prompt
        (an older one is not restored, even if it was still in the window).
        With a ``compaction``, any of those the
        window leaves out are summarized here (an LLM call), which happens
        only if they were never summarized (a session that had no
        compaction, or a compaction that failed). The latest notes are the
        current ones, and the entries stored after them are the ones since.
        """
        self._configure(
            storage,
            dump_storage,
            dump_every,
            policy,
            compaction,
            note_taker,
            notes_every,
            session_id,
        )
        stored = self.storage.entries(self.session_id)
        self._restore(stored)
        self._settle(self._rechoose_pool(stored))

    @classmethod
    async def open_async(
        cls,
        storage: BaseStorage,
        dump_storage: BaseStorage | None = None,
        dump_every: int | None = None,
        policy: BasePolicy | None = None,
        compaction: BaseCompaction | None = None,
        note_taker: BaseNoteTaker | None = None,
        notes_every: int | None = None,
        session_id: str | None = None,
    ) -> Self:
        """The manager ``__init__`` makes, built with async I/O; the same arguments and errors."""
        manager = cls.__new__(cls)
        manager._configure(
            storage,
            dump_storage,
            dump_every,
            policy,
            compaction,
            note_taker,
            notes_every,
            session_id,
        )
        stored = await manager.storage.entries_async(manager.session_id)
        manager._restore(stored)
        await manager._settle_async(manager._rechoose_pool(stored))
        return manager

    def _configure(
        self,
        storage: BaseStorage,
        dump_storage: BaseStorage | None,
        dump_every: int | None,
        policy: BasePolicy | None,
        compaction: BaseCompaction | None,
        note_taker: BaseNoteTaker | None,
        notes_every: int | None,
        session_id: str | None,
    ) -> None:
        """Check the settings and keep them; no I/O."""
        if dump_storage is None and dump_every is not None:
            raise ValueError("dump_every needs a dump_storage")
        if note_taker is None and notes_every is not None:
            raise ValueError("notes_every needs a note_taker")
        for name, value in (("dump_every", dump_every), ("notes_every", notes_every)):
            if value is not None:
                Checks.positive_int(value, name)
        self.session_id: str = (
            str(uuid4())
            if session_id is None
            else Checks.non_empty_string(session_id, "session_id")
        )
        self.storage = storage
        self.dump_storage = dump_storage
        self.dump_every = dump_every
        self.compaction = compaction
        self.note_taker = note_taker
        self.notes_every = notes_every
        self._policy: BasePolicy = policy if policy is not None else FullPolicy()
        # One async operation at a time.
        self._lock = asyncio.Lock()

    def _restore(self, stored: list[Entry]) -> None:
        """Set the state from the session's ``stored`` entries; no I/O."""
        # Adds since the last dump: what dump_every counts.
        self._adds_since_dump: int = 0
        # IDs in the primary storage, so a duplicate is caught without reading it.
        self._ids: set[str] = {entry.id_ for entry in stored}
        # IDs of the original entries (not summaries or notes), in history
        # order, and where each one is: what a summary's coverage is read against.
        self._originals: list[str] = [e.id_ for e in stored if not is_derived(e)]
        self._position: dict[str, int] = {
            entry_id: i for i, entry_id in enumerate(self._originals)
        }
        # IDs of every system prompt, so pin() can refuse them without reading them.
        self._system_ids: set[str] = {
            e.id_ for e in stored if isinstance(e, SystemPrompt)
        }
        # The pinned system prompt: the latest one added.
        self._system_id: str | None = self._latest_id(stored, SystemPrompt)
        # The current summary: the latest one made. Not in the window; always first in active.
        self._summary_id: str | None = self._latest_id(stored, SummaryEntry)
        # The current notes: the latest version. Not in the window; right after the summary in active.
        self._notes_id: str | None = self._latest_id(stored, WorkingNotesEntry)
        # IDs added since the current notes version, in order: what the next update gets.
        last_notes = max(
            (i for i, e in enumerate(stored) if isinstance(e, WorkingNotesEntry)),
            default=-1,
        )
        since = [entry for entry in stored[last_notes + 1 :] if not is_derived(entry)]
        self._since_notes: list[str] = [entry.id_ for entry in since]
        # How many turns those finish: what notes_every counts.
        self._turns_since_notes: int = sum(finishes_turn(entry) for entry in since)
        # IDs that the policy can never remove from the window; always in the window.
        self._pinned: set[str] = (
            {self._system_id} if self._system_id is not None else set()
        )
        # IDs of the window, in history order; the entries stay in the primary.
        self._window: list[str] = []

    # ------------------------------------------------------------------ the policy

    @property
    def policy(self) -> BasePolicy:
        """Chooses the window."""
        return self._policy

    @policy.setter
    def policy(self, policy: BasePolicy) -> None:
        """Replace the policy; the window is chosen again from the whole history, once.

        Entries the summary covers never come back. Only the latest system
        prompt is considered: an older one still in the window leaves now,
        not summarized. With a ``compaction``, entries the new window leaves
        out are summarized now. Pins are kept.
        """
        self._policy = policy
        self._settle(self._rechoose_pool(self.entries))

    async def set_policy_async(self, policy: BasePolicy) -> None:
        """Setting ``policy``, with async I/O."""
        async with self._lock:
            self._policy = policy
            await self._settle_async(self._rechoose_pool(await self.entries_async()))

    # ------------------------------------------------------------------ add

    def add(self, entry: Entry) -> None:
        """Append ``entry``, update the window, summarize what left it, and update the notes if due.

        1. The policy chooses among the unpinned entries of the current
           window plus the new one; the pinned ones stay. A ``SystemPrompt``
           is pinned as it is added, and the previous system prompt is
           unpinned after the window is updated, so this add does not remove it.
        2. With a ``compaction``, the entries that left the window (system
           prompts aside) are folded into the summary now. If that fails, the
           error is raised and nothing leaves the window this time: the next
           ``add`` drops them again and retries, so nothing is lost.
        3. When ``notes_every`` is set and that many turns have finished
           since the last notes version, calls ``refresh_notes()``. If that
           fails, the error is raised and the entries stay "since the last
           version", so the next update gets them.
        4. When ``dump_every`` is set and that many adds were made since the
           last dump, calls ``dump()``. Last, so the dump includes the new
           summary and notes.

        Raises ``ValueError`` if an entry with the same ``id_`` is already
        there (IDs must be unique, since the dump tells entries apart by
        ID), or for a ``SummaryEntry`` or ``WorkingNotesEntry``: they come
        from the compaction and the note taker.
        """
        self._check_new(entry)
        self.storage.add(self.session_id, entry)
        previous_system = self._record(entry)
        try:
            self._settle([*self.storage.get(self.session_id, self._window), entry])
        finally:
            self._unpin_previous(previous_system)
        if self._notes_due:
            self.refresh_notes()
        if self._dump_due:
            self.dump()

    async def add_async(self, entry: Entry) -> None:
        """``add``, with async I/O."""
        async with self._lock:
            self._check_new(entry)
            await self.storage.add_async(self.session_id, entry)
            previous_system = self._record(entry)
            try:
                window = await self.storage.get_async(self.session_id, self._window)
                await self._settle_async([*window, entry])
            finally:
                self._unpin_previous(previous_system)
            if self._notes_due:
                await self._refresh_notes_async()
            if self._dump_due:
                await self._dump_async()

    def _check_new(self, entry: Entry) -> None:
        """Raise ``ValueError`` for an entry ``add`` refuses."""
        if isinstance(entry, SummaryEntry):
            raise ValueError("summaries come from compaction, not add()")
        if isinstance(entry, WorkingNotesEntry):
            raise ValueError("working notes come from the note taker, not add()")
        if entry.id_ in self._ids:
            raise ValueError(
                f"an entry with id_ {entry.id_!r} is already in the session"
            )

    def _record(self, entry: Entry) -> str | None:
        """Count the stored ``entry`` in; for a system prompt, pin it and return the previous one's ID."""
        self._ids.add(entry.id_)
        self._position[entry.id_] = len(self._originals)
        self._originals.append(entry.id_)
        self._since_notes.append(entry.id_)
        self._turns_since_notes += finishes_turn(entry)
        self._adds_since_dump += 1
        if not isinstance(entry, SystemPrompt):
            return None
        previous, self._system_id = self._system_id, entry.id_
        self._system_ids.add(entry.id_)
        self._pinned.add(entry.id_)
        return previous

    def _unpin_previous(self, previous_system: str | None) -> None:
        """Unpin the replaced system prompt; it stays in the window, and the policy may drop it from the next add on."""
        if previous_system is not None:
            self._pinned.discard(previous_system)

    @property
    def _notes_due(self) -> bool:
        """Whether ``notes_every`` turns have finished since the last notes version."""
        return (
            self.notes_every is not None and self._turns_since_notes >= self.notes_every
        )

    @property
    def _dump_due(self) -> bool:
        """Whether ``dump_every`` adds were made since the last dump."""
        return self.dump_every is not None and self._adds_since_dump >= self.dump_every

    # ------------------------------------------------------------------ compact

    def compact(
        self, max_tokens: int | None = None, encoding: str = DEFAULT_ENCODING
    ) -> SummaryUpdate | None:
        """Move the oldest window entries into the summary now: all of them, or until the window fits ``max_tokens``.

        Counting from the newest, the window keeps its entries while they
        fit and stops at the first that does not; everything older in it
        goes to the compaction in one call, and the new summary covers it.
        Pinned entries count toward ``max_tokens`` but always stay.
        ``max_tokens=None`` keeps only the pinned entries. An old system
        prompt that leaves is not summarized, as on ``add``.

        Only the window is counted, not the summary or the notes. Tokens are
        counted on each entry's ``to_text()`` with ``encoding``
        (``Tokens.count``): close, not exact, for other providers' models,
        and media count only as their placeholder.

        A one-time squeeze: on the next adds, the policy chooses from the
        smaller window, so it can grow again. What was summarized never
        comes back.

        When calling it on a token limit, squeeze to well below that limit,
        e.g. ``compact(max_tokens=limit // 2)`` once the window passes
        ``limit``. Squeezing to the limit itself leaves the window just
        under it, so the next add passes it again and every add compacts
        (an LLM call each time).

        Returns the compaction's ``SummaryUpdate`` (with its token usage), or
        ``None`` when nothing was summarized (the window already fits).
        Raises ``ValueError``, changing nothing, if there is no
        ``compaction``, ``max_tokens`` is not a non-negative integer,
        ``encoding`` is unknown, or the pinned entries alone take more than
        ``max_tokens``. If the compaction fails, its error is raised and the
        window stays as it was.
        """
        self._check_compact(max_tokens, encoding)
        window = self.window
        kept, dropped = self._squeeze(window, max_tokens, encoding)
        update = self._summarize(dropped) if dropped else None
        self._window = [entry.id_ for entry in window if entry.id_ in kept]
        return update

    async def compact_async(
        self, max_tokens: int | None = None, encoding: str = DEFAULT_ENCODING
    ) -> SummaryUpdate | None:
        """``compact``, with async I/O."""
        async with self._lock:
            self._check_compact(max_tokens, encoding)
            window = await self.window_async()
            kept, dropped = self._squeeze(window, max_tokens, encoding)
            update = await self._summarize_async(dropped) if dropped else None
            self._window = [entry.id_ for entry in window if entry.id_ in kept]
            return update

    def _check_compact(self, max_tokens: int | None, encoding: str) -> None:
        """Raise ``ValueError`` for a ``compact`` that cannot run."""
        if self.compaction is None:
            raise ValueError("this manager has no compaction")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 0
        ):
            raise ValueError(
                f"max_tokens must be a non-negative integer or None, got {max_tokens!r}"
            )
        Tokens.require_encoding(encoding)

    def _squeeze(
        self, window: list[Entry], max_tokens: int | None, encoding: str
    ) -> tuple[set[str], list[Entry]]:
        """The IDs ``compact`` keeps in ``window``, and the entries it summarizes."""
        kept = {entry.id_ for entry in window if entry.id_ in self._pinned}
        if max_tokens is not None:
            used = sum(
                Tokens.count(entry.to_text(), encoding)
                for entry in window
                if entry.id_ in kept
            )
            if used > max_tokens:
                raise ValueError(
                    f"the pinned entries alone take {used} tokens, more than max_tokens={max_tokens}; unpin some first"
                )
            for entry in reversed(window):
                if entry.id_ in kept:
                    continue
                used += Tokens.count(entry.to_text(), encoding)
                if used > max_tokens:
                    break
                kept.add(entry.id_)
        dropped: list[Entry] = [
            e for e in window if e.id_ not in kept and not isinstance(e, SystemPrompt)
        ]
        return kept, dropped

    # ------------------------------------------------------------------ notes

    def refresh_notes(self) -> NotesUpdate:
        """Update the working notes now, however few entries arrived since the last version.

        Hands the ``note_taker`` the current notes and only the entries
        added since the last version. The new version is stored in the
        primary and becomes the current one. Returns the note taker's
        ``NotesUpdate``. Raises ``ValueError`` if there is no
        ``note_taker``, nothing new, or an answer that does not fit the
        note taker's schema (the notes then stay as they were).
        """
        note_taker = self._require_note_taker()
        update = note_taker.update(
            self.notes, self.storage.get(self.session_id, self._since_notes)
        )
        entry = WorkingNotesEntry(notes=update.notes)
        self.storage.add(self.session_id, entry)
        self._install_notes(entry)
        return update

    async def refresh_notes_async(self) -> NotesUpdate:
        """``refresh_notes``, with async I/O."""
        async with self._lock:
            return await self._refresh_notes_async()

    async def _refresh_notes_async(self) -> NotesUpdate:
        """``refresh_notes_async`` without taking the lock, for ``add_async``, which holds it."""
        note_taker = self._require_note_taker()
        since = await self.storage.get_async(self.session_id, self._since_notes)
        update = await note_taker.update_async(await self.notes_async(), since)
        entry = WorkingNotesEntry(notes=update.notes)
        await self.storage.add_async(self.session_id, entry)
        self._install_notes(entry)
        return update

    def _require_note_taker(self) -> BaseNoteTaker:
        """The ``note_taker``; raises ``ValueError`` if there is none."""
        if self.note_taker is None:
            raise ValueError("this manager has no note_taker")
        return self.note_taker

    def _install_notes(self, entry: WorkingNotesEntry) -> None:
        """Make the stored ``entry`` the current notes; nothing is "since" them yet."""
        self._ids.add(entry.id_)
        self._notes_id = entry.id_
        self._since_notes = []
        self._turns_since_notes = 0

    # ------------------------------------------------------------------ dump

    def dump(self) -> list[Entry]:
        """Append the entries not yet dumped to ``dump_storage``, in order, and return them.

        The primary storage is left as it is; the count toward ``dump_every``
        starts again. Raises ``ValueError`` if there is no ``dump_storage``.
        """
        dump_storage = self._require_dump_storage()
        dumped = {entry.id_ for entry in dump_storage.entries(self.session_id)}
        new = [entry for entry in self.entries if entry.id_ not in dumped]
        for entry in new:
            dump_storage.add(self.session_id, entry)
        self._adds_since_dump = 0
        return new

    async def dump_async(self) -> list[Entry]:
        """``dump``, with async I/O."""
        async with self._lock:
            return await self._dump_async()

    async def _dump_async(self) -> list[Entry]:
        """``dump_async`` without taking the lock, for ``add_async``, which holds it."""
        dump_storage = self._require_dump_storage()
        dumped = {
            entry.id_ for entry in await dump_storage.entries_async(self.session_id)
        }
        new = [entry for entry in await self.entries_async() if entry.id_ not in dumped]
        for entry in new:  # one after another, so the dump keeps the order
            await dump_storage.add_async(self.session_id, entry)
        self._adds_since_dump = 0
        return new

    def _require_dump_storage(self) -> BaseStorage:
        """The ``dump_storage``; raises ``ValueError`` if there is none."""
        if self.dump_storage is None:
            raise ValueError("this manager has no dump_storage")
        return self.dump_storage

    # ------------------------------------------------------------------ pins

    def pin(self, entry_id: str) -> None:
        """Keep the entry with ``entry_id`` in the window, whatever the policy chooses.

        Its place in the window does not change. Pinning a pinned entry does
        nothing. Raises ``ValueError`` if the entry is not in the window, or
        is a ``SystemPrompt``, ``SummaryEntry`` or ``WorkingNotesEntry``
        (the latest system prompt is pinned by rule, and the summary and the
        notes always lead ``active``). Needs no I/O, so it has no async twin.
        """
        Checks.non_empty_string(entry_id, "entry_id")
        if entry_id in (self._summary_id, self._notes_id):
            raise ValueError(
                "the summary and the notes are pinned by rule: they always lead active"
            )
        if entry_id not in self._window:
            raise ValueError(
                f"entry {entry_id!r} is not in the window; only an active entry can be pinned"
            )
        if entry_id in self._system_ids:
            raise ValueError(
                "system prompts are pinned by rule: the latest one added, and only it"
            )
        self._pinned.add(entry_id)

    def unpin(self, entry_id: str) -> None:
        """Let the policy decide again about the entry with ``entry_id``, from the next ``add`` on.

        It stays in the window until then. Unpinning an entry that is not
        pinned does nothing. Raises ``ValueError`` for the pinned system
        prompt, the summary or the notes, which only a new one replaces.
        """
        Checks.non_empty_string(entry_id, "entry_id")
        if entry_id in (self._system_id, self._summary_id, self._notes_id):
            raise ValueError(
                "the latest system prompt, the summary and the notes stay pinned; only a new one replaces them"
            )
        self._pinned.discard(entry_id)

    # ------------------------------------------------------------------ reads

    @property
    def active(self) -> list[Entry]:
        """What goes to the model: the summary (if any), the notes (if any), then the window.

        Read from the primary by ID (``storage.get``), so only those entries
        are loaded where the storage allows it (SQLite).
        """
        return self.storage.get(self.session_id, self._leading(self._window))

    async def active_async(self) -> list[Entry]:
        """``active``, with async I/O."""
        return await self.storage.get_async(
            self.session_id, self._leading(self._window)
        )

    @property
    def window(self) -> list[Entry]:
        """The window alone, in history order: the policy's choice plus the pinned entries.

        That is ``active`` without the summary and the notes, for a caller
        that lays out the context itself or measures it.
        """
        return self.storage.get(self.session_id, self._window)

    async def window_async(self) -> list[Entry]:
        """``window``, with async I/O."""
        return await self.storage.get_async(self.session_id, self._window)

    @property
    def summary(self) -> SummaryEntry | None:
        """The current summary, or ``None`` before the first compaction."""
        if self._summary_id is None:
            return None
        return self._as(
            self.storage.get(self.session_id, [self._summary_id])[0], SummaryEntry
        )

    async def summary_async(self) -> SummaryEntry | None:
        """``summary``, with async I/O."""
        if self._summary_id is None:
            return None
        entries = await self.storage.get_async(self.session_id, [self._summary_id])
        return self._as(entries[0], SummaryEntry)

    @property
    def notes(self) -> dict[str, JsonValue] | None:
        """The current working notes (one key per section), or ``None`` before the first version."""
        if self._notes_id is None:
            return None
        return self._as(
            self.storage.get(self.session_id, [self._notes_id])[0], WorkingNotesEntry
        ).notes

    async def notes_async(self) -> dict[str, JsonValue] | None:
        """``notes``, with async I/O."""
        if self._notes_id is None:
            return None
        entries = await self.storage.get_async(self.session_id, [self._notes_id])
        return self._as(entries[0], WorkingNotesEntry).notes

    @property
    def pinned(self) -> list[Entry]:
        """The pinned entries, in ``active`` order: the summary, the notes, then the window's pinned entries."""
        return self.storage.get(self.session_id, self._pinned_ids())

    async def pinned_async(self) -> list[Entry]:
        """``pinned``, with async I/O."""
        return await self.storage.get_async(self.session_id, self._pinned_ids())

    @property
    def entries(self) -> list[Entry]:
        """The session's entries in the primary storage, oldest first."""
        return self.storage.entries(self.session_id)

    async def entries_async(self) -> list[Entry]:
        """``entries``, with async I/O."""
        return await self.storage.entries_async(self.session_id)

    def _leading(self, ids: list[str]) -> list[str]:
        """``ids`` after the current summary's and notes' IDs, those that exist."""
        return [
            entry_id
            for entry_id in (self._summary_id, self._notes_id)
            if entry_id is not None
        ] + ids

    def _pinned_ids(self) -> list[str]:
        """The IDs of ``pinned``: the summary's, the notes', then the window's pinned entries."""
        return self._leading(
            [entry_id for entry_id in self._window if entry_id in self._pinned]
        )

    @staticmethod
    def _as[T](entry: Entry, kind: type[T]) -> T:
        """``entry``, checked to be a ``kind``."""
        assert isinstance(entry, kind)
        return entry

    # ------------------------------------------------------------------ the window

    def _rechoose_pool(self, entries: list[Entry]) -> list[Entry]:
        """What the window is chosen from again: the history, minus what the summary covers and old system prompts."""
        summary = next(
            (entry for entry in entries if entry.id_ == self._summary_id), None
        )
        covered = (
            set(summary.covered([e.id_ for e in entries if not is_derived(e)]))
            if isinstance(summary, SummaryEntry)
            else set()
        )
        replaced = {
            e.id_
            for e in entries
            if isinstance(e, SystemPrompt) and e.id_ != self._system_id
        }
        return [
            e for e in entries if e.id_ not in covered | replaced and not is_derived(e)
        ]

    def _settle(self, pool: list[Entry]) -> None:
        """Make the window the choice from ``pool`` (in order), and summarize what the window leaves out.

        What the policy's filters exclude is forgotten, never summarized;
        so is everything without a ``compaction``. If the compaction fails,
        the window keeps all of ``pool`` and the error is raised: nothing
        leaves the window before it is in the summary.
        """
        window, dropped = self._plan(pool)
        if self.compaction is not None and dropped:
            try:
                self._summarize(dropped)
            except Exception:
                self._window = [entry.id_ for entry in pool]
                raise
        self._window = window

    async def _settle_async(self, pool: list[Entry]) -> None:
        """``_settle``, with async I/O."""
        window, dropped = self._plan(pool)
        if self.compaction is not None and dropped:
            try:
                await self._summarize_async(dropped)
            except Exception:
                self._window = [entry.id_ for entry in pool]
                raise
        self._window = window

    def _plan(self, pool: list[Entry]) -> tuple[list[str], list[Entry]]:
        """The window chosen from ``pool``, and the entries it drops that belong in the summary."""
        window = self._choose(pool)
        kept = set(window)
        candidates = [entry for entry in pool if entry.id_ not in self._pinned]
        wanted = {entry.id_ for entry in self._policy.filter(candidates)}
        dropped: list[Entry] = [
            e
            for e in pool
            if e.id_ not in kept and e.id_ in wanted and not isinstance(e, SystemPrompt)
        ]
        return window, dropped

    def _choose(self, pool: list[Entry]) -> list[str]:
        """The window from ``pool`` (in order): the pinned entries, plus the policy's choice from the rest."""
        chosen = {
            entry.id_
            for entry in self._policy.select(
                [e for e in pool if e.id_ not in self._pinned]
            )
        }
        return [
            entry.id_
            for entry in pool
            if entry.id_ in self._pinned or entry.id_ in chosen
        ]

    # ------------------------------------------------------------------ the summary

    def _summarize(self, dropped: list[Entry]) -> SummaryUpdate:
        """Fold ``dropped`` into the summary: the compaction gets the current summary and only these entries.

        The new summary is stored in the primary and becomes the current
        one. The compaction also gets the summary's original entries, if it
        uses them (``RebuildCompaction``).
        """
        assert self.compaction is not None
        previous = self.summary
        covered = (
            self.storage.get(self.session_id, previous.covered(self._originals))
            if previous is not None and self.compaction.uses_covered
            else []
        )
        response = self.compaction.compact(dropped, previous, covered)
        summary = self._next_summary(response, previous, dropped)
        self.storage.add(self.session_id, summary)
        self._install_summary(summary)
        return SummaryUpdate(summary, response)

    async def _summarize_async(self, dropped: list[Entry]) -> SummaryUpdate:
        """``_summarize``, with async I/O."""
        assert self.compaction is not None
        previous = await self.summary_async()
        covered = (
            await self.storage.get_async(
                self.session_id, previous.covered(self._originals)
            )
            if previous is not None and self.compaction.uses_covered
            else []
        )
        response = await self.compaction.compact_async(dropped, previous, covered)
        summary = self._next_summary(response, previous, dropped)
        await self.storage.add_async(self.session_id, summary)
        self._install_summary(summary)
        return SummaryUpdate(summary, response)

    def _next_summary(
        self, response: LLMResponse, previous: SummaryEntry | None, dropped: list[Entry]
    ) -> SummaryEntry:
        """The summary with ``response``'s text, covering what ``previous`` covered plus ``dropped``.

        It covers up to the newest of them; the original entries before that
        it does not cover (pinned, filtered, system prompts, still in the
        window) are its exceptions. Only the entries after ``previous``'s
        last one are looked at, so this does not grow with the history.
        """
        new = {entry.id_ for entry in dropped}
        if previous is not None and previous.covers_through in self._position:
            start = self._position[previous.covers_through] + 1
            skipped = [i for i in previous.covers_except if i not in new]
        else:  # the first summary, or the previous one's end was removed: start over
            start, skipped = 0, []
        end = max(start, *(self._position[entry_id] + 1 for entry_id in new))
        skipped += [i for i in self._originals[start:end] if i not in new]
        count = (previous.covers_count if previous is not None else 0) + len(new)
        return SummaryEntry(
            text=response.text,
            covers_through=self._originals[end - 1],
            covers_except=skipped,
            covers_count=count,
        )

    def _install_summary(self, summary: SummaryEntry) -> None:
        """Make the stored ``summary`` the current one."""
        self._ids.add(summary.id_)
        self._summary_id = summary.id_

    @staticmethod
    def _latest_id(entries: list[Entry], kind: type) -> str | None:
        """The ID of the latest entry of type ``kind`` in ``entries``, or ``None``."""
        return next(
            (entry.id_ for entry in reversed(entries) if isinstance(entry, kind)), None
        )
