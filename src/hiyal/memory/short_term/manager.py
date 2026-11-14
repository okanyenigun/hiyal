"""The component that manages a session's short-term memory."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, Self
from uuid import uuid4
from pydantic import JsonValue, ValidationError, create_model
from .compaction import BaseCompaction
from .entry import (
    Entry,
    RewindEntry,
    SummaryEntry,
    WorkingNotesEntry,
    finishes_turn,
    MEDIA_TOKENS,
    check_media_tokens,
    entry_tokens,
)
from .note_takers import BaseNoteTaker, NotesUpdate
from .policies import BasePolicy, FullPolicy
from .storage import BaseStorage
from ...models.schemas import LLMResponse
from ...prompts import SystemPrompt
from ...utility import Checks, Tokens
from ...utility.tokens import DEFAULT_ENCODING


class _Row(NamedTuple):
    """One stored entry as the storage's index lists it: its ID, its class name, and a rewind marker's target."""

    id_: str
    type: str
    to_id: str | None = None


# Class names of the entries made by workers, never added to the conversation.
_DERIVED = {SummaryEntry.__name__, WorkingNotesEntry.__name__}


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
      its notes update only on ``refresh_notes()``. The app can also set
      one section directly with ``set_notes()``, without an LLM call: a new
      version too, which leaves the entries since the last update queued.
      There is one version at most in ``active``, right after the summary;
      it is never in the window, so the policy cannot drop it.
    - **Rewinding:** ``rewind(entry_id)`` takes the session back to just
      after an entry, as it was before the next one was added; what came
      after no longer counts. The storage stays append-only: a
      ``RewindEntry`` marks it, and reading the history applies it.

    The window is the policy's choice plus the pinned entries: a cut can
    leave an answer without its question, or a tool result without its call.

    **Async:** every operation has an ``_async`` twin (``add_async``,
    ``compact_async``, ``refresh_notes_async``, ``set_notes_async``,
    ``rewind_async``, ``dump_async``, ``set_policy_async``), and every read an ``_async`` method
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

        For a continued session, the state is read back here: the storage's
        index of the session first (IDs and types, see
        ``BaseStorage.index``), then only the entries it needs, not the
        whole history. The latest system prompt is pinned again (pins live
        only in memory), the latest summary is the current one, and the
        window is chosen from the entries it does not cover, considering
        only the latest system prompt (an older one is not restored, even if
        it was still in the window). With a ``compaction``, any of those the
        window leaves out are summarized here (an LLM call), which happens
        only if they were never summarized (a session that had no
        compaction, or a compaction that failed). The latest notes are the
        current ones, and the entries stored after the note taker's last
        version are the ones since.
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
        self._read_state()
        self._settle(self._pool())

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
        await manager._read_state_async()
        await manager._settle_async(await manager._pool_async())
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

    def _read_state(self) -> None:
        """Set the state from the storage: its index of the session, then only the entries needed; the window is left to ``_settle``."""
        rows = self.storage.index(self.session_id)
        markers = self.storage.get(self.session_id, self._marker_ids(rows))
        history = self._restore(rows, markers)
        start = None
        for notes_id in self._notes_ids(history):
            (version,) = self.storage.get(self.session_id, [notes_id])
            if self._as(version, WorkingNotesEntry).source == "note_taker":
                start = notes_id
                break
        since = self._queue_since(history, start)
        if self.notes_every is not None:
            self._turns_since_notes = sum(
                finishes_turn(e) for e in self.storage.get(self.session_id, since)
            )

    async def _read_state_async(self) -> None:
        """``_read_state``, with async I/O."""
        rows = await self.storage.index_async(self.session_id)
        markers = await self.storage.get_async(self.session_id, self._marker_ids(rows))
        history = self._restore(rows, markers)
        start = None
        for notes_id in self._notes_ids(history):
            (version,) = await self.storage.get_async(self.session_id, [notes_id])
            if self._as(version, WorkingNotesEntry).source == "note_taker":
                start = notes_id
                break
        since = self._queue_since(history, start)
        if self.notes_every is not None:
            entries = await self.storage.get_async(self.session_id, since)
            self._turns_since_notes = sum(finishes_turn(e) for e in entries)

    @staticmethod
    def _marker_ids(rows: list[tuple[str, str]]) -> list[str]:
        """The IDs of the rewind markers among the index's ``rows``: the only entries the history's shape needs."""
        return [entry_id for entry_id, name in rows if name == RewindEntry.__name__]

    def _restore(self, rows: list[tuple[str, str]], markers: list[Entry]) -> list[_Row]:
        """Set what the index gives (``rows``, with the rewind ``markers``' targets), and return the history; no I/O.

        The history is the rows with the rewinds applied. The note taker's
        queue is set by ``_queue_since``; until then it is empty.
        """
        # Adds since the last dump: what dump_every counts.
        self._adds_since_dump: int = 0
        # IDs in the primary storage, so a duplicate is caught without reading it;
        # all of them, also the ones a rewind dropped, which stay stored.
        self._ids: set[str] = {entry_id for entry_id, _ in rows}
        targets = {m.id_: self._as(m, RewindEntry).to_id for m in markers}
        history = self._effective(
            [_Row(entry_id, name, targets.get(entry_id)) for entry_id, name in rows]
        )
        # IDs of the original entries (not summaries or notes), in history
        # order, and where each one is: what a summary's coverage is read against.
        self._originals: list[str] = [r.id_ for r in history if r.type not in _DERIVED]
        self._position: dict[str, int] = {
            entry_id: i for i, entry_id in enumerate(self._originals)
        }
        # IDs of every system prompt, so pin() can refuse them without reading them.
        self._system_ids: set[str] = {
            r.id_ for r in history if r.type == SystemPrompt.__name__
        }
        # The pinned system prompt: the latest one added.
        self._system_id: str | None = self._latest_id(history, SystemPrompt)
        # The current summary: the latest one made. Not in the window; always first in active.
        self._summary_id: str | None = self._latest_id(history, SummaryEntry)
        # The current notes: the latest version. Not in the window; right after the summary in active.
        self._notes_id: str | None = self._latest_id(history, WorkingNotesEntry)
        # IDs added since the note taker's last version, in order: what its next
        # update gets (a version the app set with set_notes does not count).
        self._since_notes: list[str] = []
        # How many turns those finish: what notes_every counts (only with it).
        self._turns_since_notes: int = 0
        # IDs that the policy can never remove from the window; always in the window.
        self._pinned: set[str] = (
            {self._system_id} if self._system_id is not None else set()
        )
        # IDs of the window, in history order; the entries stay in the primary.
        self._window: list[str] = []
        return history

    @staticmethod
    def _notes_ids(history: list[_Row]) -> list[str]:
        """The notes versions in ``history``, newest first: where to look for the note taker's last one."""
        return [r.id_ for r in reversed(history) if r.type == WorkingNotesEntry.__name__]

    def _queue_since(self, history: list[_Row], notes_id: str | None) -> list[str]:
        """Set the note taker's queue: the original entries after its last version ``notes_id`` (all without one)."""
        start = next((i + 1 for i, r in enumerate(history) if r.id_ == notes_id), 0)
        self._since_notes = [r.id_ for r in history[start:] if r.type not in _DERIVED]
        return self._since_notes

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
        self._settle(self._pool())

    async def set_policy_async(self, policy: BasePolicy) -> None:
        """Setting ``policy``, with async I/O."""
        async with self._lock:
            self._policy = policy
            await self._settle_async(await self._pool_async())

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

        Adding an entry that is already there again, with the same content
        and still in the history, does nothing: it is a repeat (a retried
        delivery), so an app can retry an ``add`` it is unsure about.
        Content is compared as stored, so ``LLMResponse.raw`` does not count.

        Raises ``ValueError`` if another entry with the same ``id_`` is
        already there, or one a rewind dropped (IDs must be unique, since
        the dump tells entries apart by ID), or for a ``SummaryEntry``,
        ``WorkingNotesEntry`` or ``RewindEntry``: they come from the
        compaction, the notes and ``rewind``.
        """
        if self._repeats(entry, self._stored_copy(entry)):
            return
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
            stored = (
                (await self.storage.get_async(self.session_id, [entry.id_]))[0]
                if entry.id_ in self._position
                else None
            )
            if self._repeats(entry, stored):
                return
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

    def _stored_copy(self, entry: Entry) -> Entry | None:
        """The stored entry with ``entry``'s ID, if one is in the history; read only then."""
        if entry.id_ not in self._position:
            return None
        return self.storage.get(self.session_id, [entry.id_])[0]

    @staticmethod
    def _repeats(entry: Entry, stored: Entry | None) -> bool:
        """Whether ``entry`` repeats ``stored``: the same class and the same content, as stored."""
        return (
            stored is not None
            and type(stored) is type(entry)
            and stored.model_dump_json() == entry.model_dump_json()
        )

    def _check_new(self, entry: Entry) -> None:
        """Raise ``ValueError`` for an entry ``add`` refuses."""
        if isinstance(entry, SummaryEntry):
            raise ValueError("summaries come from compaction, not add()")
        if isinstance(entry, WorkingNotesEntry):
            raise ValueError(
                "working notes come from the note taker or set_notes(), not add()"
            )
        if isinstance(entry, RewindEntry):
            raise ValueError("rewinds come from rewind(), not add()")
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
        if self.notes_every is not None:
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
        self,
        max_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
        media_tokens: Mapping[str, int] = MEDIA_TOKENS,
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
        (``Tokens.count``): close, not exact, for other providers' models.
        Each attached image, audio clip and document adds its estimate from
        ``media_tokens`` (keys ``"image"``, ``"audio"``, ``"document"``; a
        kind left out counts 0, so ``{}`` counts media only as their
        placeholder text). The default is ``MEDIA_TOKENS``.

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
        ``encoding`` is unknown, ``media_tokens`` has another key or a value
        that is not a non-negative integer, or the pinned entries alone take more than
        ``max_tokens``. If the compaction fails, its error is raised and the
        window stays as it was.
        """
        self._check_compact(max_tokens, encoding, media_tokens)
        window = self.window
        kept, dropped = self._squeeze(window, max_tokens, encoding, media_tokens)
        update = self._summarize(dropped) if dropped else None
        self._window = [entry.id_ for entry in window if entry.id_ in kept]
        return update

    async def compact_async(
        self,
        max_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
        media_tokens: Mapping[str, int] = MEDIA_TOKENS,
    ) -> SummaryUpdate | None:
        """``compact``, with async I/O."""
        async with self._lock:
            self._check_compact(max_tokens, encoding, media_tokens)
            window = await self.window_async()
            kept, dropped = self._squeeze(window, max_tokens, encoding, media_tokens)
            update = await self._summarize_async(dropped) if dropped else None
            self._window = [entry.id_ for entry in window if entry.id_ in kept]
            return update

    def _check_compact(
        self, max_tokens: int | None, encoding: str, media_tokens: Mapping[str, int]
    ) -> None:
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
        check_media_tokens(media_tokens)

    def _squeeze(
        self,
        window: list[Entry],
        max_tokens: int | None,
        encoding: str,
        media_tokens: Mapping[str, int],
    ) -> tuple[set[str], list[Entry]]:
        """The IDs ``compact`` keeps in ``window``, and the entries it summarizes."""
        kept = {entry.id_ for entry in window if entry.id_ in self._pinned}
        if max_tokens is not None:
            used = sum(
                entry_tokens(entry, encoding, media_tokens)
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
                used += entry_tokens(entry, encoding, media_tokens)
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

    def set_notes(self, section: str, value: JsonValue) -> None:
        """Set one section of the working notes to ``value`` directly, without an LLM call.

        For what the app knows for certain (a budget typed into a form, a
        booking number a tool returned). The current notes with that section
        replaced (or added) are stored as a new version, which becomes the
        current one. The entries since the note taker's last update stay
        queued: its next update still gets them, along with these notes.

        With a ``note_taker``, ``section`` must be one of its schema's fields
        and ``value`` must fit that field; it is stored as the schema
        serializes it. Without one, any section and JSON value will do.
        Raises ``TypeError`` or ``ValueError``, changing nothing, otherwise.
        """
        entry = self._notes_with(self.notes, section, value)
        self.storage.add(self.session_id, entry)
        self._ids.add(entry.id_)
        self._notes_id = entry.id_

    async def set_notes_async(self, section: str, value: JsonValue) -> None:
        """``set_notes``, with async I/O."""
        async with self._lock:
            entry = self._notes_with(await self.notes_async(), section, value)
            await self.storage.add_async(self.session_id, entry)
            self._ids.add(entry.id_)
            self._notes_id = entry.id_

    def _notes_with(
        self, notes: dict[str, JsonValue] | None, section: str, value: JsonValue
    ) -> WorkingNotesEntry:
        """The app's notes version: ``notes`` with ``section`` set to the checked ``value``."""
        Checks.non_empty_string(section, "section")
        if self.note_taker is not None:
            schema = self.note_taker.schema
            field = schema.model_fields.get(section)
            if field is None:
                raise ValueError(
                    f"{section!r} is not a section of {schema.__name__}: "
                    f"{', '.join(schema.model_fields)}"
                )
            # A model with only this field, keeping its type and constraints.
            definition: dict[str, Any] = {section: (field.annotation, field)}
            one = create_model(schema.__name__, **definition)
            try:
                value = one.model_validate({section: value}).model_dump(mode="json")[section]
            except ValidationError as error:
                raise ValueError(
                    f"{value!r} does not fit section {section!r} of {schema.__name__}: {error}"
                ) from error
        return WorkingNotesEntry(notes={**(notes or {}), section: value}, source="app")

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

    # ------------------------------------------------------------------ rewind

    def rewind(self, entry_id: str) -> None:
        """Go back to just after the entry with ``entry_id``: what was added after it no longer counts.

        The session becomes what it was just before the next entry was
        added: the window, the summary, the notes and the system prompt as
        they were then (the summary and notes made right after that entry
        stay), and the note taker's queue with them. New adds continue from
        there. Pins on entries that are still there are kept.

        Nothing is removed from the storage: a ``RewindEntry`` is stored,
        and reading the history applies it, so a continued session comes
        back the same. ``entries`` still lists everything, the rewound
        entries and the marker too, and their IDs cannot be used again.

        The window is chosen from the history up to there with the current
        policy; with the policy the session had then, that is the window it
        had, and nothing is summarized. With another policy, what it leaves
        out is summarized (an LLM call), as on ``policy``.

        Raises ``ValueError``, changing nothing, unless ``entry_id`` names an
        added entry still in the history (not a summary, notes, a rewind
        marker, or an entry already rewound).
        """
        self._check_rewind(entry_id)
        self.storage.add(self.session_id, RewindEntry(to_id=entry_id))
        pins, adds = self._pinned, self._adds_since_dump
        self._read_state()
        self._keep(pins, adds)
        self._settle(self._pool())

    async def rewind_async(self, entry_id: str) -> None:
        """``rewind``, with async I/O."""
        async with self._lock:
            self._check_rewind(entry_id)
            await self.storage.add_async(self.session_id, RewindEntry(to_id=entry_id))
            pins, adds = self._pinned, self._adds_since_dump
            await self._read_state_async()
            self._keep(pins, adds)
            await self._settle_async(await self._pool_async())

    def _check_rewind(self, entry_id: str) -> None:
        """Raise ``ValueError`` unless ``entry_id`` names an added entry still in the history."""
        Checks.non_empty_string(entry_id, "entry_id")
        if entry_id not in self._position:
            raise ValueError(
                f"entry {entry_id!r} is not in the history; only an added entry not "
                "already rewound can be rewound to"
            )

    def _keep(self, pins: set[str], adds: int) -> None:
        """After a rewind read the state back: keep the ``pins`` on entries still there, and the dump count."""
        self._pinned |= {
            entry_id
            for entry_id in pins
            if entry_id in self._position and entry_id not in self._system_ids
        }
        self._adds_since_dump = adds

    @staticmethod
    def _effective(rows: list[_Row]) -> list[_Row]:
        """The entries that count, oldest first: ``rows`` with each rewind marker applied in order.

        A marker drops what was stored after its entry, from the next added
        entry on: the summaries and notes made right after it stay. The
        marker itself is left out. One whose entry is not there is ignored.
        """
        history: list[_Row] = []
        for row in rows:
            if row.type != RewindEntry.__name__:
                history.append(row)
                continue
            ids = [r.id_ for r in history]
            if row.to_id not in ids:
                continue
            end = ids.index(row.to_id) + 1
            while end < len(history) and history[end].type in _DERIVED:
                end += 1
            del history[end:]
        return history

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
        """The session's entries in the primary storage, oldest first.

        Everything stored: after a ``rewind``, also the rewound entries and
        the ``RewindEntry`` marker.
        """
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

    def _pool(self) -> list[Entry]:
        """What the window is chosen from again: the history, minus what the summary covers and old system prompts."""
        return self.storage.get(self.session_id, self._pool_ids(self.summary))

    async def _pool_async(self) -> list[Entry]:
        """``_pool``, with async I/O."""
        summary = await self.summary_async()
        return await self.storage.get_async(self.session_id, self._pool_ids(summary))

    def _pool_ids(self, summary: SummaryEntry | None) -> list[str]:
        """The IDs of ``_pool``, from the history's order kept in memory; no I/O."""
        covered = set(summary.covered(self._originals)) if summary is not None else set()
        return [
            entry_id
            for entry_id in self._originals
            if entry_id not in covered
            and (entry_id not in self._system_ids or entry_id == self._system_id)
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
            if previous is not None and self.compaction.needs_covered(previous)
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
            if previous is not None and self.compaction.needs_covered(previous)
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
            version=previous.version + 1 if previous is not None else 1,
        )

    def _install_summary(self, summary: SummaryEntry) -> None:
        """Make the stored ``summary`` the current one."""
        self._ids.add(summary.id_)
        self._summary_id = summary.id_

    @staticmethod
    def _latest_id(history: list[_Row], kind: type) -> str | None:
        """The ID of the latest entry of type ``kind`` in ``history``, or ``None``."""
        return next((r.id_ for r in reversed(history) if r.type == kind.__name__), None)
