"""Every storage keeps sessions apart, keeps each ID once, and reads back exactly what it stored."""

import asyncio
import threading

import pytest

from hiyal.memory.short_term import (
    BaseStorage,
    Context,
    Entry,
    EntryKind,
    FileStorage,
    InMemoryStorage,
    SQLiteStorage,
    Summary,
)
from hiyal.prompts import UserPrompt
from tests.memory.short_term.test_entry import one_of_each_kind, response


@pytest.fixture(params=["memory", "jsonl", "txt", "md", "sqlite"])
def kind_of_storage(request):
    return request.param


@pytest.fixture
def make(kind_of_storage, tmp_path, request):
    """Opens the storage under test; opening it again reads what was stored (except in memory)."""

    def opened():
        if kind_of_storage == "memory":
            return InMemoryStorage()
        if kind_of_storage == "sqlite":
            storage = SQLiteStorage(tmp_path / "history.db")
            request.addfinalizer(storage.close)
            return storage
        return FileStorage(tmp_path / f"history.{kind_of_storage}", exist_ok=True)

    return opened


@pytest.fixture
def storage(make):
    return make()


def note(text):
    return Entry.of(Context(text=text))


def texts(entries):
    return [entry.payload.text for entry in entries]


def interleaved(storage):
    """Two sessions whose entries were added in turns, as two live conversations would."""
    for session_id, text in [("a", "a1"), ("b", "b1"), ("a", "a2"), ("b", "b2"), ("a", "a3")]:
        storage.add(session_id, note(text))


class TestSessions:
    def test_each_session_sees_only_its_own_entries_in_order(self, storage):
        interleaved(storage)
        assert texts(storage.entries("a")) == ["a1", "a2", "a3"] and texts(storage.entries("b")) == ["b1", "b2"]

    def test_a_session_nothing_was_added_to_is_empty(self, storage):
        assert storage.entries("never-used") == [] and storage.index("never-used") == []

    def test_clear_empties_one_session_only(self, storage):
        interleaved(storage)
        storage.clear("a")
        assert storage.entries("a") == [] and texts(storage.entries("b")) == ["b1", "b2"]

    def test_clearing_a_session_nothing_was_added_to_changes_nothing(self, storage):
        interleaved(storage)
        storage.clear("never-used")
        assert texts(storage.entries("a")) == ["a1", "a2", "a3"] and texts(storage.entries("b")) == ["b1", "b2"]

    @pytest.mark.parametrize("call", ["add", "get", "index", "entries", "remove", "clear"])
    def test_an_empty_session_id_is_refused(self, storage, call):
        arguments = {"add": (note("a"),), "get": ([],), "remove": ("some-id",)}.get(call, ())
        with pytest.raises(ValueError, match="session_id must not be empty"):
            getattr(storage, call)("", *arguments)

    def test_a_session_id_that_is_not_a_string_is_refused(self, storage):
        with pytest.raises(TypeError, match="session_id must be a string, got int"):
            storage.add(42, note("a"))  # type: ignore[arg-type]


class TestReads:
    @pytest.mark.parametrize("payload", one_of_each_kind(), ids=lambda p: type(p).__name__)
    def test_every_kind_reads_back_exactly(self, storage, payload):
        entry = Entry.of(payload, turn_id="t1")
        storage.add("s", entry)
        (again,) = storage.entries("s")
        assert type(again.payload) is type(payload) and again.model_dump() == entry.model_dump()

    @pytest.mark.parametrize(
        "line_break",
        ["\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "],
        ids=["LF", "CR", "CRLF", "VT", "FF", "FS", "GS", "RS", "NEL", "LS", "PS"],
    )
    def test_text_with_any_line_break_reads_back(self, storage, line_break):
        # Every character str.splitlines() breaks at; a file finds its records line by line.
        text = f"line one{line_break}line two"
        storage.add("s", note(text))
        storage.add("s", note("after"))
        assert texts(storage.entries("s")) == [text, "after"]

    def test_get_returns_the_entries_in_the_order_asked(self, storage):
        interleaved(storage)
        ids = [entry_id for entry_id, _ in storage.index("a")]
        assert texts(storage.get("a", [ids[2], ids[0]])) == ["a3", "a1"] and storage.get("a", []) == []

    def test_get_returns_an_entry_asked_twice_twice(self, storage):
        entry = note("a")
        storage.add("s", entry)
        assert storage.get("s", [entry.id_, entry.id_]) == [entry, entry]

    def test_get_names_every_missing_id(self, storage):
        storage.add("s", note("a"))
        with pytest.raises(ValueError, match=r"entries \['x', 'y'\] are not stored in session 's'"):
            storage.get("s", ["x", "y"])

    def test_get_refuses_an_id_the_session_does_not_hold(self, storage):
        interleaved(storage)
        b1 = storage.entries("b")[0]
        with pytest.raises(ValueError, match="are not stored in session 'a'"):
            storage.get("a", [b1.id_])

    def test_the_index_lists_ids_and_kinds_in_order(self, storage):
        entries = [Entry.of(UserPrompt(text="q")), Entry.of(response()), Entry.of(Summary(text="s"))]
        for entry in entries:
            storage.add("s", entry)
        assert storage.index("s") == [(e.id_, e.kind) for e in entries]
        assert [kind for _, kind in storage.index("s")] == [EntryKind.USER_PROMPT, EntryKind.LLM_RESPONSE,
                                                            EntryKind.SUMMARY]


class TestUniqueIds:
    def test_a_new_entry_is_stored(self, storage):
        assert storage.add("s", note("a")) is True

    def test_a_repeat_is_not_stored_again(self, storage):
        entry = note("a")
        storage.add("s", entry)
        assert storage.add("s", entry) is False and len(storage.entries("s")) == 1

    @pytest.mark.parametrize("payload", one_of_each_kind(), ids=lambda p: type(p).__name__)
    def test_every_kind_is_recognized_as_a_repeat_after_storing(self, storage, payload):
        entry = Entry.of(payload, turn_id="t1")
        storage.add("s", entry)
        assert storage.add("s", entry) is False

    def test_a_response_repeats_without_its_raw_payload_counting(self, storage):
        entry = Entry.of(response())
        storage.add("s", entry)
        retried = entry.model_copy(update={"payload": entry.payload.model_copy(update={"raw": {"other": 1}})})
        assert storage.add("s", retried) is False

    def test_another_entry_with_a_stored_id_is_refused(self, storage):
        entry = note("a")
        storage.add("s", entry)
        with pytest.raises(ValueError, match="another entry with id_ .* is already in session 's'"):
            storage.add("s", Entry(id_=entry.id_, kind=EntryKind.CONTEXT, payload=Context(text="other")))
        assert texts(storage.entries("s")) == ["a"]

    def test_the_same_id_can_live_in_two_sessions(self, storage):
        entry = note("a")
        assert storage.add("s1", entry) and storage.add("s2", entry)

    def test_an_id_is_free_again_after_remove_or_clear(self, storage):
        first, second = note("a"), note("b")
        storage.add("s", first)
        storage.add("s", second)
        storage.remove("s", first.id_)
        assert texts(storage.entries("s")) == ["b"] and storage.add("s", first)
        storage.clear("s")
        assert storage.add("s", second)


class TestRemove:
    def test_the_others_keep_their_order(self, storage):
        a, b, c = note("a"), note("b"), note("c")
        for entry in (a, b, c):
            storage.add("s", entry)
        storage.remove("s", b.id_)
        assert texts(storage.entries("s")) == ["a", "c"]
        assert storage.index("s") == [(a.id_, EntryKind.CONTEXT), (c.id_, EntryKind.CONTEXT)]

    def test_an_empty_entry_id_is_refused(self, storage):
        storage.add("s", note("a"))
        with pytest.raises(ValueError, match="entry_id must not be empty"):
            storage.remove("s", "")
        assert texts(storage.entries("s")) == ["a"]

    def test_removing_an_id_the_session_does_not_hold_is_refused(self, storage):
        interleaved(storage)
        b1 = storage.entries("b")[0]
        with pytest.raises(ValueError, match="is not stored in session 'a'"):
            storage.remove("a", b1.id_)
        assert texts(storage.entries("b")) == ["b1", "b2"]


class SyncOnly(BaseStorage):
    """A storage with only the sync methods, kept in memory; records the thread each call runs in."""

    def __init__(self):
        self.inner, self.threads = InMemoryStorage(), []

    def _called(self):
        self.threads.append(threading.get_ident())

    def add(self, session_id, entry):
        self._called()
        return self.inner.add(session_id, entry)

    def get(self, session_id, entry_ids):
        self._called()
        return self.inner.get(session_id, entry_ids)

    def entries(self, session_id):
        self._called()
        return self.inner.entries(session_id)

    def remove(self, session_id, entry_id):
        self._called()
        self.inner.remove(session_id, entry_id)

    def clear(self, session_id):
        self._called()
        self.inner.clear(session_id)


async def every_async_twin(storage):
    """Calls each async twin once, checking it does what its sync method does."""
    a, b = note("a"), note("b")
    assert await storage.add_async("s", a) is True
    assert await storage.add_async("s", a) is False
    await storage.add_async("s", b)
    assert await storage.entries_async("s") == [a, b]
    assert await storage.get_async("s", [b.id_]) == [b]
    assert await storage.index_async("s") == [(a.id_, EntryKind.CONTEXT), (b.id_, EntryKind.CONTEXT)]
    await storage.remove_async("s", a.id_)
    assert await storage.entries_async("s") == [b]
    with pytest.raises(ValueError, match="not stored"):
        await storage.remove_async("s", a.id_)
    await storage.clear_async("s")
    assert await storage.entries_async("s") == []


class TestAsync:
    def test_the_async_twins_do_what_the_sync_methods_do(self, storage):
        asyncio.run(every_async_twin(storage))

    def test_many_async_calls_at_once(self, storage):
        first, second = [note(f"a{n}") for n in range(40)], [note(f"b{n}") for n in range(40)]

        async def at_once(*calls):
            return await asyncio.gather(*calls)

        assert asyncio.run(at_once(*(storage.add_async("s", entry) for entry in first))) == [True] * 40
        # Removing half of them while adding others: no change may be lost.
        asyncio.run(at_once(
            *(storage.remove_async("s", entry.id_) for entry in first[:20]),
            *(storage.add_async("s", entry) for entry in second),
        ))
        stored = texts(storage.entries("s"))
        assert len(stored) == 60
        assert set(stored) == {f"a{n}" for n in range(20, 40)} | {f"b{n}" for n in range(40)}

    def test_in_memory_runs_its_async_twins_without_a_worker_thread(self):
        class Spy(InMemoryStorage):
            threads = []

            def add(self, session_id, entry):
                self.threads.append(threading.get_ident())
                return super().add(session_id, entry)

        asyncio.run(Spy().add_async("s", note("a")))
        assert Spy.threads == [threading.get_ident()]

    def test_by_default_the_async_twins_run_the_sync_methods_in_a_worker_thread(self):
        storage = SyncOnly()
        asyncio.run(every_async_twin(storage))
        assert storage.threads and threading.get_ident() not in storage.threads
