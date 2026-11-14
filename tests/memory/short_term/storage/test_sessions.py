"""Every storage serves many sessions at once and keeps them apart."""

import asyncio

import pytest

from hiyal.memory.short_term import ContextEntry, FileStorage, InMemoryStorage, RewindEntry, SQLiteStorage


@pytest.fixture(params=["memory", "jsonl", "txt", "md", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return InMemoryStorage()
    if request.param == "sqlite":
        storage = SQLiteStorage(tmp_path / "history.db")
        request.addfinalizer(storage.close)
        return storage
    return FileStorage(tmp_path / f"history.{request.param}")


def texts(storage, session_id):
    return [entry.text for entry in storage.entries(session_id)]


def interleaved(storage):
    """Two sessions whose entries were added in turns, as two live conversations would."""
    for session_id, text in [("a", "a1"), ("b", "b1"), ("a", "a2"), ("b", "b2"), ("a", "a3")]:
        storage.add(session_id, ContextEntry(text=text))


def test_each_session_sees_only_its_own_entries_in_order(storage):
    interleaved(storage)
    assert texts(storage, "a") == ["a1", "a2", "a3"]
    assert texts(storage, "b") == ["b1", "b2"]


def test_a_session_nothing_was_added_to_is_empty(storage):
    interleaved(storage)
    assert storage.entries("never-used") == []


def test_clear_empties_one_session_only(storage):
    interleaved(storage)
    storage.clear("a")
    assert texts(storage, "a") == []
    assert texts(storage, "b") == ["b1", "b2"]


def test_remove_acts_within_the_session(storage):
    interleaved(storage)
    entry = storage.entries("b")[0]
    storage.remove("b", entry.id_)
    assert texts(storage, "b") == ["b2"]
    assert texts(storage, "a") == ["a1", "a2", "a3"]


def test_an_entry_of_another_session_cannot_be_removed(storage):
    interleaved(storage)
    entry = storage.entries("b")[0]
    with pytest.raises(ValueError, match="is not stored in session 'a'"):
        storage.remove("a", entry.id_)
    assert texts(storage, "b") == ["b1", "b2"]


def test_the_same_entry_can_be_in_two_sessions(storage):
    entry = ContextEntry(text="shared")
    storage.add("a", entry)
    storage.add("b", entry)
    storage.remove("a", entry.id_)
    assert texts(storage, "a") == []
    assert texts(storage, "b") == ["shared"]


@pytest.mark.parametrize("session_id, error", [(None, TypeError), (7, TypeError), ("", ValueError)])
def test_session_id_must_be_a_non_empty_string(storage, session_id, error):
    entry = ContextEntry(text="x")
    for call in (
        lambda: storage.add(session_id, entry),
        lambda: storage.entries(session_id),
        lambda: storage.remove(session_id, entry.id_),
        lambda: storage.clear(session_id),
    ):
        with pytest.raises(error, match="session_id"):
            call()


class TestGet:
    def test_returns_the_entries_in_the_order_of_the_ids(self, storage):
        entries = [ContextEntry(text=t) for t in ("x", "y", "z")]
        for entry in entries:
            storage.add("a", entry)
        got = storage.get("a", [entries[2].id_, entries[0].id_])
        assert [entry.text for entry in got] == ["z", "x"]

    def test_no_ids_no_entries(self, storage):
        storage.add("a", ContextEntry(text="x"))
        assert storage.get("a", []) == []

    def test_an_id_of_another_session_is_not_found(self, storage):
        other = ContextEntry(text="b1")
        storage.add("b", other)
        storage.add("a", ContextEntry(text="a1"))
        with pytest.raises(ValueError, match="not stored in session 'a'"):
            storage.get("a", [other.id_])

    def test_a_repeated_id_gives_the_oldest_copy(self, storage):
        first = ContextEntry(text="first")
        storage.add("a", first)
        storage.add("a", first.model_copy(update={"text": "second"}))
        assert [entry.text for entry in storage.get("a", [first.id_])] == ["first"]

    def test_an_id_must_be_a_string(self, storage):
        entry = ContextEntry(text="x")
        storage.add("a", entry)
        with pytest.raises(TypeError, match="entry_id must be a string"):
            storage.get("a", [entry])

    def test_many_ids_at_once(self, storage):
        entries = [ContextEntry(text=str(n)) for n in range(1200)]  # more than one SQLite batch
        for entry in entries:
            storage.add("a", entry)
        wanted = [entry.id_ for entry in reversed(entries)]
        assert [entry.id_ for entry in storage.get("a", wanted)] == wanted


def test_the_index_lists_each_entrys_id_and_type_in_order_per_session(storage):
    interleaved(storage)
    storage.add("a", RewindEntry(to_id="x"))
    expected = [(entry.id_, type(entry).__name__) for entry in storage.entries("a")]
    assert storage.index("a") == expected and [name for _, name in expected][-1] == "RewindEntry"
    assert [entry_id for entry_id, _ in storage.index("b")] == [entry.id_ for entry in storage.entries("b")]
    assert storage.index("never-used") == []


def test_the_async_index_is_the_same(storage):
    interleaved(storage)
    assert asyncio.run(storage.index_async("a")) == storage.index("a")


def test_get_after_index_returns_just_those_entries(storage):
    interleaved(storage)
    ids = [entry_id for entry_id, _ in storage.index("a")]
    assert [entry.text for entry in storage.get("a", [ids[2], ids[0]])] == ["a3", "a1"]
