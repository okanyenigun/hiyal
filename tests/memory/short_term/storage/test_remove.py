"""``remove`` behaves the same in every storage: by ID only, oldest copy first."""

import pytest

from hiyal.memory.short_term import ContextEntry, FileStorage, InMemoryStorage, SQLiteStorage

S = "session-1"


@pytest.fixture(params=["memory", "jsonl", "md", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return InMemoryStorage()
    if request.param == "sqlite":
        storage = SQLiteStorage(tmp_path / "history.db")
        request.addfinalizer(storage.close)
        return storage
    return FileStorage(tmp_path / f"history.{request.param}")


def filled(storage, *texts):
    entries = [ContextEntry(text=text) for text in texts]
    for entry in entries:
        storage.add(S, entry)
    return entries


def texts(storage):
    return [entry.text for entry in storage.entries(S)]


def test_an_entry_is_refused(storage):
    _, b, _ = filled(storage, "a", "b", "c")
    with pytest.raises(TypeError, match="entry_id must be a string"):
        storage.remove(S, b)
    assert texts(storage) == ["a", "b", "c"]


def test_remove_by_id(storage):
    _, b, _ = filled(storage, "a", "b", "c")
    storage.remove(S, b.id_)
    assert texts(storage) == ["a", "c"]


def test_an_unknown_id_raises(storage):
    filled(storage, "a")
    with pytest.raises(ValueError, match="'no-such-id' is not stored"):
        storage.remove(S, "no-such-id")


def test_a_duplicated_id_loses_only_its_oldest_copy(storage):
    a, b = filled(storage, "a", "b")
    storage.add(S, a)
    storage.remove(S, a.id_)
    assert texts(storage) == ["b", "a"]
