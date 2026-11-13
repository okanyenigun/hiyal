"""Every storage's ``_async`` methods: the same results as the sync ones, and safe when run concurrently."""

import asyncio
import threading

import pytest

from hiyal.memory.short_term import ContextEntry, FileStorage, InMemoryStorage, SQLiteStorage

from tests.support import run


@pytest.fixture(params=["memory", "jsonl", "txt", "md", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return InMemoryStorage()
    if request.param == "sqlite":
        storage = SQLiteStorage(tmp_path / "history.db")
        request.addfinalizer(storage.close)
        return storage
    return FileStorage(tmp_path / f"history.{request.param}")


def texts(entries) -> list[str]:
    return [str(entry.text) for entry in entries]


def test_every_method_works_like_its_sync_twin(storage):
    a, b, c = (ContextEntry(text=text) for text in "abc")

    async def session():
        for entry in (a, b, c):
            await storage.add_async("s1", entry)
        assert texts(await storage.entries_async("s1")) == ["a", "b", "c"]
        assert texts(await storage.get_async("s1", [c.id_, a.id_])) == ["c", "a"]
        await storage.remove_async("s1", b.id_)
        assert texts(await storage.entries_async("s1")) == ["a", "c"]
        await storage.clear_async("s1")
        assert await storage.entries_async("s1") == []

    run(session())


def test_errors_are_raised_as_in_sync_code(storage):
    async def session():
        with pytest.raises(ValueError, match="not stored"):
            await storage.remove_async("s1", "missing")
        with pytest.raises(ValueError, match="not stored"):
            await storage.get_async("s1", ["missing"])
        with pytest.raises(TypeError, match="session_id must be a string"):
            await storage.entries_async(7)  # type: ignore[arg-type]

    run(session())


def test_concurrent_adds_lose_nothing_and_keep_the_file_readable(storage):
    entries = [ContextEntry(text=f"e{number}") for number in range(60)]

    async def session():
        await asyncio.gather(*(storage.add_async("s1", entry) for entry in entries))

    run(session())
    assert sorted(texts(storage.entries("s1"))) == sorted(texts(entries))  # read back, so every record is whole


def test_concurrent_adds_and_rewrites_do_not_corrupt_each_other(storage):
    kept = [ContextEntry(text=f"k{number}") for number in range(20)]
    doomed = [ContextEntry(text=f"d{number}") for number in range(20)]
    for entry in doomed:
        storage.add("s1", entry)

    async def session():
        await asyncio.gather(
            *(storage.add_async("s1", entry) for entry in kept),
            *(storage.remove_async("s1", entry.id_) for entry in doomed),  # each one rewrites a file
            *(storage.add_async("s2", ContextEntry(text=f"other{number}")) for number in range(5)),
        )

    run(session())
    assert sorted(texts(storage.entries("s1"))) == sorted(texts(kept))
    assert len(storage.entries("s2")) == 5


def test_a_sqlite_storage_can_be_used_from_another_thread(tmp_path):
    storage = SQLiteStorage(tmp_path / "history.db")
    storage.add("s1", ContextEntry(text="from the main thread"))
    seen = []
    worker = threading.Thread(target=lambda: seen.extend(texts(storage.entries("s1"))))
    worker.start()
    worker.join()
    assert seen == ["from the main thread"]
    storage.close()
