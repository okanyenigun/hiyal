import sqlite3

import pytest

from hiyal.memory.short_term import ContextEntry, SQLiteStorage, WorkingNotesEntry
from hiyal.models.schemas import ResponseContent, ResponseMessage, ToolCallContent
from hiyal.prompts import Prompt
from hiyal.tools import ToolCall

from tests.memory.short_term.storage.test_file_storage import (
    PNG,
    S,
    dumped,
    every_entry,
    response,
)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "nested" / "history.db"


@pytest.fixture
def storage(path):
    storage = SQLiteStorage(path)
    yield storage
    storage.close()


class TestSQLiteStorage:
    def test_creates_the_database_and_starts_empty(self, storage, path):
        assert path.exists() and storage.entries(S) == []

    def test_every_entry_type_round_trips_in_order(self, storage):
        entries = every_entry()
        for entry in entries:
            storage.add(S, entry)
        reloaded = storage.entries(S)
        assert [type(entry) for entry in reloaded] == [type(entry) for entry in entries]
        assert dumped(reloaded) == dumped(entries)

    def test_entries_survive_a_new_storage_object(self, storage, path):
        entries = every_entry()
        for entry in entries:
            storage.add(S, entry)
        storage.close()
        reopened = SQLiteStorage(path)
        assert dumped(reopened.entries(S)) == dumped(entries)
        reopened.close()

    def test_media_and_tool_calls_survive(self, storage):
        image = ResponseMessage(id="m1", role="assistant", content=ResponseContent(
            content_type="image", data={"mime_type": "image/png", "data": PNG}))
        call = ResponseMessage(id="m2", role="assistant", content=ToolCallContent(
            data=ToolCall(raw={"sdk": "object"}, name="get_weather", arguments={"city": "Kyoto"}, status="complete")))
        storage.add(S, every_entry()[0])
        storage.add(S, response(messages=[image, call]))
        prompt_entry, reloaded = storage.entries(S)
        assert prompt_entry.images[0].data == PNG
        assert reloaded.images[0].data == PNG and reloaded.tool_calls[0].name == "get_weather"

    def test_one_row_per_entry_with_id_and_type(self, storage, path):
        a, b = ContextEntry(text="a"), response()
        storage.add(S, a)
        storage.add(S, b)
        with sqlite3.connect(path) as connection:
            rows = connection.execute("SELECT id, type FROM entries ORDER BY position").fetchall()
        assert rows == [(a.id_, "ContextEntry"), (b.id_, "LLMResponse")]

    def test_remove_keeps_the_rest_in_order(self, storage):
        a, b, c = ContextEntry(text="a"), response(), ContextEntry(text="c")
        for entry in (a, b, c):
            storage.add(S, entry)
        storage.remove(S, b.id_)
        assert [entry.id_ for entry in storage.entries(S)] == [a.id_, c.id_]

    def test_remove_takes_only_the_oldest_copy_of_a_duplicate(self, storage):
        a, b = ContextEntry(text="a"), ContextEntry(text="b")
        for entry in (a, b, a):
            storage.add(S, entry)
        storage.remove(S, a.id_)
        assert [entry.id_ for entry in storage.entries(S)] == [b.id_, a.id_]

    def test_removing_an_entry_that_is_not_stored_raises(self, storage):
        storage.add(S, ContextEntry(text="a"))
        with pytest.raises(ValueError, match="is not stored"):
            storage.remove(S, "no-such-id")

    def test_clear_removes_every_row(self, storage):
        for entry in every_entry():
            storage.add(S, entry)
        storage.clear(S)
        assert storage.entries(S) == []

    def test_a_closed_storage_cannot_be_used(self, storage):
        storage.close()
        with pytest.raises(sqlite3.ProgrammingError):
            storage.entries(S)


class TestExistingDatabases:
    def test_an_existing_database_continues_its_sessions(self, tmp_path):
        path = tmp_path / "history.db"
        first = SQLiteStorage(path)
        first.add(S, ContextEntry(text="first"))
        first.close()
        reopened = SQLiteStorage(path)
        reopened.add(S, ContextEntry(text="second"))
        assert [entry.text for entry in reopened.entries(S)] == ["first", "second"]
        reopened.close()

    def test_a_new_session_in_an_existing_database_starts_empty(self, tmp_path):
        path = tmp_path / "history.db"
        first = SQLiteStorage(path)
        first.add(S, ContextEntry(text="first"))
        first.close()
        reopened = SQLiteStorage(path)
        assert reopened.entries("another-session") == []
        reopened.close()

    def test_each_row_records_its_session(self, storage, path):
        storage.add("a", ContextEntry(text="x"))
        storage.add("b", ContextEntry(text="y"))
        with sqlite3.connect(path) as connection:
            rows = connection.execute("SELECT session_id FROM entries ORDER BY position").fetchall()
        assert rows == [("a",), ("b",)]

    def test_memory_database_needs_no_file_and_is_not_shared(self):
        first, second = SQLiteStorage(":memory:"), SQLiteStorage(":memory:")
        first.add(S, ContextEntry(text="a"))
        assert len(first.entries(S)) == 1 and second.entries(S) == []
        first.close()
        second.close()


class TestEntries:
    def test_notes_round_trip_as_plain_data(self, storage, path):
        notes = WorkingNotesEntry(notes={"objective": "Compare", "decisions": ["Exclude large-scale"]})
        storage.add(S, notes)
        storage.close()
        reopened = SQLiteStorage(path)
        assert reopened.entries(S) == [notes]
        reopened.close()

    def test_a_non_entry_cannot_be_stored(self, storage):
        with pytest.raises(TypeError, match="cannot store"):
            storage.add(S, Prompt(text="the base prompt is not an entry"))


class TestBatchSize:
    def test_get_reads_more_ids_than_one_batch_in_order(self):
        storage = SQLiteStorage(":memory:", batch_size=2)
        entries = [ContextEntry(text=text) for text in "abcde"]
        for entry in entries:
            storage.add(S, entry)
        wanted = [entries[4].id_, entries[0].id_, entries[2].id_, entries[3].id_, entries[1].id_]
        assert [entry.text for entry in storage.get(S, wanted)] == ["e", "a", "c", "d", "b"]
        storage.close()

    @pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
    def test_must_be_a_positive_integer(self, batch_size):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            SQLiteStorage(":memory:", batch_size=batch_size)
