"""What only one storage does: the file's formats and safety, SQLite's batches."""

import sqlite3

import pytest

from hiyal.memory.short_term import Context, Entry, FileStorage, SQLiteStorage


def note(text):
    return Entry.of(Context(text=text))


class TestFileStorage:
    def test_an_existing_file_is_not_written_to_by_accident(self, tmp_path):
        path = tmp_path / "history.jsonl"
        FileStorage(path).add("s", note("a"))
        with pytest.raises(FileExistsError, match="exist_ok=True"):
            FileStorage(path)
        assert [e.payload.text for e in FileStorage(path, exist_ok=True).entries("s")] == ["a"]

    def test_an_unsupported_extension_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unsupported file extension '.csv'"):
            FileStorage(tmp_path / "history.csv")

    def test_the_extension_is_read_in_any_case(self, tmp_path):
        storage = FileStorage(tmp_path / "history.JSONL")
        storage.add("s", note("a"))
        assert [e.payload.text for e in storage.entries("s")] == ["a"]

    def test_the_file_and_its_folders_are_made_on_the_first_write(self, tmp_path):
        path = tmp_path / "sessions" / "2026" / "history.jsonl"
        storage = FileStorage(path)
        assert storage.entries("s") == [] and not path.parent.exists()
        storage.clear("s")
        assert not path.exists()
        storage.add("s", note("a"))
        assert path.exists()

    def test_a_rewrite_leaves_no_temporary_file(self, tmp_path):
        path = tmp_path / "history.jsonl"
        storage = FileStorage(path)
        first = note("a")
        storage.add("s", first)
        storage.add("s", note("b"))
        storage.remove("s", first.id_)
        storage.clear("s")
        assert [item.name for item in tmp_path.iterdir()] == ["history.jsonl"]

    def test_a_reopened_file_still_knows_its_ids(self, tmp_path):
        path = tmp_path / "history.jsonl"
        entry = note("a")
        FileStorage(path).add("s", entry)
        again = FileStorage(path, exist_ok=True)
        assert again.add("s", entry) is False
        with pytest.raises(ValueError, match="another entry"):
            again.add("s", Entry(id_=entry.id_, kind=entry.kind, payload=Context(text="other")))

    def test_txt_and_md_show_the_readable_text(self, tmp_path):
        for suffix, readable in [(".txt", "    Context:\n    Prefers trains."), (".md", "> Context:\n> Prefers trains.")]:
            path = tmp_path / f"history{suffix}"
            FileStorage(path).add("s", note("Prefers trains."))
            text = path.read_text()
            assert readable in text and "context ·" in text

    def test_markdown_survives_text_that_could_close_its_comment(self, tmp_path):
        path = tmp_path / "history.md"
        FileStorage(path).add("s", note("a --> b > c"))
        assert FileStorage(path, exist_ok=True).entries("s")[0].payload.text == "a --> b > c"


class TestSQLiteStorage:
    def test_get_reads_in_batches(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "history.db", batch_size=2)
        entries = [note(str(n)) for n in range(5)]
        for entry in entries:
            storage.add("s", entry)
        wanted = [entries[4].id_, entries[0].id_, entries[2].id_, entries[3].id_, entries[1].id_]
        assert [e.payload.text for e in storage.get("s", wanted)] == ["4", "0", "2", "3", "1"]
        storage.close()

    @pytest.mark.parametrize("batch_size", [0, -1, True])
    def test_batch_size_must_be_a_positive_integer(self, tmp_path, batch_size):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            SQLiteStorage(tmp_path / "history.db", batch_size=batch_size)

    def test_an_in_memory_database_makes_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        storage = SQLiteStorage(":memory:")
        storage.add("s", note("a"))
        assert [e.payload.text for e in storage.entries("s")] == ["a"] and list(tmp_path.iterdir()) == []
        storage.close()

    def test_the_folders_are_made(self, tmp_path):
        path = tmp_path / "sessions" / "history.db"
        SQLiteStorage(path).close()
        assert path.exists()

    def test_a_reopened_database_keeps_its_entries_and_ids(self, tmp_path):
        path = tmp_path / "history.db"
        entry = note("a")
        first = SQLiteStorage(path)
        first.add("s", entry)
        first.close()
        again = SQLiteStorage(path)
        assert again.entries("s") == [entry] and again.add("s", entry) is False
        with pytest.raises(ValueError, match="another entry"):
            again.add("s", Entry(id_=entry.id_, kind=entry.kind, payload=Context(text="other")))
        again.close()

    def test_a_closed_storage_cannot_be_used(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "history.db")
        storage.close()
        with pytest.raises(sqlite3.ProgrammingError):
            storage.entries("s")
