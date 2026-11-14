"""Storage that keeps entries in a SQLite database, so they survive the process."""

import json
import sqlite3
import threading
from pathlib import Path
from .base import BaseStorage
from ..entry import Entry, build_entry, entry_type_name
from ....utility import Checks


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    position   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    id         TEXT NOT NULL,
    type       TEXT NOT NULL,
    entry      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_by_session ON entries (session_id, position);
CREATE INDEX IF NOT EXISTS entries_by_id ON entries (session_id, id);
"""


class SQLiteStorage(BaseStorage):
    """Keeps entries in a SQLite database at ``path``, using Python's built-in ``sqlite3``.

    One row per entry in an ``entries`` table: ``position`` keeps the order,
    ``session_id`` says which session the entry belongs to, ``id`` finds an
    entry for ``remove``, ``type`` names its class, and ``entry`` holds its
    JSON. One database holds any number of sessions; every call reads or
    changes only the rows of its own session. Media bytes are stored as
    base64, and ``LLMResponse.raw`` is not stored.

    The storage keeps one connection open for its lifetime; ``close()`` ends it.
    A lock lets one call at a time use it, so the storage is safe to share
    between threads (the ``_async`` methods run in worker threads).
    """

    def __init__(self, path: str | Path, batch_size: int = 500) -> None:
        """Use the database at ``path``; it, its folders and its table are created if missing.

        An existing database is opened as it is: its sessions stay apart, so
        a new session never mixes into an old one. ``":memory:"`` gives a
        temporary database that is lost when the storage is closed or the
        process ends.

        ``batch_size`` is the most IDs one ``get`` query asks for; more IDs
        take more queries. SQLite limits the parameters of one query (999 in
        older versions), so it should stay below that. Raises ``ValueError``
        if it is not a positive integer.
        """
        self.batch_size = Checks.positive_int(batch_size, "batch_size")
        self.path = path if path == ":memory:" else Path(path)
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Usable from any thread; the lock lets one call at a time use it.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._connection:
            self._connection.executescript(_SCHEMA)

    def add(self, session_id: str, entry: Entry) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO entries (session_id, id, type, entry) VALUES (?, ?, ?, ?)",
                (
                    Checks.non_empty_string(session_id, "session_id"),
                    entry.id_,
                    entry_type_name(entry),
                    entry.model_dump_json(),
                ),
            )

    def remove(self, session_id: str, entry_id: str) -> None:
        target = Checks.non_empty_string(entry_id, "entry_id")
        Checks.non_empty_string(session_id, "session_id")
        # If an ID was stored twice, only the oldest copy goes, like list.remove.
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM entries WHERE position = "
                "(SELECT MIN(position) FROM entries WHERE session_id = ? AND id = ?)",
                (session_id, target),
            )
        if cursor.rowcount == 0:
            raise ValueError(
                f"entry {target!r} is not stored in session {session_id!r}"
            )

    def entries(self, session_id: str) -> list[Entry]:
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:
            rows = self._connection.execute(
                "SELECT type, entry FROM entries WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return [build_entry(name, json.loads(data)) for name, data in rows]

    def index(self, session_id: str) -> list[tuple[str, str]]:
        # Two small columns; no entry JSON is read or rebuilt.
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, type FROM entries WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return [(entry_id, name) for entry_id, name in rows]

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        wanted = [
            Checks.non_empty_string(entry_id, "entry_id") for entry_id in entry_ids
        ]
        Checks.non_empty_string(session_id, "session_id")
        unique = list(dict.fromkeys(wanted))
        found: list[Entry] = []
        # Only the wanted rows are read, through the (session_id, id) index; in
        # batches, as SQLite limits the number of parameters in one query.
        with self._lock:  # all batches at once, so no write lands between them
            for start in range(0, len(unique), self.batch_size):
                batch = unique[start : start + self.batch_size]
                rows = self._connection.execute(
                    f"SELECT type, entry FROM entries WHERE session_id = ? AND id IN ({', '.join('?' * len(batch))}) "
                    "ORDER BY position",
                    (session_id, *batch),
                ).fetchall()
                found += [build_entry(name, json.loads(data)) for name, data in rows]
        return self._pick(session_id, wanted, found)

    def clear(self, session_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM entries WHERE session_id = ?",
                (Checks.non_empty_string(session_id, "session_id"),),
            )

    def close(self) -> None:
        """Close the database connection; the storage cannot be used afterwards."""
        with self._lock:
            self._connection.close()
