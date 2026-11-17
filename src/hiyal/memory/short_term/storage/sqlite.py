"""Storage that keeps entries in a SQLite database, so they survive the process."""

import sqlite3
import threading
from pathlib import Path
from .base import BaseStorage
from ..schemas import Entry, EntryKind
from ....utility import Checks


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    position   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    id         TEXT NOT NULL,
    kind       TEXT NOT NULL,
    entry      TEXT NOT NULL,
    UNIQUE (session_id, id)
);
CREATE INDEX IF NOT EXISTS entries_by_session ON entries (session_id, position);
"""


class SQLiteStorage(BaseStorage):
    """Keeps entries in a SQLite database at ``path``, using Python's built-in ``sqlite3``.

    One row per entry in an ``entries`` table: ``position`` keeps the order,
    ``session_id`` says which session it belongs to, ``id`` and ``kind``
    let the index and ``get`` read only what they need, and ``entry`` holds
    its JSON. ``UNIQUE (session_id, id)`` keeps each ID once per session.
    Media bytes are stored as base64, and ``LLMResponse.raw`` is not stored.

    The storage keeps one connection open for its lifetime; ``close()`` ends
    it. Any thread may use the connection, one call at a time (a lock), so
    the async twins inherited from ``BaseStorage`` run each call in a
    worker thread without blocking the event loop.
    """

    def __init__(self, path: str | Path, batch_size: int = 500) -> None:
        """Use the database at ``path``; it, its folders and its table are created if missing.

        ``":memory:"`` gives a temporary database, lost when the storage is
        closed. ``batch_size`` is the most IDs one ``get`` query asks for
        (SQLite limits the parameters of a query); raises ``ValueError`` if
        it is not a positive integer.
        """
        self.batch_size = Checks.positive_int(batch_size, "batch_size")
        self.path = path if path == ":memory:" else Path(path)
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Shared by the threads the async twins run in; the lock lets one call use it at a time.
        # Reentrant, as add() calls get() while holding it.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock, self._connection:
            self._connection.executescript(_SCHEMA)

    def add(self, session_id: str, entry: Entry) -> bool:
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO entries (session_id, id, kind, entry) VALUES (?, ?, ?, ?)",
                        (session_id, entry.id_, entry.kind.value, entry.model_dump_json()),
                    )
            except sqlite3.IntegrityError:  # the ID is stored already
                (stored,) = self.get(session_id, [entry.id_])
                return self._repeat(session_id, entry, stored)
            return True

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        Checks.non_empty_string(session_id, "session_id")
        unique = list(dict.fromkeys(entry_ids))
        found: dict[str, Entry] = {}
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            with self._lock:
                rows = self._connection.execute(
                    f"SELECT entry FROM entries WHERE session_id = ? AND id IN ({', '.join('?' * len(batch))})",
                    (session_id, *batch),
                ).fetchall()
            for (data,) in rows:
                entry = Entry.model_validate_json(data)
                found[entry.id_] = entry
        self._missing(session_id, entry_ids, set(found))
        return [found[entry_id] for entry_id in entry_ids]

    def index(self, session_id: str) -> list[tuple[str, EntryKind]]:
        # Two small columns; no entry is read or rebuilt.
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, kind FROM entries WHERE session_id = ? ORDER BY position",
                (Checks.non_empty_string(session_id, "session_id"),),
            ).fetchall()
        return [(entry_id, EntryKind(kind)) for entry_id, kind in rows]

    def entries(self, session_id: str) -> list[Entry]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT entry FROM entries WHERE session_id = ? ORDER BY position",
                (Checks.non_empty_string(session_id, "session_id"),),
            ).fetchall()
        return [Entry.model_validate_json(data) for (data,) in rows]

    def remove(self, session_id: str, entry_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM entries WHERE session_id = ? AND id = ?",
                (
                    Checks.non_empty_string(session_id, "session_id"),
                    Checks.non_empty_string(entry_id, "entry_id"),
                ),
            )
        if cursor.rowcount == 0:
            raise ValueError(
                f"entry {entry_id!r} is not stored in session {session_id!r}"
            )

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
