"""Storage that keeps entries in a file, so they survive the process."""

import os
import threading
from pathlib import Path
from typing import Any
from .base import BaseStorage
from .formats import FORMATS, FileFormat
from ..schemas import Entry, EntryKind
from ....utility import Checks


class FileStorage(BaseStorage):
    """Keeps entries in a file at ``path``, oldest first, in a format chosen by its extension.

    - ``.jsonl``: one JSON record per line.
    - ``.txt``: each entry as readable text, followed by its record.
    - ``.md``: each entry as a Markdown section, with its record hidden in a comment.

    One file can hold many sessions: each record names its session. Adding
    an entry appends to the file; removing one, or clearing a session,
    rewrites it. A file has no index, so every read goes through the whole
    file, but only the entries asked for are rebuilt. Media bytes are
    stored as base64, and ``LLMResponse.raw`` is not stored.

    To keep IDs unique without reading the file on every add, the IDs are
    read once, on first use, and kept up to date. So one storage object
    should be the only writer of its file. Its calls take turns (a lock),
    so the async twins inherited from ``BaseStorage`` can run them in
    worker threads, even several at once.
    """

    def __init__(self, path: str | Path, exist_ok: bool = False) -> None:
        """Use the file at ``path``; it and its folders are created on the first write.

        By default the file must not exist yet, so an existing file is never
        written to by accident: ``FileExistsError`` is raised. Pass
        ``exist_ok=True`` to continue an existing file, e.g. after a restart.
        Raises ``ValueError`` for an extension other than ``.jsonl``,
        ``.txt`` or ``.md``.
        """
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix not in FORMATS:
            raise ValueError(
                f"unsupported file extension {suffix!r}; use {', '.join(FORMATS)}"
            )
        if self.path.exists() and not exist_ok:
            raise FileExistsError(
                f"{self.path} already exists; pass exist_ok=True to continue it"
            )
        self.format: FileFormat = FORMATS[suffix]
        # The IDs stored in the file, per session; read on first use (see _known).
        self._ids: dict[str, set[str]] | None = None
        # One call at a time, as the async twins run calls in worker threads.
        # Reentrant, as add() calls get().
        self._lock = threading.RLock()

    def add(self, session_id: str, entry: Entry) -> bool:
        with self._lock:
            ids = self._known().setdefault(
                Checks.non_empty_string(session_id, "session_id"), set()
            )
            if entry.id_ in ids:
                (stored,) = self.get(session_id, [entry.id_])
                return self._repeat(session_id, entry, stored)
            block = self.format.block(session_id, entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(block)
            ids.add(entry.id_)
            return True

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        with self._lock:
            Checks.non_empty_string(session_id, "session_id")
            wanted = set(entry_ids)
            found = {
                data["id_"]: data
                for session, data in self._records()
                if session == session_id and data["id_"] in wanted
            }
            self._missing(session_id, entry_ids, set(found))
            return [Entry.model_validate(found[entry_id]) for entry_id in entry_ids]

    def index(self, session_id: str) -> list[tuple[str, EntryKind]]:
        # The whole file is read, but no entry is rebuilt.
        with self._lock:
            Checks.non_empty_string(session_id, "session_id")
            return [
                (data["id_"], EntryKind(data["kind"]))
                for session, data in self._records()
                if session == session_id
            ]

    def entries(self, session_id: str) -> list[Entry]:
        with self._lock:
            Checks.non_empty_string(session_id, "session_id")
            return [
                Entry.model_validate(data)
                for session, data in self._records()
                if session == session_id
            ]

    def remove(self, session_id: str, entry_id: str) -> None:
        with self._lock:
            Checks.non_empty_string(session_id, "session_id")
            Checks.non_empty_string(entry_id, "entry_id")
            records = self._records()
            kept = [
                (session, data)
                for session, data in records
                if not (session == session_id and data["id_"] == entry_id)
            ]
            if len(kept) == len(records):
                raise ValueError(
                    f"entry {entry_id!r} is not stored in session {session_id!r}"
                )
            self._write(kept)
            self._known()[session_id].discard(entry_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            Checks.non_empty_string(session_id, "session_id")
            records = self._records()
            kept = [
                (session, data) for session, data in records if session != session_id
            ]
            if len(kept) < len(records):
                self._write(kept)
            self._known().pop(session_id, None)

    def _known(self) -> dict[str, set[str]]:
        """The IDs stored in the file, per session: read once, then kept up to date by every change."""
        if self._ids is None:
            self._ids = {}
            for session, data in self._records():
                self._ids.setdefault(session, set()).add(data["id_"])
        return self._ids

    def _records(self) -> list[tuple[str, dict[str, Any]]]:
        """Every record in the file as ``(session_id, entry JSON)``, oldest first, nothing rebuilt."""
        if not self.path.exists():
            return []
        return self.format.records(self.path.read_text(encoding="utf-8"))

    def _write(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        """Replace the file's content in one step, so a crash cannot leave it half written."""
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            file.writelines(
                self.format.block(session, Entry.model_validate(data))
                for session, data in records
            )
        os.replace(temporary, self.path)
