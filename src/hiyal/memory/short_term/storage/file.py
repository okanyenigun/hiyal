"""Storage that keeps entries in a file, so they survive the process."""

import os
import threading
from pathlib import Path
from .base import BaseStorage
from .formats import FORMATS, FileFormat
from ..entry import Entry
from ....utility import Checks


class FileStorage(BaseStorage):
    """Keeps entries in a file at ``path``, oldest first, in a format chosen by its extension.

    - ``.jsonl``: one JSON record per line.
    - ``.txt``: each entry as readable text, followed by its record.
    - ``.md``: each entry as a Markdown section, with its record hidden in a comment.

    One file can hold many sessions: each record names its session, and
    every call reads or changes only the records of its own session. Every
    format is read back exactly; see ``formats``. Adding an entry appends to
    the file; removing one, or clearing a session, rewrites it. Media bytes
    are stored as base64, and ``LLMResponse.raw`` is not stored.

    A lock lets one call at a time touch the file, so an append never lands
    in the middle of a rewrite, and the storage is safe to share between
    threads (the ``_async`` methods run in worker threads). It guards this
    object only: two storages on the same file do not see each other's lock.
    """

    def __init__(
        self,
        path: str | Path,
        exist_ok: bool = False,
    ) -> None:
        """Use the file at ``path``; it and its folders are created on the first write.

        By default the file must not exist yet, so an existing file is never
        written to by accident: ``FileExistsError`` is raised. Pass
        ``exist_ok=True`` to continue an existing file, e.g. after a restart.

        Raises ``ValueError`` for an extension other than ``.jsonl``, ``.txt``
        or ``.md``.
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
        # One call at a time touches the file.
        self._lock = threading.RLock()

    def add(self, session_id: str, entry: Entry) -> None:
        block = self.format.block(
            Checks.non_empty_string(session_id, "session_id"), entry
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(block)

    def remove(self, session_id: str, entry_id: str) -> None:
        target = Checks.non_empty_string(entry_id, "entry_id")
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:  # read and rewrite in one step
            records = self._records()
            for index, (session, stored) in enumerate(records):
                if session == session_id and stored.id_ == target:
                    self._write(records[:index] + records[index + 1 :])
                    return
        raise ValueError(f"entry {target!r} is not stored in session {session_id!r}")

    def entries(self, session_id: str) -> list[Entry]:
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:
            records = self._records()
        return [entry for session, entry in records if session == session_id]

    def get(self, session_id: str, entry_ids: list[str]) -> list[Entry]:
        # A file has no index: the whole file is read, as for entries().
        wanted = [
            Checks.non_empty_string(entry_id, "entry_id") for entry_id in entry_ids
        ]
        return self._pick(session_id, wanted, self.entries(session_id))

    def clear(self, session_id: str) -> None:
        Checks.non_empty_string(session_id, "session_id")
        with self._lock:  # read and rewrite in one step
            records = self._records()
            kept = [
                (session, entry) for session, entry in records if session != session_id
            ]
            if len(kept) < len(records):
                self._write(kept)

    def _records(self) -> list[tuple[str, Entry]]:
        """Every record in the file, of every session, oldest first."""
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        return self.format.records(text)

    def _write(self, records: list[tuple[str, Entry]]) -> None:
        """Replace the file's content in one step, so a crash cannot leave it half written."""
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            file.writelines(
                self.format.block(session, entry) for session, entry in records
            )
        os.replace(temporary, self.path)
