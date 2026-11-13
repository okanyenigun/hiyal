"""How a ``FileStorage`` lays entries out in its file, one format per extension.

Every format is lossless: each entry is kept as its JSON record, so it can
be read back exactly. ``.txt`` and ``.md`` also show
each entry's ``to_text()``, so a person can read the file; that text is
ignored when reading.

A file holds many sessions, so each record names its session, and the
entry's class name, since the JSON alone does not say which class it was:
``{"session": "abc-123", "type": "LLMResponse", "entry": {...}}``.
"""

import json
from abc import ABC, abstractmethod
from ..entry import Entry, build_entry, entry_type_name


class FileFormat(ABC):
    """Writes entries as blocks of text, and reads them back.

    Each block holds the entry's JSON record (``_encode``); a format only
    decides how the record is laid out around it, and how to find it again
    (``_find``). Reading decodes what it finds (``records``).
    """

    @abstractmethod
    def block(self, session_id: str, entry: Entry) -> str:
        """The text appended to the file for ``entry`` of ``session_id``, ending with a newline."""

    @abstractmethod
    def _find(self, text: str) -> list[str]:
        """The JSON records found in the file's ``text``, oldest first."""

    def records(self, text: str) -> list[tuple[str, Entry]]:
        """Every ``(session_id, entry)`` stored in the file's ``text``, oldest first."""
        return [self._decode(record) for record in self._find(text)]

    @staticmethod
    def _encode(session_id: str, entry: Entry) -> str:
        """One JSON record holding the session, the type name and the entry."""
        name = entry_type_name(entry)
        return f'{{"session":{json.dumps(session_id)},"type":{json.dumps(name)},"entry":{entry.model_dump_json()}}}'

    @staticmethod
    def _decode(record: str) -> tuple[str, Entry]:
        """The session and the entry, rebuilt from what ``_encode`` produced."""
        data = json.loads(record)
        return data["session"], build_entry(data["type"], data["entry"])


class JsonLinesFormat(FileFormat):
    """``.jsonl``: one encoded record per line, nothing else."""

    def block(self, session_id: str, entry: Entry) -> str:
        return self._encode(session_id, entry) + "\n"

    def _find(self, text: str) -> list[str]:
        return [line for line in (raw.strip() for raw in text.splitlines()) if line]


class TextFormat(FileFormat):
    """``.txt``: a header, the readable text indented, then the record.

    ::

        === UserPrompt · 7de59230-... ===
            User: Is this hotel near the station?
        @entry {"session":"abc-123","type":"UserPrompt","entry":{...}}

    The readable lines are indented, so none can start with ``@entry``.
    """

    PREFIX = "@entry "

    def block(self, session_id: str, entry: Entry) -> str:
        record = self._encode(
            session_id, entry
        )  # first: it rejects anything that is not an entry
        readable = "\n".join(f"    {line}" for line in entry.to_text().splitlines())
        return f"=== {type(entry).__name__} · {entry.id_} ===\n{readable}\n{self.PREFIX}{record}\n\n"

    def _find(self, text: str) -> list[str]:
        return [
            line[len(self.PREFIX) :]
            for line in text.splitlines()
            if line.startswith(self.PREFIX)
        ]


class MarkdownFormat(FileFormat):
    """``.md``: a heading, the readable text as a quote, then the record in a comment.

    ::

        ## UserPrompt · `7de59230-...`

        > User: Is this hotel near the station?

        <!-- hiyal-entry {"session":"abc-123","type":"UserPrompt","entry":{...}} -->

    The comment hides the record when the Markdown is rendered. Every ``>``
    in the record is written as ``\\u003e`` (the same character in JSON), so
    it cannot close the comment early. The readable lines are quoted, so
    none can start with the comment marker.
    """

    START = "<!-- hiyal-entry "
    END = " -->"

    def block(self, session_id: str, entry: Entry) -> str:
        record = self._encode(session_id, entry).replace(
            ">", "\\u003e"
        )  # first: it rejects anything that is not an entry
        readable = "\n".join(
            f"> {line}" if line else ">" for line in entry.to_text().splitlines()
        )
        return f"## {type(entry).__name__} · `{entry.id_}`\n\n{readable}\n\n{self.START}{record}{self.END}\n\n"

    def _find(self, text: str) -> list[str]:
        return [
            line[len(self.START) : -len(self.END)]
            for line in text.splitlines()
            if line.startswith(self.START) and line.endswith(self.END)
        ]


# The format for each supported file extension.
FORMATS: dict[str, FileFormat] = {
    ".jsonl": JsonLinesFormat(),
    ".txt": TextFormat(),
    ".md": MarkdownFormat(),
}
