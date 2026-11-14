"""What a history holds, in order: prompts, responses, added context, tool results,
a summary and working notes.

Entries carry no links to each other: their order gives them meaning. A
response answers everything added since the previous response, so the
entries before it (a system prompt, context, the user prompt) are its input.
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Self
from uuid import uuid4
from pydantic import BaseModel, Field, JsonValue, model_validator
from ...models.schemas import LLMResponse
from ...prompts import Audio, Document, Image, SystemPrompt, UserPrompt
from ...utility import Tokens
from ...utility.tokens import DEFAULT_ENCODING

# Tokens counted for each attached image, audio clip and document, which
# an entry's text shows only as a short placeholder. Rough, fixed estimates
# (LlamaIndex's); the real cost depends on the model, the size and the pages.
MEDIA_TOKENS: Mapping[str, int] = MappingProxyType(
    {"image": 256, "audio": 256, "document": 2048}
)


class ContextEntry(BaseModel):
    """Content added to memory that is neither a model input nor a response.

    For example, facts loaded from long-term memory. It stays in the history
    like any other entry and is windowed by the policy like the rest.

    Note: at least one of ``text``, ``data``, ``images``, ``audio`` or
    ``documents`` must be given.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    text: str | None = Field(default=None, description="Text content, if any.")
    data: dict[str, JsonValue] | None = Field(
        default=None,
        description="Structured content as a JSON object (a Python dict of JSON-compatible values).",
        examples=[{"budget": 3000, "currency": "USD", "direct_flights_only": True}],
    )
    images: list[Image] = Field(default_factory=list, description="Images.")
    audio: list[Audio] = Field(default_factory=list, description="Audio clips.")
    documents: list[Document] = Field(
        default_factory=list, description="Document files (PDF, DOCX, CSV, ...)."
    )
    source: str | None = Field(
        default=None,
        description="Where the content came from.",
        examples=["long_term_memory"],
    )

    def to_text(self) -> str:
        """The content as plain text for a model to read.

        A ``Context`` header with the source, then the text, the data as
        JSON, and media as placeholders.
        """
        lines = [f"Context ({self.source}):" if self.source else "Context:"]
        if self.text is not None:
            lines.append(self.text)
        if self.data is not None:
            lines.append(json.dumps(self.data, ensure_ascii=False))
        lines += [
            media.placeholder()
            for media in [*self.images, *self.audio, *self.documents]
        ]
        return "\n".join(lines)

    @model_validator(mode="after")
    def _require_content(self) -> Self:
        """At least one of text, data, images, audio or documents must be given."""
        if (
            self.text is None
            and self.data is None
            and not (self.images or self.audio or self.documents)
        ):
            raise ValueError(
                "give at least one of text, data, images, audio or documents"
            )
        return self


class ToolResultEntry(BaseModel):
    """The application's answer to one tool call the model requested."""

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    message_id: str = Field(
        min_length=1,
        description="ResponseMessage.id of the tool call this answers, inside its LLMResponse.",
    )
    call_id: str | None = Field(
        default=None,
        description=(
            "Provider correlation ID of the call, needed to send the result back. "
            "Stored here because ToolCall.raw is not serialized. None for providers "
            "without one, such as Ollama."
        ),
        examples=["call_123", "toolu_01A"],
    )
    name: str = Field(
        min_length=1, description="Function that ran.", examples=["get_weather"]
    )
    arguments: dict[str, JsonValue] = Field(
        description=(
            "Arguments the function was called with, as a JSON object; {} for a "
            "function without parameters."
        ),
        examples=[{"city": "Kyoto", "month": "April"}, {}],
    )
    output: JsonValue = Field(
        description="What the function returned; JSON null is valid."
    )
    is_error: bool = Field(default=False, description="Whether the function failed.")

    def to_text(self) -> str:
        """The result as plain text for a model to read.

        The call with its arguments, then what it returned, e.g.
        ``Tool result: get_weather({"city": "Kyoto"}) -> Mild``, or
        ``Tool error: ...`` when the function failed.
        """
        arguments = json.dumps(self.arguments, ensure_ascii=False)
        output = (
            self.output
            if isinstance(self.output, str)
            else json.dumps(self.output, ensure_ascii=False)
        )
        label = "Tool error" if self.is_error else "Tool result"
        return f"{label}: {self.name}({arguments}) -> {output}"


class SummaryEntry(BaseModel):
    """A summary standing in for earlier entries.

    The storage keeps every summary, but only the latest one counts: it
    replaces the previous one, and leads ``active``. It is never in the
    window, so policies never drop it, and the entries it covers never
    return to the window.

    What it covers is kept short, so it does not grow with the history:
    every entry stored up to ``covers_through``, except ``covers_except``
    (the few that were never summarized: pinned, filtered, system prompts).
    Summaries and notes are never covered, so they are not listed there.
    ``covered()`` turns that back into the IDs.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    text: str = Field(min_length=1, description="The summary.")
    covers_through: str = Field(
        min_length=1,
        description="ID of the newest entry this summary covers.",
    )
    covers_except: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the entries stored before covers_through that it does not "
            "cover, oldest first; summaries and notes are not listed."
        ),
    )
    covers_count: int = Field(
        ge=1,
        description=(
            "How many entries it covers. covered() finding fewer means some "
            "were removed from the storage."
        ),
    )
    version: int = Field(
        default=1,
        ge=1,
        description=(
            "1 for the session's first summary, one more for each that replaces "
            "the previous; HybridCompaction rebuilds by it."
        ),
    )

    def covered(self, originals: list[str]) -> list[str]:
        """The IDs this summary covers, out of ``originals``.

        ``originals`` are the IDs of the session's entries in history order,
        summaries and notes left out. Returns those up to ``covers_through``,
        except ``covers_except``; none if ``covers_through`` is not among
        them (removed from the storage), since where the summary ends can no
        longer be told.
        """
        try:
            end = originals.index(self.covers_through) + 1
        except ValueError:
            return []
        skipped = set(self.covers_except)
        return [entry_id for entry_id in originals[:end] if entry_id not in skipped]

    def to_text(self) -> str:
        """The summary as plain text for a model to read."""
        return f"Summary of the earlier conversation:\n{self.text}"


class WorkingNotesEntry(BaseModel):
    """A version of the working notes.

    The notes are a plain JSON object: one key per section, e.g.
    ``{"destination": "Kyoto", "constraints": ["Direct flights only"]}``. A
    Pydantic model only matters when an LLM generates them (see
    ``NoteTaker``), to describe the sections and check its answer; as
    stored data they need no class.

    Like the summary, the storage keeps every version, but only the latest
    one counts: it comes right after the summary in ``active``, and is never
    in the window, so policies never drop it. Unlike the summary, it covers
    no other entries.

    A version comes from the note taker (an LLM update, from the entries
    since the previous one) or from the app (``set_notes``, one section set
    directly); ``source`` says which.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    notes: dict[str, JsonValue] = Field(
        description="The notes: one key per section, any JSON value.",
        examples=[
            {
                "destination": "Kyoto",
                "constraints": ["Direct flights only"],
                "next_step": None,
            }
        ],
    )
    source: Literal["note_taker", "app"] = Field(
        default="note_taker",
        description=(
            "Who wrote this version: the note taker, from the entries since its "
            "previous update, or the app, setting one section directly."
        ),
    )

    def to_text(self) -> str:
        """The notes as plain text for a model to read: one bullet per section.

        Section names come from the keys (``next_step`` becomes
        ``Next step``); lists become nested bullets, other structured values
        JSON, and empty sections ``(none)``.
        """
        lines = ["Working notes:"]
        for name, value in self.notes.items():
            title = name.replace("_", " ").capitalize()
            if isinstance(value, list) and value:
                lines.append(f"- {title}:")
                lines += [f"  - {self._value_text(item)}" for item in value]
            else:
                lines.append(f"- {title}: {self._value_text(value)}")
        return "\n".join(lines)

    @staticmethod
    def _value_text(value: Any) -> str:
        """One notes value as text: strings as they are, empty values ``(none)``, the rest JSON."""
        if value is None or value == [] or value == {} or value == "":
            return "(none)"
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )


class RewindEntry(BaseModel):
    """A marker: the session went back to just after the entry ``to_id``.

    Nothing is removed from the storage; this marker says that what was
    stored after that entry, from the next entry added on, no longer counts
    (the summary and notes made right after it still do). Reading the
    history applies each marker in order (see
    ``ShortTermMemoryManager.rewind``). It is never in the window, never
    summarized, and never given to the note taker.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    to_id: str = Field(
        min_length=1, description="ID of the entry the session went back to."
    )

    def to_text(self) -> str:
        """The marker as plain text: ``Rewound to entry <to_id>``."""
        return f"Rewound to entry {self.to_id}"


Entry = (
    SystemPrompt
    | UserPrompt
    | LLMResponse
    | ContextEntry
    | ToolResultEntry
    | SummaryEntry
    | WorkingNotesEntry
    | RewindEntry
)

# Every entry class, by its name: what a storage keeps next to an entry's JSON,
# since the JSON alone does not say which class it was.
ENTRY_TYPES: dict[str, type[Entry]] = {
    cls.__name__: cls
    for cls in (
        SystemPrompt,
        UserPrompt,
        LLMResponse,
        ContextEntry,
        ToolResultEntry,
        SummaryEntry,
        WorkingNotesEntry,
        RewindEntry,
    )
}


def entry_type_name(entry: Entry) -> str:
    """The name of ``entry``'s class, as ``ENTRY_TYPES`` knows it; raises ``TypeError`` for a non-entry."""
    name = type(entry).__name__
    if ENTRY_TYPES.get(name) is not type(entry):
        raise TypeError(f"cannot store {name}; entries are {', '.join(ENTRY_TYPES)}")
    return name


def build_entry(name: str, data: dict[str, Any]) -> Entry:
    """Rebuild an entry from its class name and its JSON-decoded fields."""
    cls = ENTRY_TYPES.get(name)
    if cls is None:
        raise ValueError(f"unknown entry type {name!r}")
    return cls.model_validate(data)


def finishes_turn(entry: Entry) -> bool:
    """Whether ``entry`` is the model's answer that ends a turn: an ``LLMResponse`` with no tool calls."""
    return isinstance(entry, LLMResponse) and not entry.tool_calls


def is_derived(entry: Entry) -> bool:
    """Whether ``entry`` was made by a worker (a summary or notes), not added to the conversation."""
    return isinstance(entry, (SummaryEntry, WorkingNotesEntry))


def media_count(entry: Entry) -> dict[str, int]:
    """How many images, audio clips and documents ``entry`` carries, as ``{"image": n, "audio": n, "document": n}``."""
    if isinstance(entry, (UserPrompt, ContextEntry)):
        return {
            "image": len(entry.images),
            "audio": len(entry.audio),
            "document": len(entry.documents),
        }
    if isinstance(entry, LLMResponse):
        return {"image": len(entry.images), "audio": len(entry.audio), "document": 0}
    return {"image": 0, "audio": 0, "document": 0}


def check_media_tokens(media_tokens: Mapping[str, int]) -> Mapping[str, int]:
    """``media_tokens`` when its keys are among ``MEDIA_TOKENS``'s and its values non-negative integers; raises ``ValueError`` otherwise."""
    unknown = set(media_tokens) - set(MEDIA_TOKENS)
    if unknown:
        raise ValueError(
            f"media_tokens keys must be among {sorted(MEDIA_TOKENS)}, got {sorted(unknown)}"
        )
    for kind, tokens in media_tokens.items():
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError(
                f"media_tokens[{kind!r}] must be a non-negative integer, got {tokens!r}"
            )
    return media_tokens


def entry_tokens(
    entry: Entry,
    encoding: str = DEFAULT_ENCODING,
    media_tokens: Mapping[str, int] = MEDIA_TOKENS,
) -> int:
    """``entry``'s tokens: its ``to_text()`` counted with ``encoding``, plus the estimate in ``media_tokens`` for each attached medium (a kind left out counts 0)."""
    media = sum(
        media_tokens.get(kind, 0) * count for kind, count in media_count(entry).items()
    )
    return Tokens.count(entry.to_text(), encoding) + media
