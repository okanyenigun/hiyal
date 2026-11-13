"""What a history holds, in order: prompts, responses, added context, tool results,
a summary and working notes.

Entries carry no links to each other: their order gives them meaning. A
response answers everything added since the previous response, so the
entries before it (a system prompt, context, the user prompt) are its input.
"""

import json
from typing import Any, Self
from uuid import uuid4
from pydantic import BaseModel, Field, JsonValue, model_validator
from ...models.schemas import LLMResponse
from ...prompts import Audio, Document, Image, SystemPrompt, UserPrompt


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
    """

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    text: str = Field(min_length=1, description="The summary.")
    covers: list[str] = Field(
        min_length=1,
        description="IDs of the entries this summary stands in for, oldest first.",
    )

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


Entry = (
    SystemPrompt
    | UserPrompt
    | LLMResponse
    | ContextEntry
    | ToolResultEntry
    | SummaryEntry
    | WorkingNotesEntry
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
