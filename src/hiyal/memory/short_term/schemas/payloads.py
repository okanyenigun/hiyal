"""The content classes short-term memory adds to the prompts and responses an entry can hold.

An ``Entry`` wraps one payload. Prompts (``SystemPrompt``, ``UserPrompt``)
and responses (``LLMResponse``) come from the rest of Hiyal; the classes
here are the memory's own: a tool's result, added context, a summary, and
working notes.
"""

import json
from typing import Any, Literal, Self
from pydantic import BaseModel, Field, JsonValue, model_validator
from ....prompts import Audio, Document, Image


class ToolResult(BaseModel):
    """The application's answer to one tool call the model requested."""

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
        """The result as plain text, e.g. ``Tool result: get_weather({"city": "Kyoto"}) -> Mild``.

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


class Context(BaseModel):
    """Content added to memory that is neither a model input nor a response.

    For example, facts loaded from long-term memory. At least one of
    ``text``, ``data``, ``images``, ``audio`` or ``documents`` must be given.
    """

    text: str | None = Field(default=None, description="Text content, if any.")
    data: dict[str, JsonValue] | None = Field(
        default=None,
        description="Structured content as a JSON object.",
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

    def to_text(self) -> str:
        """A ``Context`` header with the source, then the text, the data as JSON, and media as placeholders."""
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


class Summary(BaseModel):
    """A summary standing in for earlier entries: one version, made from the previous version and the entries that left since.

    The text covers everything summarized so far, but ``newly_covered``
    lists only the entries this version added, so each ID is stored once.
    What a session's summary covers is what all its versions list.
    """

    text: str = Field(min_length=1, description="The summary.")
    newly_covered: list[str] = Field(
        default_factory=list,
        description="IDs of the entries this version summarized, on top of what the earlier versions cover.",
    )

    def to_text(self) -> str:
        """The summary as plain text: ``Summary of the earlier conversation:`` and the text."""
        return f"Summary of the earlier conversation:\n{self.text}"


class Notes(BaseModel):
    """A version of the working notes: one key per section, any JSON value.

    For example ``{"destination": "Kyoto", "constraints": ["Direct flights only"]}``.
    """

    sections: dict[str, JsonValue] = Field(
        description="The notes: one key per section, any JSON value.",
        examples=[{"destination": "Kyoto", "constraints": ["Direct flights only"]}],
    )
    source: Literal["llm", "app"] = Field(
        default="llm",
        description=(
            "Who made this version: the LLM, from the entries since its last "
            "update, or the app, setting one section."
        ),
    )

    def to_text(self) -> str:
        """One bullet per section: ``next_step`` becomes ``Next step``, lists nested bullets, empty values ``(none)``."""
        lines = ["Working notes:"]
        for name, value in self.sections.items():
            title = name.replace("_", " ").capitalize()
            if isinstance(value, list) and value:
                lines.append(f"- {title}:")
                lines += [f"  - {self._value_text(item)}" for item in value]
            else:
                lines.append(f"- {title}: {self._value_text(value)}")
        return "\n".join(lines)

    @staticmethod
    def _value_text(value: Any) -> str:
        """One value as text: strings as they are, empty values ``(none)``, the rest JSON."""
        if value is None or value == [] or value == {} or value == "":
            return "(none)"
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )


class Pin(BaseModel):
    """A record that an entry is pinned: kept in what goes to the model, whatever the policy chooses.

    Stored in the session right after the entry, so a manager opened on the
    session knows the entry is pinned.
    """

    entry_id: str = Field(min_length=1, description="ID of the pinned entry.")

    def to_text(self) -> str:
        """``Pinned entry: <ID>``."""
        return f"Pinned entry: {self.entry_id}"
