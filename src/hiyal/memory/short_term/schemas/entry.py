"""The entry: the smallest unit of short-term memory, one envelope around one piece of content."""

from enum import StrEnum
from typing import Any, Self
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .payloads import Context, Notes, Pin, Summary, ToolResult
from ....models.schemas import LLMResponse
from ....prompts import SystemPrompt, UserPrompt


class EntryKind(StrEnum):
    """What an entry holds."""

    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"
    LLM_RESPONSE = "llm_response"
    TOOL_RESULT = "tool_result"
    CONTEXT = "context"
    SUMMARY = "summary"
    NOTES = "notes"
    PIN = "pin"


# What an entry can hold.
Payload = (
    SystemPrompt
    | UserPrompt
    | LLMResponse
    | ToolResult
    | Context
    | Summary
    | Notes
    | Pin
)

# The payload class each kind holds, and the other way round.
PAYLOADS: dict[EntryKind, type[BaseModel]] = {
    EntryKind.SYSTEM_PROMPT: SystemPrompt,
    EntryKind.USER_PROMPT: UserPrompt,
    EntryKind.LLM_RESPONSE: LLMResponse,
    EntryKind.TOOL_RESULT: ToolResult,
    EntryKind.CONTEXT: Context,
    EntryKind.SUMMARY: Summary,
    EntryKind.NOTES: Notes,
    EntryKind.PIN: Pin,
}
KINDS: dict[type[BaseModel], EntryKind] = {cls: kind for kind, cls in PAYLOADS.items()}

# The kinds only the manager creates: records about the history, not part of it.
MANAGED_KINDS = frozenset({EntryKind.SUMMARY, EntryKind.NOTES, EntryKind.PIN})


class Entry(BaseModel):
    """One unit of short-term memory: an envelope with an ID, a kind and a turn, around a typed payload.

    The envelope is the same for every entry, so storages and policies
    handle one shape. The payload keeps its own class, chosen by ``kind``
    (``PAYLOADS``), so its fields stay validated and its methods (a
    response's tool calls, a prompt's images) stay available.

    ``turn_id`` ties the entries of one turn together: a user's input, the
    model's tool calls, their results, and the final answer. Entries
    outside a turn, such as the system prompt or context added between
    turns, have none. The app creates entries and turns; the memory takes
    them as they are.

    Build one with ``Entry.of(payload, turn_id=...)``, which takes the kind
    from the payload's class, or give ``kind`` yourself; a kind that does
    not match the payload is refused.
    """

    # Bytes in an untyped payload field (a response's image or audio
    # output) are written as base64 in JSON; they follow the outermost model.
    model_config = ConfigDict(ser_json_bytes="base64")

    id_: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Entry ID, generated automatically. Never empty.",
    )
    kind: EntryKind = Field(description="What the entry holds.")
    turn_id: str | None = Field(
        default=None,
        min_length=1,
        description="The turn the entry belongs to; None outside any turn.",
    )
    payload: Payload = Field(description="The content, of the class ``kind`` names.")

    @classmethod
    def of(cls, payload: Payload, turn_id: str | None = None) -> Self:
        """An entry holding ``payload``, its kind taken from the payload's class."""
        kind = KINDS.get(type(payload))
        if kind is None:
            raise TypeError(f"an entry cannot hold a {type(payload).__name__}")
        return cls(kind=kind, payload=payload, turn_id=turn_id)

    @model_validator(mode="before")
    @classmethod
    def _payload_by_kind(cls, data: Any) -> Any:
        """Rebuild a payload given as plain data (e.g. read back from JSON) as its kind's class.

        Without this, plain data could be read as the wrong class: a
        summary's ``{"text": ...}`` would also fit a ``SystemPrompt``.
        """
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            kind = data.get("kind")
            if kind is not None:
                payload_class = PAYLOADS[EntryKind(kind)]
                data = {
                    **data,
                    "payload": payload_class.model_validate(data["payload"]),
                }
        return data

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> Self:
        """Refuse a payload whose class is not the one ``kind`` names."""
        expected = PAYLOADS[self.kind]
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"a {self.kind.value} entry holds a {expected.__name__}, "
                f"not a {type(self.payload).__name__}"
            )
        return self

    def to_text(self) -> str:
        """The payload as plain text for a model to read."""
        return self.payload.to_text()
