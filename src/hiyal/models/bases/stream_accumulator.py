from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar
from ..schemas import (
    ContentDeltaEvent,
    LLMResponse,
    LLMStreamEvent,
    ResponseContent,
    ResponseTerminalEvent,
    ToolCallContent,
    ToolCallDeltaContent,
)
from ...tools.calls.responses import ToolCallDelta

if TYPE_CHECKING:
    from .translator import Translator

TranslatorT = TypeVar("TranslatorT", bound="Translator")
TerminalEventType = Literal[
    "response.completed", "response.incomplete", "response.failed"
]
_TERMINAL_EVENT_TYPES: dict[str, TerminalEventType] = {
    "completed": "response.completed",
    "incomplete": "response.incomplete",
    "failed": "response.failed",
}


class StreamAccumulator(ABC, Generic[TranslatorT]):
    """Turns one streamed request's raw SDK events into normalized stream events.

    A provider streams its answer as many small raw events (a few words of
    text, a piece of a tool call, a usage report). The accumulator is the
    memory for that one request: it is created fresh for each streamed call
    and discarded when the stream ends. ``LLMStream`` and ``AsyncLLMStream``
    drive it:

    - ``adapt`` is called once per raw event. It records the event and its
      position, then the provider's ``_adapt`` parses it into zero or more
      ``LLMStreamEvent`` objects, usually content deltas.
    - ``finalize`` is called once the raw stream ends. Providers use it to
      emit the terminal event holding the complete ``LLMResponse``, built from
      everything accumulated, so streamed and non-streamed results match.

    The base class supplies what every provider shares: the event counter and
    the ``_delta`` and ``_terminal`` helpers that build the common event
    shapes. A subclass adds its own bookkeeping (buffered text, partial tool
    calls, usage, finish reason) and implements ``_adapt``.

    It depends on the provider's ``Translator``, not on the whole LLM client; a
    subclass declares which one, e.g. ``StreamAccumulator[OpenAITranslator]``.

    Args:
        translator: The provider's translator, used to parse parts and build
            the final response.
        model: The requested model name, used until the stream reports its own.
    """

    def __init__(self, *, translator: TranslatorT, model: str | None = None) -> None:
        """Start an empty accumulator for one request, before any raw event has arrived."""
        self.translator = translator
        self.model = model
        self.response_id: str | None = None
        self.raw: Any = None
        self.sequence_number = -1

    def adapt(self, event: Any) -> Iterable[LLMStreamEvent]:
        """Record one raw SDK event and return the normalized events it produces."""
        self.raw = event
        self.sequence_number += 1
        return self._adapt(event)

    @abstractmethod
    def _adapt(self, event: Any) -> Iterable[LLMStreamEvent]:
        """Parse one raw event into zero or more normalized events, updating the provider's state."""

    def finalize(self) -> Iterable[LLMStreamEvent]:
        """Events emitted once the raw stream is exhausted; none by default."""
        return ()

    def _delta(
        self, *, message_index: int, message_id: str, content: ResponseContent
    ) -> ContentDeltaEvent:
        """Build the content-delta event for one new fragment of a message."""
        if isinstance(content, ToolCallContent):
            content = ToolCallDeltaContent(
                data=ToolCallDelta(
                    name=content.data.name,
                    arguments=content.data.arguments,
                )
            )
        return ContentDeltaEvent(
            event_type="content.delta",
            provider=self.translator.info.provider,
            sequence_number=self.sequence_number,
            response_id=self.response_id,
            # Each part is one message holding a single content part.
            message_index=message_index,
            message_id=message_id,
            content_index=0,
            delta=content,
            raw=self.raw,
        )

    def _terminal(self, response: LLMResponse) -> tuple[ResponseTerminalEvent]:
        """Build the closing event that carries the complete response."""
        return (
            ResponseTerminalEvent(
                provider=self.translator.info.provider,
                sequence_number=None
                if self.sequence_number < 0
                else self.sequence_number,
                response_id=self.response_id,
                raw=self.raw,
                event_type=_TERMINAL_EVENT_TYPES[response.status],
                status=response.status,
                response=response,
            ),
        )
