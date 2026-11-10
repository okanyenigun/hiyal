from collections.abc import Iterator
from inspect import isawaitable
from typing import Any, Self
from .types import StreamEventAdapter, StreamFinalizer
from ..schemas import LLMStreamEvent


class AsyncLLMStream:
    """Lazy provider-neutral wrapper for an asynchronous SDK stream.

    It turns the SDK's raw async stream into Hiyal's normalized
    ``LLMStreamEvent`` objects. Nothing is read from the SDK until the caller
    starts iterating, so creating the stream sends no further requests.

    Each ``async for`` step works like this:

    1. Events already produced for the previous raw event are handed out first.
    2. When none are left, the next raw event is awaited from the SDK and passed
       to ``event_adapter``, which returns zero or more normalized events (a raw
       event may carry nothing new, or several parts at once).
    3. When the SDK stream ends, ``finalizer`` (if given) supplies the closing
       events, typically the terminal response built from everything seen, and
       iteration then stops.

    If the SDK raises, the stream is marked exhausted and the error propagates.
    Use it as ``async with`` (or call ``close``) to release the SDK stream when
    stopping early; after closing, no further events are produced.

    Args:
        raw: The SDK's async stream, an async iterator that may have
            ``aclose`` or ``close``.
        event_adapter: Converts one raw event into normalized events; usually
            a ``StreamAccumulator.adapt``.
        finalizer: Returns the events to emit once the raw stream is exhausted;
            usually a ``StreamAccumulator.finalize``.
    """

    def __init__(
        self,
        *,
        raw: Any,
        event_adapter: StreamEventAdapter,
        finalizer: StreamFinalizer | None = None,
    ) -> None:
        """Wrap ``raw`` without reading from it; events are produced as iteration proceeds."""
        self.raw = raw
        self._event_adapter = event_adapter
        self._finalizer = finalizer
        self._exhausted = False
        self._pending_events: Iterator[LLMStreamEvent] = iter(())

    def __aiter__(self) -> Self:
        """Return the stream itself, since it is its own async iterator."""
        return self

    async def __anext__(self) -> LLMStreamEvent:
        """Return the next normalized event, pulling raw SDK events only when none are pending."""
        while True:
            try:
                return next(self._pending_events)
            except StopIteration:
                if self._exhausted:
                    raise StopAsyncIteration
            try:
                raw_event = await self.raw.__anext__()
            except StopAsyncIteration:
                self._exhausted = True
                self._pending_events = iter(
                    self._finalizer() if self._finalizer is not None else ()
                )
            except BaseException:
                self._exhausted = True
                raise
            else:
                self._pending_events = iter(self._event_adapter(raw_event))

    async def __aenter__(self) -> Self:
        """Enter the ``async with`` block and return the stream."""
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Close the stream when the ``async with`` block ends, even after an error."""
        await self.close()

    async def close(self) -> None:
        """Stop producing events and close the underlying SDK stream if it can be closed."""
        self._exhausted = True
        self._pending_events = iter(())
        close = getattr(self.raw, "aclose", None) or getattr(self.raw, "close", None)
        if close is None:
            return
        result = close()
        if isawaitable(result):
            await result
