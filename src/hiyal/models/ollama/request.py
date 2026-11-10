"""Ollama request helpers: the endpoint marker, plus request-local HTTP options
and retries, which Ollama's chat API lacks."""

import asyncio
from time import sleep
from typing import Any, Callable

# Marks a request for Ollama's generate endpoint (image generation) instead of chat.
ENDPOINT_KEY = "_hiyal_endpoint"


def _retryable(error: Exception) -> bool:
    import httpx
    from ollama import ResponseError

    return isinstance(
        error,
        (
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ) or (
        isinstance(error, ResponseError)
        and (error.status_code in {408, 409, 429} or 500 <= error.status_code < 600)
    )


def _delay(attempt: int) -> float:
    return 0.25 * 2 ** min(attempt, 3)


def request_with_retries(
    request: Callable[..., Any],
    retries: int,
    *args: Any,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    if stream:
        return _stream(request, retries, *args, **kwargs)
    for attempt in range(retries + 1):
        try:
            return request(*args, stream=False, **kwargs)
        except Exception as error:
            if attempt == retries or not _retryable(error):
                raise
            sleep(_delay(attempt))


def _stream(request: Callable[..., Any], retries: int, *args: Any, **kwargs: Any):
    for attempt in range(retries + 1):
        raw = None
        emitted = False
        try:
            raw = request(*args, stream=True, **kwargs)
            for item in raw:
                emitted = True
                yield item
            return
        except Exception as error:
            # Never replay a stream after any output has reached its consumer.
            if emitted or attempt == retries or not _retryable(error):
                raise
        finally:
            close = getattr(raw, "close", None)
            if close is not None:
                close()
        sleep(_delay(attempt))


async def request_with_retries_async(
    request: Callable[..., Any],
    retries: int,
    *args: Any,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    if stream:
        return _stream_async(request, retries, *args, **kwargs)
    for attempt in range(retries + 1):
        try:
            return await request(*args, stream=False, **kwargs)
        except Exception as error:
            if attempt == retries or not _retryable(error):
                raise
            await asyncio.sleep(_delay(attempt))


async def _stream_async(
    request: Callable[..., Any], retries: int, *args: Any, **kwargs: Any
):
    for attempt in range(retries + 1):
        raw = None
        emitted = False
        try:
            raw = await request(*args, stream=True, **kwargs)
            async for item in raw:
                emitted = True
                yield item
            return
        except Exception as error:
            if emitted or attempt == retries or not _retryable(error):
                raise
        finally:
            close = getattr(raw, "aclose", None)
            if close is not None:
                await close()
        await asyncio.sleep(_delay(attempt))
