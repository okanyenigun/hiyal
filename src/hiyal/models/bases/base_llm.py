from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from ...prompts.media import Media, as_media
from ..schemas import LLMConfig, LLMResponse, Provider, ProviderInfo
from .async_llm_stream import AsyncLLMStream
from .config_resolver import ConfigResolver
from .llm_stream import LLMStream
from .transport import ConnectionOptions, FileStore, Transport
from .translator import Translator


@dataclass(frozen=True)
class _PreparedCall:
    """Everything one query needs before it is sent."""

    config: LLMConfig
    request: dict[str, Any]
    connection: ConnectionOptions


class BaseLLM(ABC):
    """Base class for provider-backed language models.

    ``BaseLLM`` orchestrates a query and delegates the work to three helpers:

    - ``resolver`` (``ConfigResolver``, shared by all providers) merges
      constructor and per-call settings into validated ``LLMConfig`` objects.
    - ``translator`` (one ``Translator`` per provider) turns a config into SDK
      arguments and SDK output back into ``LLMResponse`` objects and events.
    - ``transport`` (one ``Transport`` per provider) owns the SDK client,
      sends requests, applies per-call connection settings and exposes the
      provider's Files API, if any.

    A provider subclass declares the class attributes below and implements
    ``_create_translator`` and ``_create_transport``.

    **Class attributes**

    - ``provider``: the ``Provider`` identifier, e.g. ``"openai"``.
    - ``display_name``: the provider's name as shown in error messages.
    - ``prompt_media``: which ``UserPrompt`` attachment kinds the provider
      accepts, out of ``"images"``, ``"audio"`` and ``"documents"``. Any other
      kind raises ``NotImplementedError`` instead of being dropped. The
      default, an empty set, means text only.
    - ``generation_adapter``: renames per-query settings between the provider
      SDK and ``LLMConfig``, as ``{"sdk_name": "config_name"}``. For example,
      ``{"system": "system_prompt"}`` lets callers pass ``system=...``; it is
      stored as ``system_prompt`` and sent as ``system`` again. Either
      direction may be written.
    - ``adapter``: the same renaming for client-level settings, accepted only
      when the client is created. Most providers leave it empty.

    **Configuration**

    If a value is set both in ``config`` and as a keyword argument, ``config``
    wins. Keywords that are not ``LLMConfig`` fields are kept as provider
    extras (``client_options`` on the client, ``others`` on a query). Each
    query starts from an independent copy of the client's settings.
    """

    provider: ClassVar[Provider]
    display_name: ClassVar[str]
    prompt_media: ClassVar[frozenset[str]] = frozenset()
    adapter: ClassVar[dict[str, str]] = {}
    generation_adapter: ClassVar[dict[str, str]] = {}

    def __init__(self, *, config: LLMConfig | None = None, **parameters: Any) -> None:
        """Build the client's default config, then the provider's translator and transport."""
        self.info = self._provider_info()
        self.resolver = ConfigResolver(self.info)
        self.config = self.resolver.client_config(config, parameters)
        self.translator = self._create_translator()
        self.transport = self._create_transport()

    @classmethod
    def _provider_info(cls) -> ProviderInfo:
        """Collect the subclass's class attributes, raising if a required one is missing."""
        missing = [name for name in ("provider", "display_name") if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must define {' and '.join(missing)}")
        return ProviderInfo(
            provider=cls.provider,
            display_name=cls.display_name,
            prompt_media=cls.prompt_media,
            adapter=cls.adapter,
            generation_adapter=cls.generation_adapter,
        )

    @abstractmethod
    def _create_translator(self) -> Translator:
        """Build this provider's translator from ``self.info``."""

    @abstractmethod
    def _create_transport(self) -> Transport:
        """Build this provider's transport from ``self.config`` and ``self.info``."""

    @property
    def client(self) -> Any:
        """The provider SDK client owned by the transport."""
        return self.transport.client

    # Queries ---------------------------------------------------------------

    def query(self, *, config: LLMConfig | None = None, **parameters: Any) -> LLMResponse:
        """Send one blocking request and return the parsed response."""
        call = self._prepare(config, parameters, asynchronous=False, alternative="query_async")
        created_at, started = datetime.now(timezone.utc), perf_counter()
        raw = self.transport.send(call.request, stream=False, connection=call.connection)
        return self._parse(raw, call, created_at=created_at, started=started)

    async def query_async(
        self, *, config: LLMConfig | None = None, **parameters: Any
    ) -> LLMResponse:
        """Send one request on an async client and return the parsed response."""
        call = self._prepare(config, parameters, asynchronous=True, alternative="query")
        created_at, started = datetime.now(timezone.utc), perf_counter()
        raw = await self.transport.send_async(call.request, stream=False, connection=call.connection)
        return self._parse(raw, call, created_at=created_at, started=started)

    def query_stream(self, *, config: LLMConfig | None = None, **parameters: Any) -> LLMStream:
        """Start a blocking streaming request and return a stream of normalized events."""
        call = self._prepare(
            config, parameters, asynchronous=False, alternative="query_stream_async"
        )
        accumulator = self.translator.stream_accumulator(call.config)
        raw = self.transport.send(call.request, stream=True, connection=call.connection)
        return LLMStream(raw=raw, event_adapter=accumulator.adapt, finalizer=accumulator.finalize)

    async def query_stream_async(
        self, *, config: LLMConfig | None = None, **parameters: Any
    ) -> AsyncLLMStream:
        """Start an async streaming request and return a stream of normalized events."""
        call = self._prepare(config, parameters, asynchronous=True, alternative="query_stream")
        accumulator = self.translator.stream_accumulator(call.config)
        raw = await self.transport.send_async(call.request, stream=True, connection=call.connection)
        return AsyncLLMStream(
            raw=raw, event_adapter=accumulator.adapt, finalizer=accumulator.finalize
        )

    def _prepare(
        self,
        config: LLMConfig | None,
        parameters: dict[str, Any],
        *,
        asynchronous: bool,
        alternative: str,
    ) -> _PreparedCall:
        """Check the call mode, resolve and validate the config, and build the request."""
        self._require_mode(asynchronous=asynchronous, alternative=alternative)
        resolved = self.resolver.call_config(self.config, config, parameters)
        self.translator.validate(resolved)
        return _PreparedCall(
            config=resolved,
            request=self.translator.build_request(resolved),
            connection=self.transport.connection_overrides(resolved),
        )

    def _parse(
        self, raw: Any, call: _PreparedCall, *, created_at: datetime, started: float
    ) -> LLMResponse:
        """Hand the raw SDK response to the translator with the client-side timing."""
        return self.translator.parse_response(
            raw,
            model=call.config.model,
            created_at=created_at,
            elapsed_seconds=perf_counter() - started,
        )

    def _require_mode(self, *, asynchronous: bool, alternative: str) -> None:
        """Raise when a sync method is used on an async-only client, or the reverse."""
        if not self.transport.supports(asynchronous=asynchronous):
            mode = "asynchronous" if self.config.asynchronous else "synchronous"
            raise TypeError(f"this client is {mode}; use {alternative}")

    # Files API -------------------------------------------------------------

    def upload_file(
        self, file: str | Path | bytes | Media, mime_type: str | None = None,
        *, filename: str | None = None,
    ) -> Media:
        """Upload a file to the provider's Files API.

        Returns an ``Image``, ``Audio`` or ``Document`` (chosen by MIME type)
        that references the upload; put it in a ``UserPrompt`` to reuse the
        file across requests without resending it. Raises when the provider
        has no Files API.
        """
        source = as_media(file, mime_type, filename)
        files = self._files(asynchronous=False, alternative="upload_file_async")
        return files.to_media(files.wait_until_ready(files.upload(source)), source)

    async def upload_file_async(
        self, file: str | Path | bytes | Media, mime_type: str | None = None,
        *, filename: str | None = None,
    ) -> Media:
        """Async version of ``upload_file``."""
        source = as_media(file, mime_type, filename)
        files = self._files(asynchronous=True, alternative="upload_file")
        raw = await files.upload_async(source)
        return files.to_media(await files.wait_until_ready_async(raw), source)

    def delete_file(self, file: Media | str) -> None:
        """Delete an uploaded file, given the media ``upload_file`` returned or its ID."""
        self._files(asynchronous=False, alternative="delete_file_async").delete(file)

    async def delete_file_async(self, file: Media | str) -> None:
        """Async version of ``delete_file``."""
        await self._files(asynchronous=True, alternative="delete_file").delete_async(file)

    def _files(self, *, asynchronous: bool, alternative: str) -> FileStore:
        """The provider's Files API, raising when it has none or the mode is wrong."""
        files = self.transport.files
        if files is None:
            raise NotImplementedError(
                f"{self.display_name} has no Files API in Hiyal; pass the file in UserPrompt instead"
            )
        self._require_mode(asynchronous=asynchronous, alternative=alternative)
        return files
