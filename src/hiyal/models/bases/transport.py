from abc import ABC, abstractmethod
from typing import Any, TypedDict, cast
from ..schemas import LLMConfig, ProviderInfo
from ..schemas.fields import CONNECTION_FIELDS
from ...prompts.media import Media
from ...utility import Dicts


class ConnectionOptions(TypedDict, total=False):
    """Connection settings one call changes relative to the client defaults."""

    api_key: str
    timeout: float
    max_retries: int


class FileStore(ABC):
    """One provider's Files API, built on its transport's SDK client.

    A transport returns one from ``create_files`` when the provider has a
    Files API; otherwise ``Transport.files`` is ``None``.
    """

    def __init__(self, transport: "Transport") -> None:
        self.transport = transport

    @property
    def client(self) -> Any:
        return self.transport.client

    @abstractmethod
    def upload(self, source: Media) -> Any:
        """Upload ``source`` and return the SDK's file object."""

    @abstractmethod
    async def upload_async(self, source: Media) -> Any:
        """Async version of ``upload``."""

    @abstractmethod
    def delete(self, file: Media | str) -> None:
        """Delete an uploaded file, given its media or ID."""

    @abstractmethod
    async def delete_async(self, file: Media | str) -> None:
        """Async version of ``delete``."""

    def to_media(self, raw: Any, source: Media) -> Media:
        """Media that references the upload; by default the SDK file's ``id``."""
        return type(source)(
            file_id=raw.id, mime_type=source.mime_type, filename=source.filename
        )

    def wait_until_ready(self, raw: Any) -> Any:
        """Poll until the provider has processed the upload; immediate by default."""
        return raw

    async def wait_until_ready_async(self, raw: Any) -> Any:
        """Async version of ``wait_until_ready``."""
        return raw

    @staticmethod
    def file_id(file: Media | str) -> str:
        """The uploaded file's ID from media or a plain string, raising if it is missing."""
        file_id = file if isinstance(file, str) else file.file_id
        if not file_id:
            raise ValueError(
                "expected an uploaded file: a file ID or media from upload_file"
            )
        return file_id


class Transport(ABC):
    """Owns one provider's SDK client and sends requests through it.

    It is built from the client's default config. Per-call connection changes
    arrive as ``ConnectionOptions``, and each transport applies them its own
    way without changing the shared client.
    """

    def __init__(self, *, config: LLMConfig, info: ProviderInfo) -> None:
        self.config = config
        self.info = info
        self.client = self.create_client()
        self.files = self.create_files()

    @abstractmethod
    def create_client(self) -> Any:
        """Build the SDK client from ``self.config``."""

    def create_files(self) -> FileStore | None:
        """This provider's Files API, or ``None`` when it has none."""
        return None

    def supports(self, *, asynchronous: bool) -> bool:
        """Whether this transport can serve sync or async calls.

        By default only the mode the client was created for; a transport whose
        SDK client offers both overrides this.
        """
        return asynchronous == self.config.asynchronous

    def client_kwargs(self) -> dict[str, Any]:
        """The common SDK client arguments: the config's connection settings."""
        params = self.config.model_dump(
            include=set(CONNECTION_FIELDS), exclude_none=True
        )
        if self.config.api_key is not None:
            params["api_key"] = self.config.api_key.get_secret_value()
        return params

    def connection_overrides(self, call: LLMConfig) -> ConnectionOptions:
        """The connection settings ``call`` changes relative to the client defaults."""
        options = call.model_dump(include=set(CONNECTION_FIELDS), exclude_none=True)
        changed = {
            name: value
            for name, value in options.items()
            if getattr(call, name) != getattr(self.config, name)
        }
        if "api_key" in changed and call.api_key is not None:
            changed["api_key"] = call.api_key.get_secret_value()
        return cast(ConnectionOptions, changed)

    def dict_param(
        self,
        params: dict[str, Any],
        key: str,
        *,
        pop: bool = False,
        label: str | None = None,
    ) -> dict[str, Any]:
        """``params[key]`` as a dict (default ``{}``), with this provider named in errors."""
        return Dicts.get(
            params, key, owner=self.info.display_name, pop=pop, label=label
        )

    @abstractmethod
    def send(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        """Make a synchronous SDK call and return its raw result."""

    @abstractmethod
    async def send_async(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        """Make an asynchronous SDK call and return its raw result."""


class WithOptionsTransport(Transport):
    """Transport for SDK clients that offer ``with_options`` (OpenAI, Anthropic and compatible).

    Per-call connection settings become a derived client, so the shared client
    is never changed. Subclasses implement ``create_client`` and ``call``.
    """

    def client_for(self, connection: ConnectionOptions) -> Any:
        """The shared client, or a derived one when this call changes connection settings."""
        return self.client.with_options(**connection) if connection else self.client

    def send(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        return self.call(self.client_for(connection), request, stream=stream)

    async def send_async(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        return await self.call(self.client_for(connection), request, stream=stream)

    @abstractmethod
    def call(self, client: Any, request: dict[str, Any], *, stream: bool) -> Any:
        """Make the SDK call on ``client``; it returns an awaitable for async clients."""
