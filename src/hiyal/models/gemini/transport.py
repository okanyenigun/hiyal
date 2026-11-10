import asyncio
import io
import time
from typing import Any
from ..bases import ConnectionOptions, FileStore, Transport
from ...prompts.media import Media


class GeminiTransport(Transport):
    """Sends ``generate_content`` requests through the Google GenAI SDK.

    One ``genai.Client`` serves sync calls (``client.models``) and async calls
    (``client.aio.models``), so both modes work whatever ``asynchronous`` says.
    Per-call connection settings travel in the request's ``http_options``.
    """

    def create_client(self) -> Any:
        from google import genai

        params = dict(self.config.client_options)
        http_options = self.dict_param(params, "http_options", pop=True)
        if self.config.timeout is not None:
            http_options["timeout"] = int(self.config.timeout * 1_000)
        retry_options = self.dict_param(
            http_options, "retry_options", label="http_options.retry_options"
        )
        retry_options["attempts"] = (self.config.max_retries or 0) + 1
        http_options["retry_options"] = retry_options
        if self.config.api_key is not None:
            params["api_key"] = self.config.api_key.get_secret_value()
        params["http_options"] = http_options
        return genai.Client(**params)

    def create_files(self) -> "GeminiFileStore":
        return GeminiFileStore(self)

    def supports(self, *, asynchronous: bool) -> bool:
        return True

    def send(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        request = self._with_connection(request, connection)
        if stream:
            return self.client.models.generate_content_stream(**request)
        return self.client.models.generate_content(**request)

    async def send_async(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        request = self._with_connection(request, connection)
        if stream:
            return await self.client.aio.models.generate_content_stream(**request)
        return await self.client.aio.models.generate_content(**request)

    def _with_connection(
        self, request: dict[str, Any], connection: ConnectionOptions
    ) -> dict[str, Any]:
        """The request with this call's connection settings in ``config.http_options``."""
        if not connection:
            return request
        request = dict(request)
        config = dict(self.dict_param(request, "config"))
        http_options = dict(self.dict_param(config, "http_options"))
        if "api_key" in connection:
            headers = {
                key: value
                for key, value in self.dict_param(http_options, "headers").items()
                if key.lower() != "x-goog-api-key"
            }
            headers["x-goog-api-key"] = connection["api_key"]
            http_options["headers"] = headers
        if "timeout" in connection:
            http_options["timeout"] = int(connection["timeout"] * 1_000)
        if "max_retries" in connection:
            # Preserve provider-specific backoff settings when changing attempts.
            default_http = self.dict_param(self.config.client_options, "http_options")
            http_options["retry_options"] = {
                **self.dict_param(default_http, "retry_options"),
                **self.dict_param(http_options, "retry_options"),
                "attempts": connection["max_retries"] + 1,
            }
        config["http_options"] = http_options
        request["config"] = config
        return request


class GeminiFileStore(FileStore):
    """Gemini Files API: uploads are referenced by URI and deleted by name ("files/abc").

    Large uploads are processed asynchronously, so ``wait_until_ready`` polls
    every ``poll_seconds`` until the file leaves the ``PROCESSING`` state.
    """

    poll_seconds: float = 1.0

    def upload(self, source: Media) -> Any:
        return self.client.files.upload(**self._upload_args(source))

    async def upload_async(self, source: Media) -> Any:
        return await self.client.aio.files.upload(**self._upload_args(source))

    def delete(self, file: Media | str) -> None:
        self.client.files.delete(name=self.file_name(file))

    async def delete_async(self, file: Media | str) -> None:
        await self.client.aio.files.delete(name=self.file_name(file))

    def to_media(self, raw: Any, source: Media) -> Media:
        return type(source)(
            url=raw.uri,
            mime_type=raw.mime_type or source.mime_type,
            filename=source.filename,
        )

    def wait_until_ready(self, raw: Any) -> Any:
        while self._state(raw) == "PROCESSING":
            time.sleep(self.poll_seconds)
            raw = self.client.files.get(name=raw.name)
        return raw

    async def wait_until_ready_async(self, raw: Any) -> Any:
        while self._state(raw) == "PROCESSING":
            await asyncio.sleep(self.poll_seconds)
            raw = await self.client.aio.files.get(name=raw.name)
        return raw

    @staticmethod
    def file_name(file: Media | str) -> str:
        """``files/abc`` from that name, a file URI, or media from upload_file."""
        value = file if isinstance(file, str) else (file.url or file.file_id or "")
        if "/files/" in value:
            value = "files/" + value.rsplit("/files/", 1)[1]
        if not value.startswith("files/"):
            raise ValueError(
                "expected an uploaded Gemini file: its name, URI, or media from upload_file"
            )
        return value

    @staticmethod
    def _upload_args(source: Media) -> dict[str, Any]:
        config = {"mime_type": source.mime_type}
        if source.filename:
            config["display_name"] = source.filename
        return {"file": io.BytesIO(source.data or b""), "config": config}

    @staticmethod
    def _state(raw: Any) -> str:
        state = getattr(raw, "state", None)
        name = getattr(state, "name", state)
        if name == "FAILED":
            raise RuntimeError(
                f"Gemini could not process uploaded file {raw.name}: {raw.error}"
            )
        return str(name)
