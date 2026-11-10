from typing import Any
from ..bases import FileStore, WithOptionsTransport
from ...prompts.media import Media


class AnthropicTransport(WithOptionsTransport):
    """Sends Messages API requests through the Anthropic SDK."""

    def create_client(self) -> Any:
        if self.config.asynchronous:
            from anthropic import AsyncAnthropic as Anthropic
        else:
            from anthropic import Anthropic
        return Anthropic(**self.config.client_options, **self.client_kwargs())

    def create_files(self) -> "AnthropicFileStore":
        return AnthropicFileStore(self)

    def call(self, client: Any, request: dict[str, Any], *, stream: bool) -> Any:
        return client.messages.create(**{**request, "stream": stream})


class AnthropicFileStore(FileStore):
    """Anthropic Files API; uploads are usable immediately."""

    def upload(self, source: Media) -> Any:
        return self.client.files.upload(file=self._file(source))

    async def upload_async(self, source: Media) -> Any:
        return await self.client.files.upload(file=self._file(source))

    def delete(self, file: Media | str) -> None:
        self.client.files.delete(self.file_id(file))

    async def delete_async(self, file: Media | str) -> None:
        await self.client.files.delete(self.file_id(file))

    @staticmethod
    def _file(source: Media) -> tuple[str, bytes | None, str]:
        return (source.name, source.data, source.mime_type)
