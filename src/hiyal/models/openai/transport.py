from typing import Any
from ..bases import FileStore, WithOptionsTransport
from ...prompts.media import Media


class ChatCompletionsTransport(WithOptionsTransport):
    """Sends requests to OpenAI's Chat Completions endpoint.

    Subclasses supply ``create_client``; DeepSeek inherits this transport
    because its API is the same Chat Completions format. ``call`` finishes the
    request for a streamed or blocking call and sends it.
    """

    def call(self, client: Any, request: dict[str, Any], *, stream: bool) -> Any:
        request = {**request, "stream": stream}
        if stream:
            # Usage arrives only on the final chunk, and only when requested.
            options = dict(self.dict_param(request, "stream_options"))
            options.setdefault("include_usage", True)
            request["stream_options"] = options
        else:
            request.pop("stream_options", None)
        return client.chat.completions.create(**request)


class OpenAITransport(ChatCompletionsTransport):
    """Sends OpenAI requests: Responses by default, Chat Completions for audio.

    The translator decides the API; a request with ``messages`` is a Chat
    Completions request.
    """

    def create_client(self) -> Any:
        if self.config.asynchronous:
            from openai import AsyncOpenAI as OpenAI
        else:
            from openai import OpenAI
        return OpenAI(**self.config.client_options, **self.client_kwargs())

    def create_files(self) -> "OpenAIFileStore":
        return OpenAIFileStore(self)

    def call(self, client: Any, request: dict[str, Any], *, stream: bool) -> Any:
        if "messages" in request:
            return super().call(client, request, stream=stream)
        return client.responses.create(**{**request, "stream": stream})


class OpenAIFileStore(FileStore):
    """OpenAI Files API; uploads are usable immediately (``purpose="user_data"``)."""

    def upload(self, source: Media) -> Any:
        return self.client.files.create(**self._upload_args(source))

    async def upload_async(self, source: Media) -> Any:
        return await self.client.files.create(**self._upload_args(source))

    def delete(self, file: Media | str) -> None:
        self.client.files.delete(self.file_id(file))

    async def delete_async(self, file: Media | str) -> None:
        await self.client.files.delete(self.file_id(file))

    @staticmethod
    def _upload_args(source: Media) -> dict[str, Any]:
        return {
            "file": (source.name, source.data, source.mime_type),
            "purpose": "user_data",
        }
