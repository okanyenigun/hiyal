from copy import copy
from functools import partial
from typing import Any
from .request import ENDPOINT_KEY, request_with_retries, request_with_retries_async
from .thinking import validate_ollama_think
from ..bases import ConnectionOptions, Transport


class OllamaTransport(Transport):
    """Sends chat (or generate) requests through the Ollama SDK.

    Ollama's chat API has no per-request options, so per-call timeout, headers
    and retries are bound on a local facade over the shared client; the shared
    HTTP pool, credentials and defaults are never changed or closed. A ``think``
    setting is first checked against the model's metadata from ``/api/show``.
    """

    def create_client(self) -> Any:
        if self.config.asynchronous:
            from ollama import AsyncClient as Client
        else:
            from ollama import Client

        params = dict(self.config.client_options)
        httpx_kwargs = self.dict_param(params, "httpx_kwargs", pop=True)
        params = {**httpx_kwargs, **params}
        if self.config.timeout is not None:
            params["timeout"] = self.config.timeout
        headers = dict(params.pop("headers", {}))
        if self.config.api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self.config.api_key.get_secret_value()}"
            )
        if headers:
            params["headers"] = headers
        return Client(**params)

    def send(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        client = self._client_for(request, connection, asynchronous=False)
        if request.get("think") is not None:
            info = client._request(
                dict, "POST", "/api/show", json={"model": request.get("model", "")}
            )
            request = self._validated_think(request, info)
        return self._call(client, request, stream=stream)

    async def send_async(
        self, request: dict[str, Any], *, stream: bool, connection: ConnectionOptions
    ) -> Any:
        client = self._client_for(request, connection, asynchronous=True)
        if request.get("think") is not None:
            # The SDK's ShowResponse drops this metadata, so keep the native JSON.
            info = await client._request(
                dict, "POST", "/api/show", json={"model": request.get("model", "")}
            )
            request = self._validated_think(request, info)
        return await self._call(client, request, stream=stream)

    def _client_for(
        self,
        request: dict[str, Any],
        connection: ConnectionOptions,
        *,
        asynchronous: bool,
    ) -> Any:
        """The shared client, or a local facade with this call's HTTP options and retries."""
        retries = connection.get("max_retries", self.config.max_retries or 0)
        http_options: dict[str, Any] = {}
        if "timeout" in connection:
            http_options["timeout"] = connection["timeout"]
        if "api_key" in connection:
            http_options["headers"] = {
                "Authorization": f"Bearer {connection['api_key']}"
            }
        if not http_options and not retries and request.get("think") is None:
            return self.client
        local = copy(self.client)
        retry = request_with_retries_async if asynchronous else request_with_retries
        local._request = partial(retry, self.client._request, retries, **http_options)
        return local

    @staticmethod
    def _validated_think(
        request: dict[str, Any], info: dict[str, Any]
    ) -> dict[str, Any]:
        model = request.get("model", "")
        return {
            **request,
            "think": validate_ollama_think(request["think"], info, model),
        }

    @staticmethod
    def _call(client: Any, request: dict[str, Any], *, stream: bool) -> Any:
        """chat(), or generate() for image generation requests."""
        request = dict(request)
        if request.pop(ENDPOINT_KEY, "chat") == "generate":
            if stream:
                raise NotImplementedError(
                    "Ollama image generation cannot be streamed; use query"
                )
            request.pop("stream", None)
            return client.generate(**request, stream=False)
        return client.chat(**{**request, "stream": stream})
