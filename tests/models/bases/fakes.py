"""A minimal fake provider, so the base layer can be tested without any SDK."""

from __future__ import annotations

from typing import Any

from hiyal.models.bases import BaseLLM, FileStore, StreamAccumulator, Translator, WithOptionsTransport
from hiyal.models.schemas import LLMResponse, ResponseContent
from hiyal.prompts import Media


class FakeClient:
    """Records calls; ``with_options`` returns a derived client, like the real SDKs."""

    def __init__(self, log: list, options: dict | None = None, asynchronous: bool = False,
                 result: Any = "raw-response", stream_items: tuple = ("a", "b")) -> None:
        self.log, self.options, self.asynchronous = log, options or {}, asynchronous
        self.result, self.stream_items = result, stream_items

    def with_options(self, **options: Any) -> FakeClient:
        return FakeClient(self.log, {**self.options, **options}, self.asynchronous, self.result, self.stream_items)

    def create(self, request: dict, stream: bool) -> Any:
        self.log.append({"request": request, "stream": stream, "options": self.options})
        if self.asynchronous:
            return self._create_async(stream)
        return iter(self.stream_items) if stream else self.result

    async def _create_async(self, stream: bool) -> Any:
        if not stream:
            return self.result

        async def items():
            for item in self.stream_items:
                yield item
        return items()


class FakeTransport(WithOptionsTransport):
    def create_client(self) -> FakeClient:
        return FakeClient(self.config.client_options.get("log", []), asynchronous=self.config.asynchronous)

    def call(self, client: FakeClient, request: dict, *, stream: bool) -> Any:
        return client.create(request, stream)


class FakeAccumulator(StreamAccumulator["FakeTranslator"]):
    def _adapt(self, event: Any) -> list:
        return [self._delta(message_index=0, message_id="m0", content=ResponseContent(content_type="text", text=event))]


class FakeTranslator(Translator):
    """Sends the prompt as ``messages`` and returns the parse inputs instead of an LLMResponse."""

    def build_request(self, config: Any) -> dict:
        params = super().build_request(config)
        params["messages"] = self.chat_messages(config)
        # The prompt travels in messages; drop its common and renamed spellings.
        for name in ("user_prompt", "system_prompt", "system"):
            params.pop(name, None)
        return params

    def parse_response(self, response: Any, *, model=None, created_at=None, elapsed_seconds=None) -> Any:
        return {"raw": response, "model": model, "created_at": created_at, "elapsed": elapsed_seconds}

    def stream_accumulator(self, config: Any) -> FakeAccumulator:
        return FakeAccumulator(translator=self, model=config.model)


class FakeLLM(BaseLLM):
    provider = "openai"
    display_name = "Fake"
    prompt_media = frozenset({"images"})
    generation_adapter = {"system": "system_prompt", "max_tokens": "max_output_tokens"}

    def _create_translator(self) -> FakeTranslator:
        return FakeTranslator(self.info)

    def _create_transport(self) -> FakeTransport:
        return FakeTransport(config=self.config, info=self.info)


class FakeFiles(FileStore):
    def __init__(self, transport: Any) -> None:
        super().__init__(transport)
        self.calls: list = []

    def upload(self, source: Media) -> Any:
        self.calls.append(("upload", source))
        return type("Raw", (), {"id": "file_1"})()

    async def upload_async(self, source: Media) -> Any:
        self.calls.append(("upload_async", source))
        return type("Raw", (), {"id": "file_1"})()

    def delete(self, file: Any) -> None:
        self.calls.append(("delete", self.file_id(file)))

    async def delete_async(self, file: Any) -> None:
        self.calls.append(("delete_async", self.file_id(file)))


class FilesTransport(FakeTransport):
    def create_files(self) -> FakeFiles:
        return FakeFiles(self)


class FilesLLM(FakeLLM):
    def _create_transport(self) -> FilesTransport:
        return FilesTransport(config=self.config, info=self.info)


def fake_llm(cls: type = FakeLLM, **settings: Any) -> tuple[Any, list]:
    """A fake LLM and the log its client records calls into.

    The log list is deep-copied with the client config (plain lists are config
    data), so the log returned is the one the transport's client actually uses.
    """
    options = {"log": [], **settings.pop("client_options", {})}
    llm = cls(model=settings.pop("model", "m"), client_options=options, **settings)
    return llm, llm.client.log


def parsed_response(**overrides: Any) -> LLMResponse:
    from datetime import datetime, timezone

    values = {"text": "", "messages": [], "provider": "openai", "id_": "r", "model": "m",
              "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "status": "completed", "raw": None}
    values.update(overrides)
    return LLMResponse(**values)
