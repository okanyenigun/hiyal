import pytest

from hiyal.models.bases import AsyncLLMStream, FileStore, LLMStream
from hiyal.models.schemas import (
    LLMConfig,
    ProviderInfo,
    ResponseContent,
    ToolCallContent,
    ToolCallDeltaContent,
)
from hiyal.prompts import Image
from hiyal.tools import ToolCall

from tests.support import run

from .fakes import FakeAccumulator, FakeTranslator, FakeTransport, FilesTransport, parsed_response

INFO = ProviderInfo(provider="openai", display_name="Fake")


def transport(**values):
    log = []
    values.setdefault("model", "m")
    return FakeTransport(config=LLMConfig(client_options={"log": log}, **values), info=INFO), log


class TestTransport:
    def test_client_kwargs_unwrap_the_key_and_skip_unset_values(self):
        fake, _ = transport(api_key="k", timeout=5)
        assert fake.client_kwargs() == {"api_key": "k", "timeout": 5.0, "max_retries": 2}
        assert transport()[0].client_kwargs() == {"max_retries": 2}

    def test_connection_overrides_contain_only_changes(self):
        fake, _ = transport(api_key="k", timeout=5)
        call = LLMConfig(model="m", api_key="k2", timeout=5, max_retries=0)
        assert fake.connection_overrides(call) == {"api_key": "k2", "max_retries": 0}
        assert fake.connection_overrides(fake.config) == {}

    def test_connection_overrides_ignore_missing_call_values(self):
        fake, _ = transport(timeout=5)
        assert fake.connection_overrides(LLMConfig(model="m", timeout=None)) == {}

    def test_supports_the_mode_the_client_was_built_for(self):
        assert transport()[0].supports(asynchronous=False)
        assert not transport()[0].supports(asynchronous=True)
        assert transport(asynchronous=True)[0].supports(asynchronous=True)

    def test_no_files_api_by_default(self):
        assert transport()[0].files is None

    def test_dict_param_names_the_provider(self):
        with pytest.raises(TypeError, match="^Fake 'x' must be a dictionary$"):
            transport()[0].dict_param({"x": 1}, "x")


class TestWithOptionsTransport:
    def test_no_changes_use_the_shared_client(self):
        fake, log = transport()
        assert fake.client_for({}) is fake.client
        fake.send({"a": 1}, stream=False, connection={})
        assert log == [{"request": {"a": 1}, "stream": False, "options": {}}]

    def test_changes_use_a_derived_client(self):
        fake, log = transport()
        fake.send({}, stream=False, connection={"max_retries": 0})
        assert log[0]["options"] == {"max_retries": 0} and fake.client.options == {}

    def test_async_send_awaits_the_call(self):
        fake, log = transport(asynchronous=True)
        assert run(fake.send_async({}, stream=False, connection={"timeout": 1.0})) == "raw-response"
        assert log[0]["options"] == {"timeout": 1.0}


class TestFileStore:
    files = FilesTransport(config=LLMConfig(), info=INFO).files

    def test_file_id_from_media_or_string(self):
        assert FileStore.file_id("file-1") == "file-1"
        assert FileStore.file_id(Image(mime_type="image/png", file_id="file-2")) == "file-2"

    @pytest.mark.parametrize("file", ["", Image(mime_type="image/png", data=b"x")])
    def test_file_id_needs_an_uploaded_file(self, file):
        with pytest.raises(ValueError, match="expected an uploaded file"):
            FileStore.file_id(file)

    def test_default_media_and_readiness(self):
        source = Image(mime_type="image/png", data=b"x", filename="a.png")
        raw = type("Raw", (), {"id": "file-1"})()
        media = self.files.to_media(raw, source)
        assert type(media) is Image and (media.file_id, media.mime_type, media.filename) == ("file-1", "image/png", "a.png")
        assert self.files.wait_until_ready(raw) is raw
        assert run(self.files.wait_until_ready_async(raw)) is raw

    def test_client_is_the_transports(self):
        assert self.files.client is self.files.transport.client


class TestStreamAccumulator:
    def accumulator(self):
        return FakeAccumulator(translator=FakeTranslator(INFO), model="m")

    def test_adapt_counts_and_remembers_raw_events(self):
        accumulator = self.accumulator()
        assert accumulator.sequence_number == -1
        [event] = accumulator.adapt("a")
        assert (event.sequence_number, event.raw, event.provider, event.delta.text) == (0, "a", "openai", "a")
        accumulator.adapt("b")
        assert (accumulator.sequence_number, accumulator.raw) == (1, "b")
        assert list(accumulator.finalize()) == []

    def test_delta_turns_tool_calls_into_fragments(self):
        accumulator = self.accumulator()
        accumulator.adapt("first")  # _delta runs inside adapt, after the counter moves
        call = ToolCall.normalize(name="f", arguments={"a": 1}, raw=None, complete=False)
        event = accumulator._delta(message_index=2, message_id="m2", content=ToolCallContent(data=call))
        assert isinstance(event.delta, ToolCallDeltaContent)
        assert (event.delta.data.name, event.delta.data.arguments) == ("f", {"a": 1})
        assert (event.message_index, event.message_id, event.content_index) == (2, "m2", 0)

    @pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
    def test_terminal_event_type_follows_the_status(self, status):
        accumulator = self.accumulator()
        accumulator.response_id = "r1"
        [event] = accumulator._terminal(parsed_response(status=status))
        assert (event.event_type, event.status, event.response_id) == (f"response.{status}", status, "r1")
        assert event.sequence_number is None
        accumulator.adapt("x")
        assert accumulator._terminal(parsed_response())[0].sequence_number == 0


class RawStream:
    """A raw SDK stream double that records how it is used."""

    def __init__(self, items, *, error_at=None, close=True):
        self.items, self.error_at, self.position, self.closed = list(items), error_at, 0, 0
        if not close:
            self.close = None  # type: ignore[assignment]

    def __iter__(self):
        return self

    def __next__(self):
        if self.error_at is not None and self.position == self.error_at:
            raise RuntimeError("network down")
        if self.position >= len(self.items):
            raise StopIteration
        self.position += 1
        return self.items[self.position - 1]

    def close(self):
        self.closed += 1


def text_events(raw):
    return [f"event:{raw}"] if raw != "skip" else []


class TestLLMStream:
    def test_is_lazy(self):
        raw = RawStream(["a"])
        LLMStream(raw=raw, event_adapter=text_events)
        assert raw.position == 0

    def test_adapts_each_raw_event_and_skips_empty_results(self):
        stream = LLMStream(raw=RawStream(["a", "skip", "b"]), event_adapter=lambda r: [] if r == "skip" else [r, r])
        assert list(stream) == ["a", "a", "b", "b"]

    def test_finalizer_runs_once_after_exhaustion(self):
        calls = []
        stream = LLMStream(raw=RawStream(["a"]), event_adapter=text_events,
                           finalizer=lambda: calls.append(1) or ["done"])
        assert list(stream) == ["event:a", "done"]
        assert list(stream) == [] and calls == [1]

    def test_error_propagates_and_ends_the_stream(self):
        stream = LLMStream(raw=RawStream(["a", "b"], error_at=1), event_adapter=text_events,
                           finalizer=lambda: ["done"])
        assert next(stream) == "event:a"
        with pytest.raises(RuntimeError, match="network down"):
            next(stream)
        with pytest.raises(StopIteration):
            next(stream)

    def test_close_stops_events_and_closes_the_raw_stream(self):
        raw = RawStream(["a", "b"])
        stream = LLMStream(raw=raw, event_adapter=text_events, finalizer=lambda: ["done"])
        next(stream)
        stream.close()
        assert raw.closed == 1 and list(stream) == []

    def test_context_manager_closes(self):
        raw = RawStream(["a", "b"])
        with LLMStream(raw=raw, event_adapter=text_events) as stream:
            next(stream)
        assert raw.closed == 1

    def test_context_manager_closes_after_an_error(self):
        raw = RawStream(["a"])
        with pytest.raises(KeyError):
            with LLMStream(raw=raw, event_adapter=text_events):
                raise KeyError("boom")
        assert raw.closed == 1

    def test_raw_streams_without_close_are_fine(self):
        LLMStream(raw=iter(["a"]), event_adapter=text_events).close()


class AsyncRawStream:
    def __init__(self, items, *, error_at=None, close_name="aclose", async_close=True):
        self.items, self.error_at, self.position, self.closed = list(items), error_at, 0, []
        if close_name:
            async def aclose():
                self.closed.append(close_name)

            def close():
                self.closed.append(close_name)
            setattr(self, close_name, aclose if async_close else close)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.error_at is not None and self.position == self.error_at:
            raise RuntimeError("network down")
        if self.position >= len(self.items):
            raise StopAsyncIteration
        self.position += 1
        return self.items[self.position - 1]


async def collect(stream):
    return [event async for event in stream]


class TestAsyncLLMStream:
    def test_adapts_and_finalizes(self):
        stream = AsyncLLMStream(raw=AsyncRawStream(["a", "skip", "b"]), event_adapter=text_events,
                                finalizer=lambda: ["done"])
        assert run(collect(stream)) == ["event:a", "event:b", "done"]

    def test_is_lazy(self):
        raw = AsyncRawStream(["a"])
        AsyncLLMStream(raw=raw, event_adapter=text_events)
        assert raw.position == 0

    def test_error_propagates_and_ends_the_stream(self):
        async def scenario():
            stream = AsyncLLMStream(raw=AsyncRawStream(["a"], error_at=1), event_adapter=text_events,
                                    finalizer=lambda: ["done"])
            first = await stream.__anext__()
            with pytest.raises(RuntimeError):
                await stream.__anext__()
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
            return first
        assert run(scenario()) == "event:a"

    @pytest.mark.parametrize("close_name, async_close", [("aclose", True), ("close", False), ("close", True)])
    def test_close_uses_aclose_or_close(self, close_name, async_close):
        raw = AsyncRawStream(["a"], close_name=close_name, async_close=async_close)

        async def scenario():
            async with AsyncLLMStream(raw=raw, event_adapter=text_events) as stream:
                await stream.__anext__()
            return await collect(stream)
        assert run(scenario()) == [] and raw.closed == [close_name]

    def test_raw_streams_without_close_are_fine(self):
        run(AsyncLLMStream(raw=AsyncRawStream([], close_name=None), event_adapter=text_events).close())


def test_accumulator_and_stream_work_together():
    accumulator = FakeAccumulator(translator=FakeTranslator(INFO), model="m")
    stream = LLMStream(raw=iter(["a", "b"]), event_adapter=accumulator.adapt, finalizer=accumulator.finalize)
    assert [event.delta.text for event in stream] == ["a", "b"]
    assert ResponseContent(content_type="text", text="a")
