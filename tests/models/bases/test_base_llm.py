from datetime import datetime

import pytest
from pydantic import ValidationError

from hiyal.models.bases import AsyncLLMStream, BaseLLM, LLMStream
from hiyal.models.schemas import LLMConfig
from hiyal.prompts import Audio, Image, UserPrompt

from tests.support import run

from .fakes import FakeLLM, FakeTranslator, FakeTransport, FilesLLM, fake_llm


class TestConstruction:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLM()

    def test_provider_info_comes_from_class_attributes(self):
        llm, _ = fake_llm()
        assert (llm.info.provider, llm.info.display_name) == ("openai", "Fake")
        assert llm.info.prompt_media == frozenset({"images"})
        assert dict(llm.info.generation_adapter) == FakeLLM.generation_adapter

    def test_helpers_are_built_once_and_share_the_config(self):
        llm, _ = fake_llm(temperature=0.2)
        assert isinstance(llm.translator, FakeTranslator) and isinstance(llm.transport, FakeTransport)
        assert llm.transport.config is llm.config and llm.translator.info is llm.info
        assert llm.client is llm.transport.client

    @staticmethod
    def _provider_class(name, **attributes):
        return type(name, (BaseLLM,), {
            "_create_translator": lambda self: FakeTranslator(self.info),
            "_create_transport": lambda self: FakeTransport(config=self.config, info=self.info),
            **attributes,
        })

    def test_missing_required_class_attributes_are_named(self):
        with pytest.raises(TypeError, match="^NoName must define display_name$"):
            self._provider_class("NoName", provider="openai")()
        with pytest.raises(TypeError, match="^Bare must define provider and display_name$"):
            self._provider_class("Bare")()

    def test_subclasses_inherit_the_attributes(self):
        class Child(FakeLLM):
            pass

        assert Child(model="m").info.display_name == "Fake"

    def test_client_config_is_independent_of_the_callers(self):
        source = LLMConfig(model="m", others={"a": [1]})
        llm, _ = fake_llm(config=source)
        llm.config.others["a"].append(2)
        assert source.others == {"a": [1]}


class TestQuery:
    def test_sends_the_translated_request_and_parses_the_response(self):
        llm, log = fake_llm(system_prompt="sys")
        result = llm.query(user_prompt="hi", max_tokens=7, metadata={"a": 1})
        assert log == [{"request": {"model": "m", "max_tokens": 7, "metadata": {"a": 1}, "messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]},
            "stream": False, "options": {}}]
        assert (result["raw"], result["model"]) == ("raw-response", "m")
        assert isinstance(result["created_at"], datetime) and result["created_at"].tzinfo is not None
        assert result["elapsed"] >= 0

    def test_config_object_and_keywords(self):
        llm, log = fake_llm()
        llm.query(config=LLMConfig(user_prompt="from config", temperature=0.9), temperature=0.1)
        assert log[0]["request"]["temperature"] == 0.9

    def test_per_call_connection_changes_use_a_derived_client(self):
        llm, log = fake_llm(api_key="k", timeout=30)
        llm.query(user_prompt="hi", api_key="k2", timeout=5, max_retries=0)
        llm.query(user_prompt="hi")
        assert log[0]["options"] == {"api_key": "k2", "timeout": 5.0, "max_retries": 0}
        assert log[1]["options"] == {}
        assert llm.config.api_key.get_secret_value() == "k" and llm.client.options == {}

    def test_calls_do_not_leak_into_each_other_or_the_client(self):
        llm, log = fake_llm(temperature=0.5)
        llm.query(user_prompt="first", temperature=0.1, extra=1)
        llm.query(user_prompt="second")
        assert log[1]["request"]["temperature"] == 0.5 and "extra" not in log[1]["request"]
        assert llm.config.user_prompt is None

    @pytest.mark.parametrize("parameters, error", [
        ({}, ValidationError),
        ({"user_prompt": UserPrompt(text="x", audio=Audio(mime_type="audio/wav", data=b"x"))}, NotImplementedError),
        ({"user_prompt": "x", "asynchronous": True}, ValueError),
        ({"user_prompt": "x", "stream": True}, ValueError),
        ({"user_prompt": "x", "tool_choice": "required"}, ValueError),
        ({"user_prompt": "x", "temperature": -1}, ValidationError),
    ])
    def test_invalid_calls_fail_before_anything_is_sent(self, parameters, error):
        llm, log = fake_llm()
        with pytest.raises(error):
            llm.query(**parameters)
        assert log == []

    def test_missing_model_fails_before_sending(self):
        llm, log = fake_llm(model=None)
        with pytest.raises(ValueError, match="A model is required"):
            llm.query(user_prompt="x")
        llm.query(user_prompt="x", model="per-call")
        assert log[0]["request"]["model"] == "per-call"

    def test_supported_media_is_validated_and_sent(self):
        llm, log = fake_llm()
        llm.query(user_prompt=UserPrompt(text="x", images=Image(mime_type="image/png", data=b"x")))
        assert log


class TestStreaming:
    def test_query_stream_returns_a_lazy_event_stream(self):
        llm, log = fake_llm()
        stream = llm.query_stream(user_prompt="hi")
        assert isinstance(stream, LLMStream)
        assert log[0]["stream"] is True
        assert [event.delta.text for event in stream] == ["a", "b"]

    def test_stream_events_are_numbered(self):
        llm, _ = fake_llm()
        assert [event.sequence_number for event in llm.query_stream(user_prompt="hi")] == [0, 1]


class TestModeGuard:
    def test_sync_methods_on_an_async_client_fail_fast(self):
        llm, log = fake_llm(asynchronous=True)
        with pytest.raises(TypeError, match="this client is asynchronous; use query_async"):
            llm.query(user_prompt="hi")
        with pytest.raises(TypeError, match="use query_stream_async"):
            llm.query_stream(user_prompt="hi")
        assert log == []

    def test_async_methods_on_a_sync_client_fail_fast(self):
        llm, log = fake_llm()
        with pytest.raises(TypeError, match="this client is synchronous; use query$"):
            run(llm.query_async(user_prompt="hi"))
        with pytest.raises(TypeError, match="use query_stream$"):
            run(llm.query_stream_async(user_prompt="hi"))
        assert log == []

    def test_mode_is_checked_before_the_config(self):
        llm, _ = fake_llm(asynchronous=True)
        with pytest.raises(TypeError):
            llm.query()


class TestAsync:
    def test_query_async(self):
        llm, log = fake_llm(asynchronous=True, api_key="k")
        result = run(llm.query_async(user_prompt="hi", api_key="k2"))
        assert result["raw"] == "raw-response" and log[0]["options"] == {"api_key": "k2"}

    def test_query_stream_async(self):
        llm, _ = fake_llm(asynchronous=True)

        async def scenario():
            stream = await llm.query_stream_async(user_prompt="hi")
            assert isinstance(stream, AsyncLLMStream)
            return [event.delta.text async for event in stream]
        assert run(scenario()) == ["a", "b"]


PNG = b"\x89PNG\r\n\x1a\n0000"


class TestFilesApi:
    def test_providers_without_a_files_api_raise(self):
        llm, _ = fake_llm()
        with pytest.raises(NotImplementedError, match="Fake has no Files API in Hiyal"):
            llm.upload_file(PNG, "image/png")
        with pytest.raises(NotImplementedError):
            llm.delete_file("file-1")
        with pytest.raises(NotImplementedError):
            run(llm.upload_file_async(PNG, "image/png"))

    def test_upload_returns_media_referencing_the_file(self):
        llm, _ = fake_llm(FilesLLM)
        media = llm.upload_file(PNG, "image/png", filename="a.png")
        assert type(media) is Image and (media.file_id, media.filename, media.data) == ("file_1", "a.png", None)
        assert llm.transport.files.calls[0][0] == "upload"

    def test_upload_accepts_paths(self, tmp_path):
        path = tmp_path / "chart.png"
        path.write_bytes(PNG)
        llm, _ = fake_llm(FilesLLM)
        assert llm.upload_file(path).filename == "chart.png"

    def test_upload_input_errors(self):
        llm, _ = fake_llm(FilesLLM)
        with pytest.raises(ValueError, match="pass mime_type"):
            llm.upload_file(PNG)
        with pytest.raises(ValueError, match="must have inline data"):
            llm.upload_file(Image(mime_type="image/png", url="https://e.com/a.png"))

    def test_delete_accepts_media_or_an_id(self):
        llm, _ = fake_llm(FilesLLM)
        llm.delete_file("file-9")
        llm.delete_file(Image(mime_type="image/png", file_id="file-8"))
        assert llm.transport.files.calls == [("delete", "file-9"), ("delete", "file-8")]

    def test_async_files(self):
        llm, _ = fake_llm(FilesLLM, asynchronous=True)
        media = run(llm.upload_file_async(PNG, "image/png"))
        run(llm.delete_file_async(media))
        assert [call[0] for call in llm.transport.files.calls] == ["upload_async", "delete_async"]

    def test_files_follow_the_mode_guard(self):
        sync_llm, _ = fake_llm(FilesLLM)
        async_llm, _ = fake_llm(FilesLLM, asynchronous=True)
        with pytest.raises(TypeError, match="use upload_file$"):
            run(sync_llm.upload_file_async(PNG, "image/png"))
        with pytest.raises(TypeError, match="use delete_file_async"):
            async_llm.delete_file("file-1")


class TestConcurrency:
    def test_parallel_async_calls_stay_isolated(self):
        import asyncio
        llm, log = fake_llm(asynchronous=True, temperature=0.5, api_key="k")

        async def scenario():
            return await asyncio.gather(*(
                llm.query_async(user_prompt=f"call {i}", temperature=i / 10, api_key=f"key-{i}")
                for i in range(5)))
        run(scenario())
        by_prompt = {entry["request"]["messages"][-1]["content"]: entry for entry in log}
        for i in range(5):
            entry = by_prompt[f"call {i}"]
            assert entry["request"]["temperature"] == i / 10
            assert entry["options"] == {"api_key": f"key-{i}"}
        assert llm.config.temperature == 0.5 and llm.config.api_key.get_secret_value() == "k"
