import re
from types import SimpleNamespace as N

import httpx
import pytest

from hiyal.models import GeminiLLM
from hiyal.models.gemini import GeminiFileStore
from hiyal.prompts import Audio, Document, Image

from tests.support import auth, body, gemini_response, make_llm, run, sse


def respond(payload=None, *, stream_chunks=None, status=200):
    def handler(request):
        if "streamGenerateContent" in request.url.path:
            return httpx.Response(200, text=sse(stream_chunks or [gemini_response()], named=False),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(status, json=payload or gemini_response())
    return handler


class TestQuery:
    def test_request_path_headers_and_body(self):
        llm, requests = make_llm(GeminiLLM, respond(), system_prompt="Be brief.")
        result = llm.query(user_prompt="Hi", temperature=0.2)
        request = requests[0]
        # The API version is the SDK's default, which differs between releases
        # (google-genai 2.x uses v1beta, 3.x uses v1); Hiyal does not set it.
        assert re.fullmatch(r"/v1(beta)?/models/gemini-2\.5-flash:generateContent", request.url.path)
        assert auth(request) == "test-key"
        sent = body(request)
        assert sent["contents"] == [{"parts": [{"text": "Hi"}], "role": "user"}]
        assert sent["systemInstruction"]["parts"] == [{"text": "Be brief."}]
        assert sent["generationConfig"]["temperature"] == 0.2
        assert result.text == "Hello." and result.provider == "gemini" and result.timing.source == "calculated"

    def test_both_modes_work_on_one_client(self):
        sync_llm, _ = make_llm(GeminiLLM, respond())
        assert run(sync_llm.query_async(user_prompt="Hi")).text == "Hello."
        async_llm, _ = make_llm(GeminiLLM, respond(), asynchronous=True)
        assert run(async_llm.query_async(user_prompt="Hi")).text == "Hello."

    def test_streaming(self):
        chunks = [gemini_response([{"text": "Hel"}], finish_reason=None),
                  gemini_response([{"text": "lo."}], finish_reason="STOP")]
        llm, requests = make_llm(GeminiLLM, respond(stream_chunks=chunks))
        events = list(llm.query_stream(user_prompt="Hi"))
        assert requests[0].url.path.endswith(":streamGenerateContent")
        assert [e.delta.text for e in events[:-1]] == ["Hel", "lo."]
        assert events[-1].response.text == "Hello."

    def test_async_streaming(self):
        llm, _ = make_llm(GeminiLLM, respond(stream_chunks=[gemini_response()]), asynchronous=True)

        async def scenario():
            return [e async for e in await llm.query_stream_async(user_prompt="Hi")]
        assert run(scenario())[-1].response.text == "Hello."

    def test_tool_call_round_trip(self):
        payload = gemini_response([{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}])
        llm, requests = make_llm(GeminiLLM, respond(payload))
        tool = {"name": "get_weather", "description": "Weather.", "parameters": {"properties": [
            {"name": "city", "type": "string", "description": "City.", "required": True}]}}
        result = llm.query(user_prompt="Weather?", tools=[tool], tool_choice="required")
        sent = body(requests[0])
        assert sent["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
        assert sent["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}


class TestConnection:
    def test_per_call_overrides_are_request_local(self):
        llm, requests = make_llm(GeminiLLM, respond(), timeout=30,
                                 client_options={"http_options": {"headers": {"x-fixed": "keep"}}})
        llm.query(user_prompt="Hi", api_key="call-key", timeout=5)
        llm.query(user_prompt="Hi")
        assert [auth(r) for r in requests] == ["call-key", "test-key"]
        assert [r.extensions["timeout"]["read"] for r in requests] == [5, 30]
        assert all(r.headers["x-fixed"] == "keep" for r in requests)
        assert "call-key" not in requests[0].content.decode()

    def test_retries_follow_the_per_call_limit(self):
        attempts = []

        def flaky(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(503, json={"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})
            return httpx.Response(200, json=gemini_response())
        llm, _ = make_llm(GeminiLLM, flaky)
        assert llm.query(user_prompt="Hi", max_retries=1).text == "Hello."
        assert len(attempts) == 2

    def test_client_retry_options_are_validated(self):
        with pytest.raises(TypeError, match="Gemini 'http_options.retry_options' must be a dictionary"):
            make_llm(GeminiLLM, respond(), client_options={"http_options": {"retry_options": 3}})
        with pytest.raises(TypeError, match="Gemini 'http_options' must be a dictionary"):
            GeminiLLM(model="m", api_key="k", client_options={"http_options": 3})

    def test_default_retry_attempts(self):
        llm = GeminiLLM(model="m", api_key="k", max_retries=4)
        assert llm.transport.client._api_client._http_options.retry_options.attempts == 5


class FakeFiles:
    def __init__(self, states=("ACTIVE",)):
        self.states, self.calls = list(states), []

    def _raw(self):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return N(name="files/abc", uri="https://g/v1beta/files/abc", mime_type="image/png",
                 state=N(name=state), error="bad file")

    def upload(self, **kwargs):
        self.calls.append(("upload", kwargs))
        return self._raw()

    def get(self, name):
        self.calls.append(("get", name))
        return self._raw()

    def delete(self, name):
        self.calls.append(("delete", name))


class AsyncFakeFiles(FakeFiles):
    async def upload(self, **kwargs):
        return super().upload(**kwargs)

    async def get(self, name):
        return super().get(name)

    async def delete(self, name):
        return super().delete(name)


def store(states=("ACTIVE",)):
    files, aio_files = FakeFiles(states), AsyncFakeFiles(states)
    file_store = GeminiFileStore(N(client=N(files=files, aio=N(files=aio_files))))
    file_store.poll_seconds = 0
    return file_store, files, aio_files


class TestFileStore:
    def test_upload_sends_bytes_type_and_display_name(self):
        file_store, files, _ = store()
        source = Image(mime_type="image/png", data=b"png", filename="chart.png")
        raw = file_store.upload(source)
        kind, kwargs = files.calls[0]
        assert kind == "upload" and kwargs["file"].read() == b"png"
        assert kwargs["config"] == {"mime_type": "image/png", "display_name": "chart.png"}
        media = file_store.to_media(raw, source)
        assert type(media) is Image and (media.url, media.mime_type, media.filename) == (
            "https://g/v1beta/files/abc", "image/png", "chart.png")

    def test_upload_without_a_filename_omits_the_display_name(self):
        file_store, files, _ = store()
        file_store.upload(Document(mime_type="application/pdf", data=b"%PDF"))
        assert files.calls[0][1]["config"] == {"mime_type": "application/pdf"}

    def test_waits_while_processing(self):
        file_store, files, _ = store(["PROCESSING", "PROCESSING", "ACTIVE"])
        raw = file_store.wait_until_ready(file_store.upload(Image(mime_type="image/png", data=b"x")))
        assert raw.state.name == "ACTIVE" and [c[0] for c in files.calls] == ["upload", "get", "get"]

    def test_failed_processing_raises(self):
        file_store, _, _ = store(["PROCESSING", "FAILED"])
        raw = file_store.upload(Image(mime_type="image/png", data=b"x"))
        with pytest.raises(RuntimeError, match="could not process uploaded file files/abc: bad file"):
            file_store.wait_until_ready(raw)

    def test_async_upload_wait_and_delete(self):
        file_store, _, aio_files = store(["PROCESSING", "ACTIVE"])

        async def scenario():
            raw = await file_store.upload_async(Audio(mime_type="audio/wav", data=b"x"))
            raw = await file_store.wait_until_ready_async(raw)
            await file_store.delete_async("files/abc")
            return raw
        assert run(scenario()).state.name == "ACTIVE"
        assert [c[0] for c in aio_files.calls] == ["upload", "get", "delete"]

    @pytest.mark.parametrize("file, name", [
        ("files/abc", "files/abc"),
        ("https://generativelanguage.googleapis.com/v1beta/files/abc", "files/abc"),
        (Image(mime_type="image/png", url="https://generativelanguage.googleapis.com/v1beta/files/xyz"), "files/xyz"),
        (Image(mime_type="image/png", file_id="files/f1"), "files/f1"),
    ])
    def test_delete_accepts_names_uris_and_media(self, file, name):
        file_store, files, _ = store()
        file_store.delete(file)
        assert files.calls == [("delete", name)]

    @pytest.mark.parametrize("file", ["abc", "", Image(mime_type="image/png", url="https://e.com/a.png")])
    def test_delete_rejects_non_gemini_files(self, file):
        with pytest.raises(ValueError, match="expected an uploaded Gemini file"):
            GeminiFileStore.file_name(file)

    def test_llm_files_api_is_wired(self):
        llm = GeminiLLM(model="m", api_key="k")
        assert isinstance(llm.transport.files, GeminiFileStore)
        assert llm.transport.supports(asynchronous=True) and llm.transport.supports(asynchronous=False)
