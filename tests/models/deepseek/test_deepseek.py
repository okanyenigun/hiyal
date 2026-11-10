import httpx2
import pytest

from hiyal.models import DeepSeekLLM
from hiyal.models.openai import ChatCompletionsTranslator, ChatCompletionsTransport
from hiyal.models.schemas import UnsupportedThinkingError
from hiyal.prompts import Document, Image, UserPrompt

from tests.support import auth, body, build_request, chat_chunk, chat_completion, make_llm, run, sse

SPEC = {"name": "get_weather", "description": "Get weather.", "parameters": {"properties": []}}


@pytest.fixture
def llm():
    return DeepSeekLLM(model="deepseek-chat", api_key="test-key")


class TestStructure:
    def test_reuses_openai_chat_completions(self, llm):
        assert isinstance(llm.translator, ChatCompletionsTranslator)
        assert isinstance(llm.transport, ChatCompletionsTransport)
        assert llm.transport.files is None


class TestRequest:
    def test_chat_messages(self, llm):
        assert build_request(llm, system_prompt="Be brief.", temperature=0.2, max_output_tokens=50,
                             stop_sequences=["END"]) == {
            "model": "deepseek-chat", "temperature": 0.2, "max_tokens": 50, "stop": ["END"],
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}

    def test_provider_spelling(self, llm):
        params = build_request(llm, max_tokens=9, stop=["X"])
        assert (params["max_tokens"], params["stop"]) == (9, ["X"])

    def test_images_use_chat_content_parts(self, llm):
        prompt = UserPrompt(text="Describe", images=Image(mime_type="image/png", data=b"x"))
        content = build_request(llm, user_prompt=prompt)["messages"][-1]["content"]
        assert content == [{"type": "text", "text": "Describe"},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]

    def test_unsupported_inputs(self, llm):
        with pytest.raises(ValueError, match="DeepSeek does not support top_k"):
            build_request(llm, top_k=5)
        with pytest.raises(NotImplementedError, match="DeepSeek does not support user_prompt documents"):
            build_request(llm, user_prompt=UserPrompt(text="x", documents=Document(mime_type="text/plain", data=b"x")))
        with pytest.raises(ValueError, match="DeepSeek only supports the TEXT response modality"):
            build_request(llm, response_modalities=["IMAGE"])

    def test_normalized_tools_are_not_supported_yet(self, llm):
        with pytest.raises(NotImplementedError, match="DeepSeek does not support normalized tool configuration"):
            build_request(llm, tools=[SPEC])


class TestThinking:
    @pytest.mark.parametrize("effort, wire", [("low", "low"), ("medium", "high"), ("high", "high"), ("max", "max")])
    def test_effort(self, llm, effort, wire):
        params = build_request(llm, thinking_effort=effort)
        assert params["reasoning_effort"] == wire and "extra_body" not in params

    def test_modes_go_to_extra_body(self, llm):
        assert build_request(llm, thinking_mode="enabled")["extra_body"] == {"thinking": {"type": "enabled"}}
        params = build_request(llm, thinking_mode="disabled", others={"reasoning_effort": "high"})
        assert params["extra_body"] == {"thinking": {"type": "disabled"}} and "reasoning_effort" not in params

    def test_native_options_merge(self, llm):
        params = build_request(llm, thinking_mode="enabled", thinking={"others": {"budget_hint": 1}},
                               others={"extra_body": {"other": 1, "thinking": {"keep": True}}})
        assert params["extra_body"] == {"other": 1, "thinking": {"keep": True, "budget_hint": 1, "type": "enabled"}}

    def test_budget_is_not_supported(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="deepseek model 'deepseek-chat': thinking_budget"):
            build_request(llm, thinking_budget=100)


class TestClient:
    def test_api_key_is_required_and_never_taken_from_openai(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        with pytest.raises(ValueError, match="DeepSeek needs an api_key or the DEEPSEEK_API_KEY"):
            DeepSeekLLM(model="deepseek-chat")

    def test_api_key_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekLLM(model="deepseek-chat").client.api_key == "env-key"

    def test_explicit_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekLLM(model="deepseek-chat", api_key="explicit").client.api_key == "explicit"

    def test_base_url_default_and_override(self):
        assert str(DeepSeekLLM(model="m", api_key="k").client.base_url).startswith("https://api.deepseek.com")
        custom = DeepSeekLLM(model="m", api_key="k", base_url="https://proxy.example/v1")
        assert str(custom.client.base_url) == "https://proxy.example/v1/"


def respond(*, stream=None):
    def handler(request):
        if body(request).get("stream"):
            return httpx2.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=chat_completion({"reasoning_content": "hmm"}))
    return handler


class TestEndToEnd:
    def test_query(self):
        llm, requests = make_llm(DeepSeekLLM, respond())
        result = llm.query(user_prompt="Hi", thinking_effort="medium")
        request = requests[0]
        assert (str(request.url), auth(request)) == ("https://api.deepseek.com/chat/completions", "test-key")
        assert body(request)["reasoning_effort"] == "high" and body(request)["stream"] is False
        assert (result.text, result.reasoning, result.provider) == ("Hello.", "hmm", "deepseek")

    def test_stream_requests_usage(self):
        stream = sse([chat_chunk({"content": "x"}, finish_reason="stop"),
                      chat_chunk(choices=False, usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})],
                     named=False, done=True)
        llm, requests = make_llm(DeepSeekLLM, respond(stream=stream))
        events = list(llm.query_stream(user_prompt="Hi"))
        assert body(requests[0])["stream_options"] == {"include_usage": True}
        assert events[-1].response.provider == "deepseek" and events[-1].response.usage.total_tokens == 2

    def test_per_call_overrides(self):
        llm, requests = make_llm(DeepSeekLLM, respond(), timeout=30)
        llm.query(user_prompt="Hi", api_key="call-key", timeout=5)
        llm.query(user_prompt="Hi")
        assert [auth(r) for r in requests] == ["call-key", "test-key"]
        assert [r.extensions["timeout"]["read"] for r in requests] == [5, 30]

    def test_async(self):
        llm, _ = make_llm(DeepSeekLLM, respond(), asynchronous=True)
        assert run(llm.query_async(user_prompt="Hi")).text == "Hello."

    def test_no_files_api(self, llm):
        with pytest.raises(NotImplementedError, match="DeepSeek has no Files API"):
            llm.upload_file(b"x", "image/png")


class TestToolCallResponses:
    TOOL_MESSAGE = {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}

    def test_calls_are_typed(self):
        llm, _ = make_llm(DeepSeekLLM, lambda r: httpx2.Response(
            200, json=chat_completion(self.TOOL_MESSAGE, finish_reason="tool_calls")))
        result = llm.query(user_prompt="Weather?")
        assert result.has_tool_calls
        [call] = result.tool_calls
        assert (call.name, call.arguments, call.status, call.raw.id) == ("get_weather", {"city": "Paris"}, "complete", "c1")
        assert result.messages[-1].finish_reason == "tool_call"

    def test_streamed_calls_are_typed_and_fragments_are_deltas(self):
        stream = sse([
            chat_chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"ci'}}]}),
            chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Paris"}'}}]},
                       finish_reason="tool_calls"),
        ], named=False, done=True)
        llm, _ = make_llm(DeepSeekLLM, lambda r: httpx2.Response(
            200, text=stream, headers={"content-type": "text/event-stream"}))
        *deltas, terminal = list(llm.query_stream(user_prompt="Weather?"))
        assert [(d.delta.data.name, d.delta.data.arguments_delta) for d in deltas] == [
            ("get_weather", '{"ci'), (None, 'ty": "Paris"}')]
        assert [(c.arguments, c.status) for c in terminal.response.tool_calls] == [({"city": "Paris"}, "complete")]
        assert deltas[0].message_id == terminal.response.messages[0].id

    def test_a_stream_cut_short_keeps_calls_incomplete(self):
        stream = sse([chat_chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                  "function": {"name": "f", "arguments": '{"a": 1}'}}]})],
                     named=False, done=True)
        llm, _ = make_llm(DeepSeekLLM, lambda r: httpx2.Response(
            200, text=stream, headers={"content-type": "text/event-stream"}))
        terminal = list(llm.query_stream(user_prompt="x"))[-1]
        assert (terminal.status, terminal.response.tool_calls[0].status) == ("incomplete", "incomplete")
