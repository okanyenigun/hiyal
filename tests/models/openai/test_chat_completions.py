import base64
from datetime import datetime, timezone
from types import SimpleNamespace as N

import pytest
from openai._models import construct_type
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from hiyal.models.bases import LLMStream
from hiyal.models.openai import ChatCompletionsTranslator, ChatContents
from hiyal.models.schemas import ProviderInfo, ToolCallContent, ToolCallDeltaContent

from tests.support import chat_chunk, chat_completion

INFO = ProviderInfo(provider="deepseek", display_name="Chat")
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20


def message(content=None, reasoning=None, tool_calls=None, audio=None):
    return N(content=content, reasoning_content=reasoning, tool_calls=tool_calls, audio=audio)


def call(index=None, call_id=None, name=None, arguments=None):
    function = None if name is None and arguments is None else N(name=name, arguments=arguments)
    return N(index=index, id=call_id, function=function)


class TestChatContents:
    def test_full_message_splits_into_ordered_parts(self):
        contents = ChatContents()
        fragments = contents.apply(message("Hello.", "think", [call(None, "c1", "f", '{"a":1}')]))
        assert [(i, c.content_type) for i, c in fragments] == [(0, "reasoning"), (1, "text"), (2, "tool_call")]
        assert isinstance(fragments[2][1], ToolCallDeltaContent)
        assert (fragments[2][1].data.name, fragments[2][1].data.arguments_delta) == ("f", '{"a":1}')
        final = contents.final_contents(complete=True)
        assert [c.text for c in final[:2]] == ["think", "Hello."]
        typed = final[2].data
        assert isinstance(final[2], ToolCallContent)
        assert (typed.name, typed.arguments, typed.status, typed.raw.id) == ("f", {"a": 1}, "complete", "c1")
        assert len(set(contents.ids)) == 3 and all(i.startswith("item_") for i in contents.ids)

    def test_deltas_append_to_the_same_parts(self):
        contents = ChatContents()
        contents.apply(message("Hel", "th"))
        fragments = contents.apply(message("lo.", "ink"))
        assert [(i, c.text) for i, c in fragments] == [(0, "ink"), (1, "lo.")]
        assert [c.text for c in contents.final_contents(complete=True)] == ["think", "Hello."]

    def test_parts_are_ordered_by_first_appearance(self):
        contents = ChatContents()
        contents.apply(message("text first"))
        contents.apply(message(None, "reasoning later"))
        assert [c.content_type for c in contents.contents] == ["text", "reasoning"]

    def test_empty_values_add_nothing(self):
        contents = ChatContents()
        assert contents.apply(message("", "", [], None)) == []
        assert contents.contents == []

    def test_streamed_tool_calls_are_matched_by_index(self):
        contents = ChatContents()
        contents.apply(message(tool_calls=[call(0, "c1", "f", '{"ci'), call(1, "c2", "g", "")]))
        fragments = contents.apply(message(tool_calls=[call(0, arguments='ty":"Paris"}'), call(1, arguments="{}")]))
        assert [(i, c.data.name, c.data.arguments_delta) for i, c in fragments] == [
            (0, None, 'ty":"Paris"}'), (1, None, "{}")]
        assert contents.apply(message(tool_calls=[call(0, arguments="")])) == []
        final = contents.final_contents(complete=True)
        assert [(c.data.name, c.data.arguments) for c in final] == [("f", {"city": "Paris"}), ("g", {})]

    def test_unusable_calls_are_invalid_not_dropped(self):
        contents = ChatContents()
        contents.apply(message(tool_calls=[call(None, "c1", "f", "not json"), call(None, "c2", "g", None),
                                           call(None, "c3", "h", "[1]")]))
        calls = [c.data for c in contents.final_contents(complete=True)]
        assert [(c.status, c.issue.code) for c in calls] == [
            ("invalid", "invalid_json"), ("invalid", "missing_arguments"), ("invalid", "arguments_not_object")]
        assert [c.name for c in calls] == ["f", "g", "h"]

    def test_incomplete_generation_keeps_calls_incomplete(self):
        contents = ChatContents()
        contents.apply(message(tool_calls=[call(None, "c1", "f", '{"a": 1}')]))
        assert contents.final_contents(complete=False)[0].data.status == "incomplete"
        assert contents.contents[0].data.status == "incomplete"  # while still being assembled

    def test_tool_call_without_function(self):
        contents = ChatContents()
        contents.apply(message(tool_calls=[N(index=0, id="c1", function=None)]))
        call_ = contents.final_contents(complete=True)[0].data
        assert (call_.name, call_.status, call_.issue.code) == (None, "invalid", "missing_name")

    def test_audio_chunks_join_and_transcripts_concatenate(self):
        contents = ChatContents()
        first, second = WAV[:10], WAV[10:]
        contents.apply(message(audio={"data": base64.b64encode(first).decode(), "transcript": "Hel"}))
        fragments = contents.apply(message(audio=N(data=base64.b64encode(second).decode(), transcript="lo")))
        assert fragments[0][1].data == {"data": second, "transcript": "lo"}
        [audio] = contents.final_contents(complete=True)
        assert audio.data == {"mime_type": "audio/wav", "data": WAV, "transcript": "Hello"}

    def test_headerless_audio_is_pcm(self):
        contents = ChatContents()
        contents.apply(message(audio={"data": base64.b64encode(b"\x00\x01").decode()}))
        assert contents.contents[0].data["mime_type"] == "audio/pcm"

    def test_empty_audio_delta_adds_nothing(self):
        contents = ChatContents()
        assert contents.apply(message(audio={"data": "", "transcript": ""})) == []
        assert contents.apply(message(audio={"id": "a1"})) == []
        assert contents.contents == []


@pytest.fixture
def translator():
    return ChatCompletionsTranslator(INFO)


def parse(translator, payload, **kwargs):
    # The SDK parses responses leniently, keeping provider-specific values
    # (e.g. DeepSeek's finish reasons) that its strict model would reject.
    return translator.parse_response(construct_type(type_=ChatCompletion, value=payload), **kwargs)


class TestChatParsing:
    def test_text_response(self, translator):
        result = parse(translator, chat_completion(), elapsed_seconds=1.25)
        assert (result.text, result.id_, result.model, result.status, result.provider) == (
            "Hello.", "chatcmpl-1", "chat-model", "completed", "deepseek")
        assert result.created_at == datetime.fromtimestamp(1792499155, tz=timezone.utc)
        assert result.messages[0].finish_reason == "stop"
        assert (result.timing.source, result.timing.total_seconds) == ("calculated", 1.25)
        assert result.usage.model_dump(exclude_none=True) == {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39}

    @pytest.mark.parametrize("reason, finish, status", [
        ("stop", "stop", "completed"), ("length", "length", "incomplete"),
        ("content_filter", "content_filter", "completed"), ("tool_calls", "tool_call", "completed"),
        ("insufficient_system_resource", "other", "incomplete"),
    ])
    def test_finish_reasons_and_status(self, translator, reason, finish, status):
        result = parse(translator, chat_completion(finish_reason=reason))
        assert (result.messages[0].finish_reason, result.status) == (finish, status)

    def test_reasoning_and_tool_calls(self, translator):
        payload = chat_completion({"reasoning_content": "hmm", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
            finish_reason="tool_calls")
        result = parse(translator, payload)
        assert [m.content.content_type for m in result.messages] == ["reasoning", "text", "tool_call"]
        assert result.reasoning == "hmm" and result.messages[0].finish_reason is None
        assert [(c.name, c.arguments, c.status) for c in result.tool_calls] == [("f", {"a": 1}, "complete")]
        assert result.has_tool_calls and result.messages[2].finish_reason == "tool_call"

    def test_chat_tool_calls_are_typed_tool_calls(self, translator):
        payload = chat_completion({"tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "oops"}}]})
        calls = parse(translator, payload).tool_calls
        assert [(c.name, c.arguments, c.status) for c in calls] == [("f", {}, "complete"), ("g", None, "invalid")]
        assert calls[0].raw.id == "c1"

    def test_cut_off_responses_keep_calls_incomplete(self, translator):
        payload = chat_completion({"tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
            finish_reason="length")
        assert parse(translator, payload).tool_calls[0].status == "incomplete"

    def test_cache_hits_prefer_deepseek_field_then_prompt_details(self, translator):
        usage = {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39, "prompt_cache_hit_tokens": 4,
                 "prompt_tokens_details": {"cached_tokens": 7}, "completion_tokens_details": {"reasoning_tokens": 3}}
        result = parse(translator, chat_completion(usage=usage))
        assert (result.usage.cache_read_tokens, result.usage.reasoning_tokens) == (4, 3)
        del usage["prompt_cache_hit_tokens"]
        assert parse(translator, chat_completion(usage=usage)).usage.cache_read_tokens == 7

    def test_no_usage_and_no_choices(self, translator):
        payload = chat_completion(usage=None)
        payload["usage"] = None
        payload["choices"] = []
        result = parse(translator, payload)
        assert (result.text, result.messages, result.usage, result.status) == ("", [], None, "completed")


def stream(translator, chunks):
    accumulator = translator.stream_accumulator(type("C", (), {"model": "requested"})())
    raw = iter([ChatCompletionChunk.model_validate(chunk) for chunk in chunks])
    return list(LLMStream(raw=raw, event_adapter=accumulator.adapt, finalizer=accumulator.finalize))


class TestChatStreaming:
    def test_deltas_and_terminal_response_agree(self, translator):
        events = stream(translator, [
            chat_chunk({"role": "assistant", "reasoning_content": "hm", "content": "Hel"}),
            chat_chunk({"content": "lo.", "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                          "function": {"name": "f", "arguments": '{"a"'}}]}),
            chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}, finish_reason="tool_calls"),
            chat_chunk(choices=False, usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
        ])
        *deltas, terminal = events
        assert [(e.message_index, e.delta.content_type) for e in deltas] == [
            (0, "reasoning"), (1, "text"), (1, "text"), (2, "tool_call"), (2, "tool_call")]
        assert [(e.delta.data.name, e.delta.data.arguments_delta) for e in deltas[3:]] == [
            ("f", '{"a"'), (None, ": 1}")]
        assert terminal.event_type == "response.completed"
        response = terminal.response
        assert (response.text, response.usage.total_tokens) == ("Hello.", 3)
        assert [m.id for m in response.messages] == [deltas[0].message_id, deltas[1].message_id, deltas[3].message_id]
        assert [(c.arguments, c.status) for c in response.tool_calls] == [({"a": 1}, "complete")]
        assert all(e.response_id == "chatcmpl-1" for e in events)

    def test_alternative_choices_are_ignored(self, translator):
        events = stream(translator, [chat_chunk({"content": "main"}, finish_reason="stop"),
                                     chat_chunk({"content": "alt"}, index=1)])
        assert events[-1].response.text == "main" and len(events) == 2

    def test_stream_cut_short_is_incomplete(self, translator):
        events = stream(translator, [chat_chunk({"content": "partial"})])
        assert (events[-1].event_type, events[-1].status) == ("response.incomplete", "incomplete")

    def test_empty_stream_has_no_terminal_event(self, translator):
        assert stream(translator, []) == []
