from datetime import datetime, timezone

import pytest
from google.genai import types

from hiyal.models import GeminiLLM
from hiyal.models.bases import LLMStream
from hiyal.models.schemas import UnsupportedThinkingError
from hiyal.prompts import Audio, Document, Image, UserPrompt
from hiyal.tools import AnthropicToolSpec, GeminiToolSpec

from tests.support import build_request

SPEC = {"name": "get_weather", "description": "Get weather.",
        "parameters": {"properties": [{"name": "city", "type": "string", "description": "City.", "required": True}]}}
AFC_OFF = {"automatic_function_calling": {"disable": True}}


@pytest.fixture
def llm():
    return GeminiLLM(model="gemini-2.5-flash", api_key="test-key")


def candidate(*parts, index=0, reason=None, role="model"):
    return types.Candidate(index=index, finish_reason=reason, content=types.Content(role=role, parts=list(parts)))


def chunk(*candidates, usage=None, **metadata):
    return types.GenerateContentResponse(candidates=list(candidates), usage_metadata=usage, **metadata)


def text(value, thought=None):
    return types.Part(text=value, thought=thought)


def function_call(name="get_weather", args=None, call_id=None, will_continue=None):
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args, will_continue=will_continue))


def usage(prompt=3, output=4, total=7, thoughts=None, cached=None):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=output, total_token_count=total,
        thoughts_token_count=thoughts, cached_content_token_count=cached)


class TestRequest:
    def test_text_prompt_nests_settings_in_config(self, llm):
        assert build_request(llm, system_prompt="Be brief.", temperature=0.2, max_output_tokens=50) == {
            "model": "gemini-2.5-flash", "contents": "Hi",
            "config": {"temperature": 0.2, "max_output_tokens": 50, "system_instruction": "Be brief.", **AFC_OFF}}

    def test_no_system_prompt(self, llm):
        assert build_request(llm) == {"model": "gemini-2.5-flash", "contents": "Hi", "config": AFC_OFF}

    def test_provider_spelling(self, llm):
        params = build_request(llm, contents="Hello", system_instruction="sys")
        assert params["contents"] == "Hello" and params["config"]["system_instruction"] == "sys"

    def test_native_config_is_merged_and_flat_settings_win(self, llm):
        # Gemini's native ``config`` goes through others: ``config=`` is the LLMConfig argument.
        params = build_request(llm, temperature=0.1, others={"config": {"candidate_count": 1, "temperature": 0.9}},
                               response_modalities=["TEXT", "IMAGE"], speech_config={"voice_config": {}})
        assert params["config"] == {"candidate_count": 1, "temperature": 0.1, "response_modalities": ["TEXT", "IMAGE"],
                                    "speech_config": {"voice_config": {}}, **AFC_OFF}

    def test_explicit_afc_settings_are_kept(self, llm):
        params = build_request(llm, automatic_function_calling={"disable": False, "maximum_remote_calls": 3})
        assert params["config"]["automatic_function_calling"] == {"disable": False, "maximum_remote_calls": 3}

    def test_non_dict_native_config_raises(self, llm):
        with pytest.raises(TypeError, match="Gemini 'config' must be a dictionary"):
            build_request(llm, others={"config": [1]})

    def test_media_parts_come_before_the_text(self, llm):
        prompt = UserPrompt(text="Describe", images=Image(mime_type="image/png", data=b"x"),
                            audio=Audio.from_url("https://generativelanguage.googleapis.com/v1beta/files/a"
                                                 , "audio/wav"),
                            documents=Document(mime_type="application/pdf", data=b"%PDF"))
        [content] = build_request(llm, user_prompt=prompt)["contents"]
        assert content.role == "user"
        image, audio, document, prompt_text = content.parts
        assert (image.inline_data.data, image.inline_data.mime_type) == (b"x", "image/png")
        assert (audio.file_data.file_uri, audio.file_data.mime_type) == (
            "https://generativelanguage.googleapis.com/v1beta/files/a", "audio/wav")
        assert document.inline_data.mime_type == "application/pdf" and prompt_text.text == "Describe"

    def test_file_id_media_is_rejected(self, llm):
        with pytest.raises(ValueError, match="references uploaded files by URI, not file_id"):
            build_request(llm, user_prompt=UserPrompt(text="x", images=Image(mime_type="image/png", file_id="f")))

    @pytest.mark.parametrize("choice, mode", [("auto", "AUTO"), ("required", "ANY"), ("none", "NONE")])
    def test_tools(self, llm, choice, mode):
        config = build_request(llm, tools=[SPEC], tool_choice=choice)["config"]
        assert config["tools"] == [{"function_declarations": [{
            "name": "get_weather", "description": "Get weather.", "parameters_json_schema": {
                "type": "object", "properties": {"city": {"type": "string", "description": "City."}},
                "required": ["city"], "additionalProperties": False}}]}]
        assert config["tool_config"] == {"function_calling_config": {"mode": mode}} and config.update(AFC_OFF) is None

    def test_parallel_false_must_not_be_explicit(self, llm):
        with pytest.raises(ValueError, match="cannot enforce parallel_tool_calls=False"):
            build_request(llm, tools=[SPEC], parallel_tool_calls=False)
        assert build_request(llm, tools=[SPEC], parallel_tool_calls=False, tool_choice="none")
        assert build_request(llm, tools=[SPEC])

    def test_live_only_result_behavior_is_rejected(self, llm):
        with pytest.raises(ValueError, match="does not support Live-only result_behavior"):
            build_request(llm, tools=[GeminiToolSpec(**SPEC, result_behavior="blocking")])

    def test_foreign_tool_specs_are_rejected(self, llm):
        with pytest.raises(ValueError, match="Gemini tools must use ToolSpec or GeminiToolSpec"):
            build_request(llm, tools=[AnthropicToolSpec(**SPEC)])


class TestThinking:
    def config(self, llm, **values):
        return build_request(llm, **values)["config"]

    @pytest.mark.parametrize("effort, budget", [("low", 1024), ("medium", 8192), ("high", 24576)])
    def test_legacy_efforts_become_budgets(self, llm, effort, budget):
        assert self.config(llm, thinking_effort=effort)["thinking_config"] == {"thinking_budget": budget}

    def test_legacy_rejects_other_efforts(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="use low/medium/high effort"):
            self.config(llm, thinking_effort="xhigh")

    def test_newer_models_take_a_level(self, llm):
        assert self.config(llm, model="gemini-3-pro", thinking_effort="high")["thinking_config"] == {
            "thinking_level": "high"}
        with pytest.raises(UnsupportedThinkingError, match="effort='minimal' is not supported"):
            self.config(llm, model="gemini-3.1-pro", thinking_effort="minimal")

    def test_budget(self, llm):
        assert self.config(llm, thinking_budget=512)["thinking_config"] == {"thinking_budget": 512}
        assert self.config(llm, model="gemini-3-pro", thinking_budget=512,
                           thinking={"others": {"thinking_level": "low"}})["thinking_config"] == {"thinking_budget": 512}

    def test_effort_and_budget_conflict(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="choose an effort or a thinking_budget"):
            self.config(llm, thinking_effort="low", thinking_budget=512)

    def test_disabled(self, llm):
        assert self.config(llm, thinking_mode="disabled")["thinking_config"] == {"thinking_budget": 0}
        for model in ("gemini-2.5-pro", "gemini-3-flash"):
            with pytest.raises(UnsupportedThinkingError, match="cannot be disabled for this model"):
                self.config(llm, model=model, thinking_mode="disabled")
        with pytest.raises(UnsupportedThinkingError, match="cannot be disabled"):
            self.config(llm, model="gemini-3-flash", thinking_budget=0)

    def test_enabled(self, llm):
        assert self.config(llm, thinking_mode="enabled")["thinking_config"] == {"thinking_budget": -1}
        assert "thinking_config" not in self.config(llm, model="gemini-3-pro", thinking_mode="enabled")
        with pytest.raises(UnsupportedThinkingError, match="cannot determine an enabled-mode mapping"):
            self.config(llm, model="gemini-1.5-pro", thinking_mode="enabled")
        with pytest.raises(UnsupportedThinkingError, match="enabled mode conflicts with thinking_budget=0"):
            self.config(llm, thinking_mode="enabled", thinking_budget=0)

    def test_native_modes_are_rejected(self, llm):
        with pytest.raises(UnsupportedThinkingError, match="mode='adaptive' is not a native mode"):
            self.config(llm, thinking_mode="adaptive")

    def test_native_options_merge_from_every_source(self, llm):
        thinking = self.config(llm, thinking_effort="low", thinking={"others": {"include_thoughts": True}},
                               others={"config": {"thinking_config": {"a": 1}}})["thinking_config"]
        assert thinking == {"a": 1, "include_thoughts": True, "thinking_budget": 1024}


def parse(llm, response, **kwargs):
    return llm.translator.parse_response(response, **kwargs)


class TestParsing:
    def test_text_excludes_thoughts_and_fragments_merge(self, llm):
        result = parse(llm, chunk(candidate(text("plan", thought=True), text("Hel"), text("lo."), reason="STOP"),
                                  usage=usage(thoughts=2, cached=1), response_id="g1", model_version="gemini-x",
                                  create_time=datetime(2026, 1, 1, tzinfo=timezone.utc)), elapsed_seconds=1.0)
        assert (result.text, result.reasoning, result.id_, result.model) == ("Hello.", "plan", "g1", "gemini-x")
        assert [m.content.content_type for m in result.messages] == ["reasoning", "text"]
        assert result.messages[1].finish_reason == "stop" and result.status == "completed"
        assert result.usage.model_dump(exclude_none=True) == {"input_tokens": 3, "output_tokens": 4,
                                                              "reasoning_tokens": 2, "total_tokens": 7,
                                                              "cache_read_tokens": 1}
        assert result.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc) and result.timing.total_seconds == 1.0

    def test_missing_metadata_falls_back(self, llm):
        requested = datetime(2025, 5, 5, tzinfo=timezone.utc)
        result = parse(llm, chunk(candidate(text("x"), reason="STOP")), model="requested", created_at=requested)
        assert (result.id_, result.model, result.created_at, result.usage) == ("", "requested", requested, None)

    def test_only_the_first_candidate_is_used(self, llm):
        result = parse(llm, chunk(candidate(text("main"), reason="STOP"), candidate(text("alt"), index=1, reason="STOP")))
        assert result.text == "main" and len(result.messages) == 1

    @pytest.mark.parametrize("reason, finish", [("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "content_filter"),
                                                ("RECITATION", "recitation"), ("LANGUAGE", "unsupported_language"),
                                                ("MALFORMED_FUNCTION_CALL", "invalid_output"), ("OTHER", "other"),
                                                ("FINISH_REASON_UNSPECIFIED", "unknown"), (None, "unknown")])
    def test_finish_reasons(self, llm, reason, finish):
        assert parse(llm, chunk(candidate(text("x"), reason=reason))).messages[0].finish_reason == finish

    def test_max_tokens_is_incomplete_and_keeps_calls_incomplete(self, llm):
        result = parse(llm, chunk(candidate(function_call(args={"a": 1}), reason="MAX_TOKENS")))
        assert result.status == "incomplete" and result.tool_calls[0].status == "incomplete"

    def test_function_calls_keep_their_part_and_order(self, llm):
        first, second = function_call(args={"city": "Istanbul"}), function_call(args={"city": "Paris"})
        result = parse(llm, chunk(candidate(first, second, reason="STOP")))
        assert [c.arguments["city"] for c in result.tool_calls] == ["Istanbul", "Paris"]
        assert result.tool_calls[0].raw is first and result.messages[0].finish_reason == "tool_call"

    def test_missing_arguments_do_not_become_an_empty_object(self, llm):
        call = parse(llm, chunk(candidate(function_call(args=None), reason="STOP"))).tool_calls[0]
        assert (call.arguments, call.status, call.issue.code) == (None, "invalid", "missing_arguments")

    def test_partial_function_calls_are_not_executable(self, llm):
        call = parse(llm, chunk(candidate(function_call(args={"a": 1}, will_continue=True), reason="STOP"))).tool_calls[0]
        assert (call.status, call.issue.code) == ("incomplete", "unsupported_partial_arguments")

    def test_function_responses_and_media(self, llm):
        result = parse(llm, chunk(candidate(
            types.Part(function_response=types.FunctionResponse(id="r1", name="f", response={"ok": True})),
            types.Part(inline_data=types.Blob(mime_type="image/png", data=b"png")),
            types.Part(file_data=types.FileData(mime_type="audio/wav", file_uri="gs://a.wav")),
            types.Part(inline_data=types.Blob(mime_type="application/pdf", data=b"%PDF")),
            reason="STOP")))
        assert [m.content.content_type for m in result.messages] == ["tool_result", "image", "audio"]
        assert result.messages[0].content.data == {"id": "r1", "name": "f", "result": {"ok": True}}
        assert result.images[0].data == b"png" and result.audio[0].url == "gs://a.wav"

    def test_candidate_without_parts_keeps_its_finish_reason(self, llm):
        result = parse(llm, chunk(candidate(reason="SAFETY")))
        assert [(m.content.content_type, m.finish_reason) for m in result.messages] == [("other", "content_filter")]

    def test_no_candidates(self, llm):
        result = parse(llm, chunk())
        assert (result.text, result.messages, result.status) == ("", [], "completed")

    @pytest.mark.parametrize("role, expected", [("model", "assistant"), ("tool", "tool"), ("user", "other")])
    def test_roles(self, llm, role, expected):
        assert parse(llm, chunk(candidate(text("x"), reason="STOP", role=role))).messages[0].role == expected


def stream(llm, chunks):
    accumulator = llm.translator.stream_accumulator(type("C", (), {"model": "requested"})())
    return list(LLMStream(raw=iter(chunks), event_adapter=accumulator.adapt, finalizer=accumulator.finalize))


class TestStreaming:
    def test_text_accumulates_and_latest_usage_wins(self, llm):
        events = stream(llm, [chunk(candidate(text("Hel")), usage=usage(output=1, total=4), response_id="g1"),
                              chunk(candidate(text("lo."), reason="STOP"), usage=usage(output=2, total=5))])
        *deltas, terminal = events
        assert [d.delta.text for d in deltas] == ["Hel", "lo."]
        assert deltas[0].message_id == deltas[1].message_id and deltas[0].message_index == 0
        assert (terminal.response.text, terminal.response.usage.total_tokens, terminal.response_id) == ("Hello.", 5, "g1")
        assert terminal.response.model == "requested"

    def test_thoughts_and_audio_accumulate(self, llm):
        events = stream(llm, [
            chunk(candidate(text("a", thought=True), types.Part(inline_data=types.Blob(mime_type="audio/L16", data=b"\x01")))),
            chunk(candidate(types.Part(inline_data=types.Blob(mime_type="audio/L16", data=b"\x02")), reason="STOP")),
        ])
        response = events[-1].response
        assert response.reasoning == "a" and response.audio[0].data == b"\x01\x02"

    def test_alternative_candidates_do_not_change_the_primary(self, llm):
        events = stream(llm, [chunk(candidate(text("main"), reason="STOP"), candidate(text("alt"), index=1))])
        assert events[-1].response.text == "main" and events[-1].status == "completed"
        assert [e.delta.text for e in events[:-1]] == ["main"]

    def test_empty_signature_parts_add_no_deltas(self, llm):
        signed = types.Part(text="", thought_signature=b"opaque")
        events = stream(llm, [chunk(candidate(signed, reason="STOP"))])
        assert [e.event_type for e in events] == ["response.completed"]

    def test_blocked_prompt_completes(self, llm):
        blocked = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
        assert stream(llm, [blocked])[-1].status == "completed"

    def test_unfinished_or_empty_streams_are_incomplete(self, llm):
        assert stream(llm, [chunk(candidate(text("Partial")))])[-1].status == "incomplete"
        assert stream(llm, [])[-1].status == "incomplete"

    def test_max_tokens_stream_is_incomplete(self, llm):
        assert stream(llm, [chunk(candidate(text("x"), reason="MAX_TOKENS"))])[-1].status == "incomplete"

    def test_repeated_call_ids_are_snapshots_not_duplicates(self, llm):
        first = function_call(args={"city": "Istanbul"}, call_id="a")
        last = function_call(args={"city": "Paris"}, call_id="a")
        events = stream(llm, [chunk(candidate(first)), chunk(candidate(last, reason="STOP"))])
        result = events[-1].response
        assert len(result.tool_calls) == 1 and result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.tool_calls[0].raw is first and events[0].message_id == events[1].message_id

    def test_incremental_calls_without_ids_continue_the_pending_call(self, llm):
        events = stream(llm, [chunk(candidate(function_call(args={}, will_continue=True))),
                              chunk(candidate(function_call(name=None, args={"city": "Paris"}), reason="STOP"))])
        call = events[-1].response.tool_calls
        assert len(call) == 1 and (call[0].status, call[0].issue.code) == ("incomplete", "unsupported_partial_arguments")

    def test_calls_without_finish_reason_stay_incomplete(self, llm):
        events = stream(llm, [chunk(candidate(function_call(args={"a": 1})))])
        assert events[-1].response.tool_calls[0].status == "incomplete"

    def test_errors_do_not_fabricate_a_terminal_response(self, llm):
        def failing():
            yield chunk(candidate(text("partial")))
            raise ConnectionError("dropped")
        accumulator = llm.translator.stream_accumulator(type("C", (), {"model": "m"})())
        events = LLMStream(raw=failing(), event_adapter=accumulator.adapt, finalizer=accumulator.finalize)
        assert next(events).delta.text == "partial"
        with pytest.raises(ConnectionError):
            next(events)
        assert list(events) == []


class TestEdgeCases:
    def test_candidate_without_content(self, llm):
        result = parse(llm, chunk(types.Candidate(index=0, finish_reason="SAFETY")))
        assert [(m.content.content_type, m.finish_reason) for m in result.messages] == [("other", "content_filter")]

    def test_unknown_parts_are_skipped(self, llm):
        code = types.Part(executable_code=types.ExecutableCode(code="print(1)", language="PYTHON"))
        result = parse(llm, chunk(candidate(code, text("x"), reason="STOP")))
        assert [m.content.content_type for m in result.messages] == ["text"]

    def test_a_pending_call_followed_by_another_function_is_a_new_call(self, llm):
        events = stream(llm, [chunk(candidate(function_call(name="first", args={}, will_continue=True))),
                              chunk(candidate(function_call(name="second", args={"a": 1}), reason="STOP"))])
        calls = events[-1].response.tool_calls
        assert [c.name for c in calls] == ["first", "second"]
        assert calls[0].status == "incomplete" and calls[1].status == "complete"

    def test_enabled_legacy_keeps_a_native_budget(self, llm):
        params = build_request(llm, thinking_mode="enabled", thinking={"others": {"thinking_budget": 500}})
        assert params["config"]["thinking_config"] == {"thinking_budget": 500}

    def test_text_and_reasoning_alternating_stay_separate(self, llm):
        result = parse(llm, chunk(candidate(text("a"), text("t", thought=True), text("b"), reason="STOP")))
        assert [(m.content.content_type, m.content.text) for m in result.messages] == [
            ("text", "a"), ("reasoning", "t"), ("text", "b")]
        assert result.text == "ab"

    def test_audio_of_different_types_is_not_joined(self, llm):
        result = parse(llm, chunk(candidate(types.Part(inline_data=types.Blob(mime_type="audio/L16", data=b"1")),
                                            types.Part(inline_data=types.Blob(mime_type="audio/mp3", data=b"2")),
                                            reason="STOP")))
        assert [a.mime_type for a in result.audio] == ["audio/L16", "audio/mp3"]
