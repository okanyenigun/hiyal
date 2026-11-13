from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hiyal.memory.short_term import (
    ContextEntry,
    SummaryEntry,
    ToolResultEntry,
    WorkingNotesEntry,
)
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import Image, SystemPrompt, UserPrompt

PNG = b"\x89PNG\r\n\x1a\n\x00\xff\xfe"


def response(**overrides):
    values = {"text": "Hi.", "messages": [], "provider": "openai", "id_": "resp-1", "model": "m",
              "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "status": "completed", "raw": {"sdk": "object"}}
    return LLMResponse(**{**values, **overrides})


def every_memory_entry():
    return [
        SystemPrompt(text="Be brief."),
        UserPrompt(text="Hi"),
        ContextEntry(text="Budget is $3,000", source="long_term_memory"),
        ToolResultEntry(message_id="item_1", call_id="call_1", name="get_weather", arguments={"city": "Kyoto"},
                        output={"c": 21}),
        SummaryEntry(text="Trip to Japan in April.", covers_through="b", covers_count=2),
        WorkingNotesEntry(notes={"objective": "Compare three technologies", "decisions": []}),
    ]


class TestIds:
    def test_every_entry_gets_its_own_generated_id(self):
        first, second = every_memory_entry(), every_memory_entry()
        ids = [entry.id_ for entry in first + second]
        assert all(ids) and len(set(ids)) == len(ids)

    @pytest.mark.parametrize("entry", every_memory_entry(), ids=lambda entry: type(entry).__name__)
    def test_empty_id_is_rejected(self, entry):
        with pytest.raises(ValidationError):
            type(entry).model_validate({**entry.model_dump(), "id_": ""})


class TestContextEntry:
    def test_content_is_required(self):
        with pytest.raises(ValidationError, match="at least one of"):
            ContextEntry(source="long_term_memory")

    @pytest.mark.parametrize("content", [
        {"text": "Budget is $3,000"},
        {"data": {"budget": 3000, "cities": ["Tokyo", "Kyoto"], "nested": {"direct": True}}},
        {"images": [Image(data=PNG, mime_type="image/png")]},
    ], ids=["text", "data", "image"])
    def test_any_single_kind_of_content_is_enough(self, content):
        assert ContextEntry(**content).id_

    def test_data_must_be_json_compatible(self):
        with pytest.raises(ValidationError):
            ContextEntry(data={"when": object()})


class TestJsonRoundTrip:
    @pytest.mark.parametrize("entry", every_memory_entry()[:5], ids=lambda entry: type(entry).__name__)
    def test_entries_survive_json(self, entry):
        assert type(entry).model_validate_json(entry.model_dump_json()) == entry

    def test_media_bytes_survive_json(self):
        entry = ContextEntry(images=[Image(data=PNG, mime_type="image/png")])
        assert ContextEntry.model_validate_json(entry.model_dump_json()).images[0].data == PNG

    def test_response_survives_json_without_raw(self):
        original = response()
        reloaded = LLMResponse.model_validate_json(original.model_dump_json())
        assert reloaded.raw is None
        assert reloaded.model_dump() == original.model_dump()

    def test_notes_survive_json_without_knowing_any_class(self):
        entry = WorkingNotesEntry(notes={"objective": "Compare", "decisions": ["Exclude large-scale"],
                                         "budget": {"usd": 3000}, "next_step": None})
        assert WorkingNotesEntry.model_validate_json(entry.model_dump_json()) == entry

    def test_notes_must_be_json(self):
        with pytest.raises(ValidationError):
            WorkingNotesEntry(notes={"when": object()})


class TestToText:
    def test_user_prompt_shows_its_text_and_media_placeholders(self):
        prompt = UserPrompt(text="What is this?", images=[Image(data=PNG, mime_type="image/png", filename="map.png")])
        assert prompt.to_text() == "User: What is this?\n[image: map.png (image/png)]"

    def test_prompts_without_media(self):
        assert UserPrompt(text="Hi").to_text() == "User: Hi"
        assert SystemPrompt(text="Be brief.").to_text() == "System: Be brief."

    def test_context_shows_source_text_data_and_media(self):
        entry = ContextEntry(text="Prefers direct flights.", data={"budget": 3000}, source="long_term_memory",
                             images=[Image(data=PNG, mime_type="image/png")])
        assert entry.to_text() == ('Context (long_term_memory):\nPrefers direct flights.\n{"budget": 3000}\n'
                                   "[image: image.png (image/png)]")

    def test_context_without_source(self):
        assert ContextEntry(text="a").to_text() == "Context:\na"

    def test_tool_result_and_error_show_the_call_with_its_arguments(self):
        result = ToolResultEntry(message_id="m", name="get_weather", arguments={"city": "Kyoto"}, output={"c": 21})
        assert result.to_text() == 'Tool result: get_weather({"city": "Kyoto"}) -> {"c": 21}'
        error = ToolResultEntry(message_id="m", name="get_weather", arguments={"city": "Kyoto"}, output="Timeout",
                                is_error=True)
        assert error.to_text() == 'Tool error: get_weather({"city": "Kyoto"}) -> Timeout'

    def test_tool_result_without_arguments(self):
        result = ToolResultEntry(message_id="m", name="get_time", arguments={}, output="12:00")
        assert result.to_text() == "Tool result: get_time({}) -> 12:00"

    def test_tool_result_needs_its_arguments(self):
        with pytest.raises(ValidationError, match="arguments"):
            ToolResultEntry(message_id="m", name="get_weather", output="Mild")

    def test_summary(self):
        assert SummaryEntry(text="Trip to Japan.", covers_through="a", covers_count=1).to_text() == \
            "Summary of the earlier conversation:\nTrip to Japan."

    def test_a_summary_covers_everything_up_to_its_last_entry_except_its_exceptions(self):
        summary = SummaryEntry(text="s", covers_through="d", covers_except=["a", "c"], covers_count=2)
        assert summary.covered(["a", "b", "c", "d", "e"]) == ["b", "d"]

    def test_a_summary_whose_last_entry_was_removed_covers_nothing_it_can_tell(self):
        summary = SummaryEntry(text="s", covers_through="d", covers_count=4)
        assert summary.covered(["a", "b", "c", "e"]) == []

    def test_notes_show_one_bullet_per_section(self):
        entry = WorkingNotesEntry(notes={"destination": "Kyoto", "decisions": ["Fly direct", "Stay central"],
                                         "next_step": None, "budget": {"usd": 3000}})
        assert entry.to_text() == ("Working notes:\n- Destination: Kyoto\n- Decisions:\n  - Fly direct\n"
                                   '  - Stay central\n- Next step: (none)\n- Budget: {"usd": 3000}')
