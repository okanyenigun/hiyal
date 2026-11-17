from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hiyal.memory.short_term import (
    KINDS,
    PAYLOADS,
    Context,
    Entry,
    EntryKind,
    Notes,
    Pin,
    Summary,
    ToolResult,
)
from hiyal.memory.short_term.schemas import MANAGED_KINDS
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import Image, SystemPrompt, UserPrompt

PNG = b"\x89PNG\r\n\x1a\n\x00\xff\xfe"


def response(**overrides):
    values = {"text": "April is best.", "messages": [], "provider": "openai", "id_": "resp-1", "model": "m",
              "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "status": "completed", "raw": {"sdk": "x"}}
    return LLMResponse(**{**values, **overrides})


def one_of_each_kind():
    """One payload of every kind an entry can hold."""
    return [
        SystemPrompt(text="Be brief."),
        UserPrompt(text="Plan a trip.", images=[Image(data=PNG, mime_type="image/png")]),
        response(),
        ToolResult(message_id="item_1", call_id="call_1", name="get_weather", arguments={"city": "Kyoto"},
                   output={"c": 21}),
        Context(text="Budget is $3,000", source="long_term_memory"),
        Summary(text="Trip to Japan in April."),
        Notes(sections={"destination": "Kyoto", "constraints": []}),
        Pin(entry_id="e-1"),
    ]


class TestKinds:
    def test_every_kind_has_one_payload_class(self):
        assert set(PAYLOADS) == set(EntryKind)

    def test_kinds_is_the_other_way_round(self):
        assert {kind: cls for cls, kind in KINDS.items()} == PAYLOADS

    def test_the_manager_makes_summaries_notes_and_pins(self):
        assert MANAGED_KINDS == {EntryKind.SUMMARY, EntryKind.NOTES, EntryKind.PIN}

    @pytest.mark.parametrize("payload", one_of_each_kind(), ids=lambda p: type(p).__name__)
    def test_of_takes_the_kind_from_the_payload(self, payload):
        entry = Entry.of(payload, turn_id="t1")
        assert PAYLOADS[entry.kind] is type(payload) and entry.payload is payload and entry.turn_id == "t1"

    def test_a_kind_that_does_not_match_the_payload_is_refused(self):
        with pytest.raises(ValidationError, match="a summary entry holds a Summary, not a SystemPrompt"):
            Entry(kind=EntryKind.SUMMARY, payload=SystemPrompt(text="Be brief."))

    def test_an_unknown_payload_class_is_refused(self):
        with pytest.raises(TypeError, match="cannot hold a str"):
            Entry.of("just text")  # type: ignore[arg-type]


class TestEnvelope:
    def test_every_entry_gets_its_own_id_and_no_turn_by_default(self):
        first, second = Entry.of(Summary(text="s")), Entry.of(Summary(text="s"))
        assert first.id_ and first.id_ != second.id_ and first.turn_id is None

    @pytest.mark.parametrize("field", ["id_", "turn_id"])
    def test_empty_ids_are_refused(self, field):
        with pytest.raises(ValidationError):
            Entry(kind=EntryKind.SUMMARY, payload=Summary(text="s"), **{field: ""})

    def test_to_text_is_the_payloads(self):
        assert Entry.of(ToolResult(message_id="m", name="get_time", arguments={}, output="12:00")).to_text() == \
            "Tool result: get_time({}) -> 12:00"


class TestJson:
    @pytest.mark.parametrize("payload", one_of_each_kind(), ids=lambda p: type(p).__name__)
    def test_every_kind_reads_back_as_its_own_class(self, payload):
        entry = Entry.of(payload, turn_id="t1")
        again = Entry.model_validate_json(entry.model_dump_json())
        assert type(again.payload) is type(payload) and again.model_dump() == entry.model_dump()

    def test_a_summary_is_not_read_back_as_a_system_prompt(self):
        again = Entry.model_validate_json(Entry.of(Summary(text="s")).model_dump_json())
        assert isinstance(again.payload, Summary)  # {"text": "s"} would fit a SystemPrompt too

    def test_media_bytes_survive(self):
        entry = Entry.of(UserPrompt(text="Our hotel.", images=[Image(data=PNG, mime_type="image/png")]))
        again = Entry.model_validate_json(entry.model_dump_json())
        assert isinstance(again.payload, UserPrompt) and again.payload.images[0].data == PNG

    def test_a_responses_raw_payload_is_not_stored(self):
        again = Entry.model_validate_json(Entry.of(response()).model_dump_json())
        assert isinstance(again.payload, LLMResponse) and again.payload.raw is None

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValidationError):
            Entry.model_validate({"kind": "memo", "payload": {"text": "x"}})

    def test_plain_data_is_rebuilt_as_the_kinds_class(self):
        entry = Entry.model_validate({"kind": "summary", "payload": {"text": "s"}})
        assert type(entry.payload) is Summary

    def test_plain_data_without_a_kind_is_refused(self):
        with pytest.raises(ValidationError, match="kind"):
            Entry.model_validate({"payload": {"text": "s"}})


class TestPayloads:
    def test_context_needs_some_content(self):
        with pytest.raises(ValidationError, match="at least one of text, data"):
            Context(source="long_term_memory")

    def test_context_text(self):
        assert Context(text="Prefers direct flights.", data={"budget": 3000}, source="ltm").to_text() == \
            'Context (ltm):\nPrefers direct flights.\n{"budget": 3000}'

    def test_context_with_only_data_or_only_media(self):
        assert Context(data={"budget": 3000}).to_text() == 'Context:\n{"budget": 3000}'
        photo = Image(data=PNG, mime_type="image/png")
        assert Context(images=[photo]).to_text() == f"Context:\n{photo.placeholder()}"

    def test_context_data_must_be_a_json_object(self):
        with pytest.raises(ValidationError):
            Context(data=["not", "an", "object"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("output, shown", [
        ({"c": 21, "sky": "clear"}, '{"c": 21, "sky": "clear"}'),
        (None, "null"),
        ([1, 2], "[1, 2]"),
        ("Güneşli", "Güneşli"),
    ])
    def test_a_tool_output_that_is_not_text_is_shown_as_json(self, output, shown):
        result = ToolResult(message_id="m", name="hava", arguments={"şehir": "İzmir"}, output=output)
        assert result.to_text() == f'Tool result: hava({{"şehir": "İzmir"}}) -> {shown}'

    @pytest.mark.parametrize("field", ["message_id", "name"])
    def test_a_tool_result_needs_its_ids(self, field):
        values = {"message_id": "m", "name": "f", "arguments": {}, "output": "x"}
        with pytest.raises(ValidationError):
            ToolResult(**{**values, field: ""})

    def test_a_failed_tool_says_so(self):
        assert ToolResult(message_id="m", name="f", arguments={"a": 1}, output="boom", is_error=True).to_text() == \
            'Tool error: f({"a": 1}) -> boom'

    def test_summary_text(self):
        assert Summary(text="Trip to Japan.").to_text() == "Summary of the earlier conversation:\nTrip to Japan."

    def test_summary_text_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            Summary(text="")

    def test_a_summary_covers_nothing_new_by_default(self):
        assert Summary(text="s").newly_covered == []

    def test_notes_show_one_bullet_per_section(self):
        notes = Notes(sections={"destination": "Kyoto", "decisions": ["Fly direct", "Stay central"],
                                "next_step": None, "budget": {"usd": 3000}})
        assert notes.to_text() == ("Working notes:\n- Destination: Kyoto\n- Decisions:\n  - Fly direct\n"
                                   '  - Stay central\n- Next step: (none)\n- Budget: {"usd": 3000}')


    def test_notes_show_empty_values_as_none(self):
        notes = Notes(sections={"a": "", "b": {}, "c": [], "d": None, "e": 0, "f": False})
        assert notes.to_text() == ("Working notes:\n- A: (none)\n- B: (none)\n- C: (none)\n- D: (none)\n"
                                   "- E: 0\n- F: false")

    def test_notes_list_items_that_are_not_text_are_json(self):
        assert Notes(sections={"stops": [{"city": "Kyoto"}, 3]}).to_text() == \
            'Working notes:\n- Stops:\n  - {"city": "Kyoto"}\n  - 3'

    def test_notes_with_no_sections(self):
        assert Notes(sections={}).to_text() == "Working notes:"

    def test_notes_are_made_by_the_llm_by_default_or_the_app(self):
        assert Notes(sections={}).source == "llm" and Notes(sections={}, source="app").source == "app"
        with pytest.raises(ValidationError):
            Notes(sections={}, source="user")  # type: ignore[arg-type]

    def test_pin_text_and_id(self):
        assert Pin(entry_id="e-1").to_text() == "Pinned entry: e-1"
        with pytest.raises(ValidationError):
            Pin(entry_id="")
