import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, Field

from hiyal.memory.short_term import (
    BaseNoteTaker,
    ContextEntry,
    NoteTaker,
    NotesUpdate,
)
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import UserPrompt
from hiyal.prompts.hub import hub


DEFAULT_NOTES_INSTRUCTIONS = hub.get("memory/notes", "v1")


class TripNotes(BaseModel):
    """A schema for the LLM updater; the notes themselves are stored as a dict."""

    destination: str = Field(description="Where the user is going.")
    constraints: list[str] = Field(default_factory=list, description="Limits the plan must respect.")
    next_step: str | None = Field(default=None, description="What to do next.")


class FakeLLM:
    """Answers each query with the next prepared text, and records the queries."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def query(self, **parameters):
        self.calls.append(parameters)
        return LLMResponse(text=self.answers.pop(0), messages=[], provider="openai", id_=f"r{len(self.calls)}",
                           model="m", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")

    async def query_async(self, **parameters):
        return self.query(**parameters)


@pytest.fixture(autouse=True, params=["sync", "async"])
def mode(request, monkeypatch):
    """Run every test twice: through ``update``, then through ``update_async``, which must behave the same."""
    if request.param == "async":
        original = NoteTaker.update_async
        monkeypatch.setattr(NoteTaker, "update", lambda self, notes, entries: asyncio.run(original(self, notes, entries)))
    return request.param


def context(text):
    return ContextEntry(text=text)


def notes_json(**values):
    return TripNotes(**values).model_dump_json()


class TestNoteTaker:
    def test_the_schema_must_be_a_pydantic_model_class(self):
        for schema in (TripNotes(destination="x"), dict, "TripNotes"):
            with pytest.raises(TypeError, match="Pydantic model class"):
                NoteTaker(FakeLLM(), schema=schema)

    def test_writes_the_first_notes_from_the_entries(self):
        llm = FakeLLM(notes_json(destination="Kyoto", constraints=["Direct flights only"]))
        update = NoteTaker(llm, schema=TripNotes).update(None, [UserPrompt(text="Plan a Kyoto trip; direct flights only.")])
        assert isinstance(update, NotesUpdate) and update.response is not None
        assert update.notes == {"destination": "Kyoto", "constraints": ["Direct flights only"], "next_step": None}
        assert "Current notes:\n{}" in llm.calls[0]["user_prompt"].text

    def test_sends_the_notes_the_schema_and_the_entries(self):
        llm = FakeLLM(notes_json(destination="Kyoto", constraints=["Daytime flights only"]))
        NoteTaker(llm, schema=TripNotes).update({"destination": "Kyoto"}, [UserPrompt(text="No overnight flights.")])
        call = llm.calls[0]
        assert call["system_prompt"].text == DEFAULT_NOTES_INSTRUCTIONS
        prompt = call["user_prompt"].text
        assert '"destination": "Kyoto"' in prompt and '"description": "Limits the plan must respect."' in prompt
        assert "User: No overnight flights." in prompt

    @pytest.mark.parametrize("wrap", [lambda s: s, lambda s: f"```json\n{s}\n```", lambda s: f"```\n{s}\n```"])
    def test_reads_plain_or_fenced_json(self, wrap):
        llm = FakeLLM(wrap(notes_json(destination="Kyoto", next_step="Book hotels")))
        assert NoteTaker(llm, schema=TripNotes).update(None, [context("a")]).notes["next_step"] == "Book hotels"

    def test_an_answer_that_does_not_fit_the_schema_raises(self):
        llm = FakeLLM(json.dumps({"destination": ["not", "a", "string"]}))
        with pytest.raises(ValueError, match="do not fit TripNotes"):
            NoteTaker(llm, schema=TripNotes).update({"destination": "Kyoto"}, [context("a")])

    def test_nothing_to_update_with(self):
        llm = FakeLLM()
        with pytest.raises(ValueError, match="no entries"):
            NoteTaker(llm, schema=TripNotes).update({"destination": "Kyoto"}, [])
        assert llm.calls == []


class TestLeftOutSections:
    """A section that had content must be in the answer; emptying it on purpose is fine."""

    NOTES = {"destination": "Kyoto", "constraints": ["Direct flights only"], "next_step": "Book hotels"}

    def test_an_answer_that_leaves_out_sections_with_content_is_refused(self):
        llm = FakeLLM(json.dumps({"destination": "Kyoto"}))  # fits TripNotes: the other sections have defaults
        with pytest.raises(ValueError, match="leave out sections that had content: constraints, next_step"):
            NoteTaker(llm, schema=TripNotes).update(self.NOTES, [context("a")])

    def test_a_section_emptied_on_purpose_is_accepted(self):
        llm = FakeLLM(json.dumps({"destination": "Kyoto", "constraints": [], "next_step": None}))
        update = NoteTaker(llm, schema=TripNotes).update(self.NOTES, [context("Hotels booked; any flight is fine.")])
        assert update.notes == {"destination": "Kyoto", "constraints": [], "next_step": None}

    def test_left_out_sections_that_were_empty_are_fine(self):
        llm = FakeLLM(json.dumps({"destination": "Osaka"}))
        update = NoteTaker(llm, schema=TripNotes).update({"destination": "Kyoto", "constraints": []}, [context("a")])
        assert update.notes["destination"] == "Osaka"

    def test_the_first_notes_have_nothing_to_lose(self):
        update = NoteTaker(FakeLLM(json.dumps({"destination": "Kyoto"})), schema=TripNotes).update(None, [context("a")])
        assert update.notes["destination"] == "Kyoto"


class TestAsync:
    def test_a_note_taker_without_its_own_async_update_runs_update_in_a_thread(self):
        class Fixed(BaseNoteTaker):
            def update(self, notes, entries):
                return NotesUpdate({"destination": "Kyoto"}, None)

        update = asyncio.run(Fixed(FakeLLM(), schema=TripNotes).update_async(None, [context("a")]))
        assert update.notes == {"destination": "Kyoto"}
