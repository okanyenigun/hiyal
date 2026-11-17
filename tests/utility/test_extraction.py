import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, Field

from hiyal.models.schemas import LLMResponse
from hiyal.utility import Extraction


class TripNotes(BaseModel):
    destination: str | None = Field(default=None, description="Where the user is going.")
    constraints: list[str] = Field(default_factory=list, description="What the plan must respect.")


class ScriptedLLM:
    """Answers with ``text`` (and ``status``) and records the calls."""

    def __init__(self, text, status="completed"):
        self.text, self.status = text, status
        self.calls = []
        self.modes = []  # "sync" or "async", per call

    def query(self, **parameters):
        self.modes.append("sync")
        return self._answer(parameters)

    async def query_async(self, **parameters):
        self.modes.append("async")
        return self._answer(parameters)

    def _answer(self, parameters):
        self.calls.append(parameters)
        return LLMResponse(text=self.text, messages=[], provider="openai", id_="resp-1", model="m",
                           created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status=self.status)


TEXT = "User: I want to go to Kyoto, direct flights only."
KYOTO = '{"destination": "Kyoto", "constraints": ["direct flights only"]}'


class Side:
    """Runs ``extract`` (``"sync"``) or ``extract_async`` (``"async"``)."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *arguments):
        if self.name == "sync":
            return Extraction.extract(*arguments)
        return asyncio.run(Extraction.extract_async(*arguments))


@pytest.fixture(params=["sync", "async"])
def extract(request):
    """A test taking ``extract`` runs twice: once with ``extract``, once with ``extract_async``."""
    return Side(request.param)


class TestExtract:
    def test_each_side_calls_its_own_client(self, extract):
        llm = ScriptedLLM(KYOTO)
        extract(llm, "Extract the trip.", TEXT, TripNotes)
        assert llm.modes == [extract.name]

    def test_the_schema_is_filled_in_from_the_text(self, extract):
        notes = extract(ScriptedLLM(KYOTO), "Extract the trip.", TEXT, TripNotes)
        assert notes == TripNotes(destination="Kyoto", constraints=["direct flights only"])

    def test_one_call_with_the_system_prompt_the_text_and_the_schema(self, extract):
        llm = ScriptedLLM(KYOTO)
        extract(llm, "Extract the trip.", TEXT, TripNotes)
        (call,) = llm.calls
        assert call["system_prompt"].text == "Extract the trip."
        user = call["user_prompt"].text
        assert user.startswith(f"Text:\n{TEXT}\n\nJSON schema:\n")
        assert json.dumps(TripNotes.model_json_schema(), indent=2) in user
        assert user.endswith("with every one of its fields (an empty value when the text has nothing "
                             "for a field), and nothing else.")

    def test_an_answer_in_a_code_fence_is_accepted(self, extract):
        notes = extract(ScriptedLLM(f"```json\n{KYOTO}\n```"), "Extract.", TEXT, TripNotes)
        assert notes.destination == "Kyoto"

    def test_an_answer_in_a_fence_without_a_language_is_accepted(self, extract):
        notes = extract(ScriptedLLM(f"```\n{KYOTO}\n```"), "Extract.", TEXT, TripNotes)
        assert notes.destination == "Kyoto"

    def test_fields_the_schema_does_not_have_are_ignored(self, extract):
        answer = '{"destination": "Kyoto", "constraints": [], "hotel": "Gracery"}'
        assert extract(ScriptedLLM(answer), "Extract.", TEXT, TripNotes) == TripNotes(destination="Kyoto")

    def test_empty_values_are_accepted(self, extract):
        notes = extract(ScriptedLLM('{"destination": null, "constraints": []}'), "Extract.",
                                   TEXT, TripNotes)
        assert notes == TripNotes()

    def test_a_field_left_out_is_refused_even_with_a_default(self, extract):
        with pytest.raises(ValueError, match="leaves out fields: constraints"):
            extract(ScriptedLLM('{"destination": "Kyoto"}'), "Extract.", TEXT, TripNotes)

    def test_a_field_with_an_alias_is_answered_under_the_alias(self, extract):
        class Step(BaseModel):
            next_step: str = Field(alias="next")

        assert extract(ScriptedLLM('{"next": "Book"}'), "Extract.", TEXT, Step).next_step == "Book"
        with pytest.raises(ValueError, match="leaves out fields: next_step"):
            extract(ScriptedLLM('{"next_step": "Book"}'), "Extract.", TEXT, Step)

    @pytest.mark.parametrize("text, message", [
        ("Kyoto, I think", "the answer is not JSON"),
        ('["Kyoto"]', "the answer is not a JSON object, but list"),
        ('{"destination": 3, "constraints": "direct"}', "the answer does not fit TripNotes"),
    ])
    def test_an_answer_that_is_not_the_schema_is_refused(self, text, message, extract):
        with pytest.raises(ValueError, match=message):
            extract(ScriptedLLM(text), "Extract.", TEXT, TripNotes)

    def test_a_cut_off_answer_is_refused(self, extract):
        with pytest.raises(ValueError, match="'incomplete', not 'completed'"):
            extract(ScriptedLLM('{"destination": "Ky', "incomplete"), "Extract.", TEXT, TripNotes)

    def test_a_schema_that_is_not_a_pydantic_model_is_refused(self, extract):
        with pytest.raises(TypeError, match="schema must be a Pydantic model class"):
            extract(ScriptedLLM(KYOTO), "Extract.", TEXT, dict)  # type: ignore[type-var]

    @pytest.mark.parametrize("text", ["", "  \n"])
    def test_a_blank_text_is_refused_with_no_call(self, text, extract):
        llm = ScriptedLLM(KYOTO)
        with pytest.raises(ValueError, match="no text to extract from"):
            extract(llm, "Extract.", text, TripNotes)
        assert llm.calls == []
