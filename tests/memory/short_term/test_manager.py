import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from hiyal.memory.short_term import (
    Context,
    Entry,
    EntryKind,
    HeadTailPolicy,
    InMemoryStorage,
    LastNPolicy,
    NoNotesSettingsError,
    Notes,
    Pin,
    PredicateFilterPolicy,
    ShortTermMemoryManager,
    SingleCallSummarizer,
    Summary,
    SummaryWorker,
)
from hiyal.models.schemas import LLMResponse, ResponseMessage, ToolCallContent
from hiyal.memory.short_term.manager import SummaryKeeper
from hiyal.prompts import SystemPrompt
from hiyal.prompts.hub import hub
from hiyal.tools import ToolCall
from hiyal.utility import Tokens


class AsyncDriven:
    """A manager driven through its async side: opened with ``open_async``, and each method or read runs its ``_async`` twin.

    The ``side`` fixture swaps it in for ``ShortTermMemoryManager``, so every
    test written for the sync side checks the async side too.
    """

    manager_class = ShortTermMemoryManager
    READS = {"active", "window", "pinned", "system_prompt", "summary", "notes", "entries"}
    WRITES = {"add", "remove", "unpin", "compact", "update_notes", "set_notes", "rewind", "set_policy", "dump"}

    def __init__(self, storage, **settings):
        object.__setattr__(self, "manager", asyncio.run(self.manager_class.open_async(storage, **settings)))

    def __getattr__(self, name):
        if name in self.READS:
            return asyncio.run(getattr(self.manager, f"{name}_async")())
        if name in self.WRITES:
            twin = getattr(self.manager, f"{name}_async")
            return lambda *args, **kwargs: asyncio.run(twin(*args, **kwargs))
        return getattr(self.manager, name)

    def __setattr__(self, name, value):
        setattr(self.manager, name, value)


def refuse_sync(self, **parameters):
    raise AssertionError("the async side must call query_async")


@pytest.fixture(autouse=True, params=["sync", "async"])
def side(request, monkeypatch):
    """Every test runs twice: on the manager's sync side, then on its async side (``AsyncDriven``)."""
    if request.param == "async":
        monkeypatch.setitem(globals(), "ShortTermMemoryManager", AsyncDriven)
        monkeypatch.setattr(ScriptedLLM, "query", refuse_sync)
    return request.param


class TestSession:
    def test_a_new_session_gets_a_generated_uuid(self):
        first = ShortTermMemoryManager(InMemoryStorage())
        second = ShortTermMemoryManager(InMemoryStorage())
        assert str(UUID(first.session_id)) == first.session_id
        assert first.session_id != second.session_id

    def test_a_given_session_id_is_kept(self):
        assert ShortTermMemoryManager(InMemoryStorage(), session_id="trip-42").session_id == "trip-42"

    def test_the_storage_is_kept(self):
        storage = InMemoryStorage()
        assert ShortTermMemoryManager(storage).storage is storage

    def test_the_session_id_cannot_be_changed(self):
        manager = ShortTermMemoryManager(InMemoryStorage(), session_id="trip-42")
        with pytest.raises(AttributeError):
            manager.session_id = "other"  # type: ignore[misc]

    def test_an_empty_session_id_is_refused(self):
        with pytest.raises(ValueError, match="session_id must not be empty"):
            ShortTermMemoryManager(InMemoryStorage(), session_id="")

    def test_a_session_id_that_is_not_a_string_is_refused(self):
        with pytest.raises(TypeError, match="session_id must be a string, got int"):
            ShortTermMemoryManager(InMemoryStorage(), session_id=42)  # type: ignore[arg-type]

    def test_a_storage_that_is_not_a_storage_is_refused(self):
        with pytest.raises(TypeError, match="storage must be a BaseStorage, got dict"):
            ShortTermMemoryManager({})  # type: ignore[arg-type]


class Trip(BaseModel):
    destination: str | None = None


LLM = object()  # the settings are only stored, never called


def manager(**settings):
    return ShortTermMemoryManager(InMemoryStorage(), **settings)


class TestSettings:
    def test_with_no_settings_every_feature_is_off(self):
        made = manager()
        assert made.policy is None
        assert made.summarizer is None
        assert (made.notes_llm, made.notes_schema, made.notes_every) == (None, None, None)
        assert (made.dump_storage, made.dump_every) == (None, None)

    def test_the_settings_are_kept(self):
        policy, summarizer, archive = LastNPolicy(5), SingleCallSummarizer(LLM), InMemoryStorage()  # type: ignore[arg-type]
        made = manager(policy=policy, summarizer=summarizer, notes_llm=LLM, notes_schema=Trip, notes_every=2,
                       dump_storage=archive, dump_every=10)
        assert made.policy is policy and made.summarizer is summarizer
        assert (made.notes_llm, made.notes_schema, made.notes_every) == (LLM, Trip, 2)
        assert (made.dump_storage, made.dump_every) == (archive, 10)

    @pytest.mark.parametrize("settings, message", [
        ({"policy": "LastNPolicy(5)"}, "policy must be a Policy, got str"),
        ({"summarizer": LLM}, "summarizer must be a SummaryWorker, got object"),
        ({"dump_storage": {}}, "dump_storage must be a BaseStorage, got dict"),
        ({"notes_llm": LLM, "notes_schema": dict}, "notes_schema must be a Pydantic model class"),
    ])
    def test_a_setting_of_the_wrong_type_is_refused(self, settings, message):
        with pytest.raises(TypeError, match=message):
            manager(**settings)

    @pytest.mark.parametrize("settings, message", [
        ({"notes_llm": LLM, "notes_schema": Trip, "notes_every": 0}, "notes_every must be a positive integer"),
        ({"dump_storage": InMemoryStorage(), "dump_every": -1}, "dump_every must be a positive integer"),
        ({"notes_llm": LLM}, "notes_llm and notes_schema go together"),
        ({"notes_schema": Trip}, "notes_llm and notes_schema go together"),
        ({"notes_every": 3}, "notes_every needs notes_llm and notes_schema"),
        ({"dump_every": 10}, "dump_every needs a dump_storage"),
    ])
    def test_settings_that_do_not_fit_are_refused(self, settings, message):
        with pytest.raises(ValueError, match=message):
            manager(**settings)

    def test_the_notes_instructions_are_read_from_the_hub(self):
        assert manager().notes_instructions == hub.get("memory/notes", "v1")
        chosen = manager(notes_prompt_name="memory/condense_notes", notes_prompt_version="v1")
        assert chosen.notes_instructions == hub.get("memory/condense_notes", "v1")

    def test_a_notes_prompt_the_hub_does_not_have_is_refused(self):
        with pytest.raises(ValueError, match="'memory/notes' has no version 'v9'"):
            manager(notes_prompt_version="v9")


def note(name, text=None):
    """A context entry with ID ``name``."""
    return Entry.of(Context(text=text or name)).model_copy(update={"id_": name})


class TestAdd:
    def test_an_entry_is_stored_in_the_session(self):
        made = manager(session_id="trip")
        assert made.add(note("a")) is True
        assert made.add(note("b")) is True
        assert [entry.id_ for entry in made.storage.entries("trip")] == ["a", "b"]

    def test_other_sessions_in_the_storage_are_not_touched(self):
        storage = InMemoryStorage()
        ShortTermMemoryManager(storage, session_id="trip").add(note("a"))
        assert storage.entries("other") == []

    def test_the_same_entry_again_is_a_repeat(self):
        made = manager(session_id="trip")
        made.add(note("a"))
        assert made.add(note("a")) is False
        assert len(made.storage.entries("trip")) == 1

    def test_a_different_entry_with_a_stored_id_is_refused(self):
        made = manager(session_id="trip")
        made.add(note("a"))
        with pytest.raises(ValueError, match="another entry with id_ 'a'"):
            made.add(note("a", text="something else"))

    @pytest.mark.parametrize(
        "payload", [Summary(text="Earlier: trip planning."), Notes(sections={"budget": 3000}), Pin(entry_id="a")]
    )
    def test_summary_notes_and_pin_entries_are_refused(self, payload):
        made = manager(session_id="trip")
        with pytest.raises(ValueError, match="come from the manager, not from add"):
            made.add(Entry.of(payload))
        assert made.storage.entries("trip") == []

    def test_only_an_entry_can_be_added(self):
        with pytest.raises(TypeError, match="add takes an Entry, not a Context"):
            manager().add(Context(text="a"))  # type: ignore[arg-type]



def summary(text):
    return Entry.of(Summary(text=text))


def notes(destination):
    return Entry.of(Notes(sections={"destination": destination}))


def ids(entries):
    return [entry.id_ for entry in entries]


def opened(*stored, policy=None):
    """A manager opened on session "trip" of a storage already holding ``stored``; it reads them when it starts.
    """
    storage = InMemoryStorage()
    for entry in stored:
        storage.add("trip", entry)
    return ShortTermMemoryManager(storage, session_id="trip", policy=policy)


class CountingStorage(InMemoryStorage):
    """Counts the reads that go through the whole session: ``index`` and ``entries``."""

    whole_reads = 0

    def index(self, session_id):
        self.whole_reads += 1
        return super().index(session_id)

    def entries(self, session_id):
        self.whole_reads += 1
        return super().entries(session_id)


class TestReads:
    def test_a_new_session_reads_empty(self):
        made = manager()
        assert (made.entries, made.window, made.pinned, made.active) == ([], [], [], [])
        assert (made.summary, made.notes) == (None, None)

    def test_the_history_is_what_was_added_without_summaries_and_notes(self):
        made = opened(note("a"), summary("Earlier: a."), note("b"), notes("Kyoto"))
        assert ids(made.entries) == ["a", "b"]

    def test_without_a_policy_the_window_is_the_whole_history(self):
        made = manager()
        for name in "abc":
            made.add(note(name))
        assert ids(made.window) == ["a", "b", "c"]

    def test_with_a_policy_the_window_is_its_choice(self):
        made = manager(policy=PredicateFilterPolicy(lambda entry: entry.id_ != "c").then(LastNPolicy(2)))
        for name in "abcd":
            made.add(note(name))
        assert ids(made.window) == ["b", "d"]

    def test_the_summary_and_the_notes_are_the_latest_ones(self):
        first, second, paris, kyoto = summary("first"), summary("second"), notes("Paris"), notes("Kyoto")
        made = opened(first, paris, second, kyoto)
        assert (made.summary, made.notes) == (second, kyoto)

    def test_active_is_the_summary_the_notes_the_pinned_then_the_window(self):
        earlier, kyoto = summary("Earlier: a."), notes("Kyoto")
        made = opened(note("a"), earlier, kyoto, note("b"), policy=LastNPolicy(1))
        made.add(note("rule"), pinned=True)
        made.add(note("c"))
        assert ids(made.active) == [earlier.id_, kyoto.id_, "rule", "c"]

    def test_only_what_exists_leads_active(self):
        kyoto = notes("Kyoto")
        assert ids(opened(note("a"), kyoto).active) == [kyoto.id_, "a"]

    def test_a_manager_on_a_stored_session_reads_it(self):
        storage = InMemoryStorage()
        ShortTermMemoryManager(storage, session_id="trip").add(note("a"))
        assert ids(ShortTermMemoryManager(storage, session_id="trip").entries) == ["a"]

    def test_reads_do_not_go_through_the_whole_session(self):
        storage, earlier, kyoto = CountingStorage(), summary("Earlier."), notes("Kyoto")
        for entry in (note("a"), earlier, kyoto, note("b"), note("c")):
            storage.add("trip", entry)
        made = ShortTermMemoryManager(storage, session_id="trip", policy=LastNPolicy(2))
        storage.whole_reads = 0
        for _ in range(3):
            assert (made.summary, made.notes, made.pinned) == (earlier, kyoto, [])
            assert ids(made.active) == [earlier.id_, kyoto.id_, "b", "c"]
        assert storage.whole_reads == 0


class TestPinning:
    def test_a_pin_record_is_stored_right_after_the_entry(self):
        made = manager(session_id="trip")
        made.add(note("rule"), pinned=True)
        rule, record = made.storage.entries("trip")
        assert rule.id_ == "rule" and record.payload == Pin(entry_id="rule")
        assert ids(made.entries) == ["rule"]

    def test_a_pinned_entry_stays_in_active_after_the_window_leaves_it(self):
        made = manager(policy=LastNPolicy(1))
        made.add(note("rule"), pinned=True)
        for name in "abc":
            made.add(note(name))
        assert ids(made.pinned) == ["rule"]
        assert ids(made.window) == ["c"]
        assert ids(made.active) == ["rule", "c"]

    def test_a_pinned_entry_takes_no_place_in_the_window(self):
        made = manager(policy=LastNPolicy(2))
        made.add(note("a"))
        made.add(note("rule"), pinned=True)
        made.add(note("b"))
        assert ids(made.window) == ["a", "b"]

    def test_a_filter_does_not_hide_a_pinned_entry(self):
        made = manager(policy=PredicateFilterPolicy(lambda entry: entry.id_ != "rule"))
        made.add(note("rule"), pinned=True)
        assert ids(made.active) == ["rule"]

    def test_pinned_entries_are_oldest_first_and_still_in_the_history(self):
        made = manager()
        made.add(note("first"), pinned=True)
        made.add(note("a"))
        made.add(note("second"), pinned=True)
        assert ids(made.pinned) == ["first", "second"]
        assert ids(made.entries) == ["first", "a", "second"]

    def test_a_repeat_does_not_pin(self):
        made = manager()
        made.add(note("a"))
        assert made.add(note("a"), pinned=True) is False
        assert made.pinned == []


def archived(archive, session_id="trip"):
    return ids(archive.entries(session_id))


class TestDump:
    def test_without_a_dump_storage_dump_is_refused(self):
        with pytest.raises(ValueError, match="no dump_storage"):
            manager().dump()

    def test_the_entries_are_copied_in_order_under_the_same_session(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive)
        made.add(note("a"))
        made.add(note("b"))
        assert ids(made.dump()) == ["a", "b"]
        assert archived(archive) == ["a", "b"]
        assert ids(made.storage.entries("trip")) == ["a", "b"]  # the session's storage is left as it is

    def test_a_dump_copies_only_what_is_new(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive)
        made.add(note("a"))
        made.dump()
        made.add(note("b"))
        assert ids(made.dump()) == ["b"]
        assert made.dump() == []
        assert archived(archive) == ["a", "b"]

    def test_everything_stored_is_copied_summaries_notes_and_pins_too(self):
        storage, archive = InMemoryStorage(), InMemoryStorage()
        earlier, kyoto = summary("Earlier."), notes("Kyoto")
        for entry in (note("a"), earlier, kyoto):
            storage.add("trip", entry)
        made = ShortTermMemoryManager(storage, session_id="trip", dump_storage=archive)
        made.add(note("rule"), pinned=True)
        made.dump()
        assert archive.entries("trip") == made.storage.entries("trip")
        assert [entry.kind.value for entry in archive.entries("trip")] == ["context", "summary", "notes", "context", "pin"]

    def test_every_n_adds_the_new_entries_are_dumped(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive, dump_every=2)
        made.add(note("a"))
        assert archived(archive) == []
        made.add(note("b"))
        assert archived(archive) == ["a", "b"]
        made.add(note("c"))
        assert archived(archive) == ["a", "b"]
        made.add(note("d"))
        assert archived(archive) == ["a", "b", "c", "d"]

    def test_a_dump_on_demand_starts_the_count_again(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive, dump_every=2)
        made.add(note("a"))
        made.dump()
        made.add(note("b"))
        assert archived(archive) == ["a"]
        made.add(note("c"))
        assert archived(archive) == ["a", "b", "c"]

    def test_a_repeat_is_not_an_add(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive, dump_every=2)
        made.add(note("a"))
        made.add(note("a"))
        assert archived(archive) == []


class TripNotes(BaseModel):
    destination: str | None = None
    budget: int | None = None
    constraints: list[str] = Field(default_factory=list)
    next_step: str | None = Field(default=None, alias="next")


EMPTY = {"destination": None, "budget": None, "constraints": [], "next_step": None}


def with_notes(storage=None):
    return ShortTermMemoryManager(storage or InMemoryStorage(), session_id="trip", notes_llm=LLM,  # type: ignore[arg-type]
                                  notes_schema=TripNotes)


class TestSetNotes:
    def test_the_first_version_holds_every_section(self):
        made = with_notes()
        entry = made.set_notes("destination", "Kyoto")
        assert entry.payload == Notes(sections={**EMPTY, "destination": "Kyoto"}, source="app")
        assert made.notes == entry
        assert made.active == [entry]
        assert made.storage.entries("trip") == [entry]

    def test_each_set_is_a_new_version_that_keeps_the_other_sections(self):
        made = with_notes()
        first = made.set_notes("destination", "Kyoto")
        second = made.set_notes("budget", 3000)
        assert second.payload == Notes(sections={**EMPTY, "destination": "Kyoto", "budget": 3000}, source="app")
        assert made.notes == second
        assert made.storage.entries("trip") == [first, second]

    def test_the_value_is_stored_as_the_schema_writes_it(self):
        assert with_notes().set_notes("budget", "3000").payload == Notes(sections={**EMPTY, "budget": 3000}, source="app")

    def test_a_section_with_an_alias_is_set_by_its_field_name(self):
        notes = with_notes().set_notes("next_step", "Book the ryokan")
        assert notes.payload == Notes(sections={**EMPTY, "next_step": "Book the ryokan"}, source="app")

    def test_a_manager_on_a_stored_session_sets_on_its_notes(self):
        storage = InMemoryStorage()
        storage.add("trip", Entry.of(Notes(sections={**EMPTY, "destination": "Paris"})))
        notes = with_notes(storage).set_notes("budget", 3000)
        assert notes.payload == Notes(sections={**EMPTY, "destination": "Paris", "budget": 3000}, source="app")

    def test_without_notes_settings_it_is_refused(self):
        with pytest.raises(NoNotesSettingsError, match="no notes settings"):
            manager().set_notes("destination", "Kyoto")

    def test_a_section_that_is_not_in_the_schema_is_refused(self):
        made = with_notes()
        with pytest.raises(ValueError, match="'hotel' is not a section of TripNotes: destination, budget"):
            made.set_notes("hotel", "Ryokan")
        assert made.notes is None

    def test_a_value_that_does_not_fit_is_refused_and_nothing_changes(self):
        made = with_notes()
        kept = made.set_notes("destination", "Kyoto")
        with pytest.raises(ValueError, match="'cheap' does not fit section 'budget' of TripNotes"):
            made.set_notes("budget", "cheap")
        assert made.notes == kept
        assert made.storage.entries("trip") == [kept]


class TestChooseWindow:
    def test_the_window_is_chosen_again_when_an_entry_arrives(self):
        made = manager(policy=LastNPolicy(2))
        made.add(note("a"))
        assert ids(made.window) == ["a"]
        made.add(note("b"))
        made.add(note("c"))
        assert ids(made.window) == ["b", "c"]

    def test_a_manager_on_a_stored_session_chooses_its_window_when_it_starts(self):
        assert ids(opened(note("a"), note("b"), note("c"), policy=LastNPolicy(2)).window) == ["b", "c"]

    def test_the_policy_chooses_only_from_the_history(self):
        seen = []

        def keep(entry):
            seen.append(entry.id_)
            return True

        made = opened(note("a"), summary("Earlier."), notes("Kyoto"), policy=PredicateFilterPolicy(keep))
        made.add(note("rule"), pinned=True)
        assert set(seen) == {"a"}  # no summary, notes or pinned entry


def llm_response(text, messages=()):
    return LLMResponse(text=text, messages=list(messages), provider="openai", id_="resp", model="m",
                       created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")


def answer(name, calls=()):
    """The model's answer with ID ``name``; with tool ``calls``, the turn goes on."""
    messages = [ResponseMessage(id=call, role="assistant", content=ToolCallContent(
        data=ToolCall(name="search", arguments={}, status="complete", raw=None))) for call in calls]
    return Entry.of(llm_response(name, messages)).model_copy(update={"id_": name})


class ScriptedLLM:
    """Answers each call with the next of ``answers`` and records the calls."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def query(self, **parameters):
        return self._answer(parameters)

    async def query_async(self, **parameters):
        return self._answer(parameters)

    def _answer(self, parameters):
        self.calls.append(parameters)
        return llm_response(self.answers.pop(0))


def notes_answer(**sections):
    """The LLM's answer: the complete notes, under the schema's names (``next`` is ``next_step``'s alias)."""
    return json.dumps({"destination": None, "budget": None, "constraints": [], "next": None, **sections})


def noting(llm, storage=None, **settings):
    return ShortTermMemoryManager(storage if storage is not None else InMemoryStorage(), session_id="trip",
                                  notes_llm=llm, notes_schema=TripNotes, **settings)


class TestUpdateNotes:
    def test_without_notes_settings_the_error_is_a_value_error(self):
        assert issubclass(NoNotesSettingsError, ValueError)

    def test_the_answer_becomes_the_new_notes_version(self):
        made = noting(ScriptedLLM(notes_answer(destination="Kyoto")))
        made.add(note("a", "I want to go to Kyoto."))
        entry = made.update_notes()
        assert entry.payload == Notes(sections={**EMPTY, "destination": "Kyoto"})
        assert made.notes == entry

    def test_the_llm_gets_the_notes_instructions_the_notes_and_the_entries(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm)
        made.add(note("a", "I want to go to Kyoto."))
        made.update_notes()
        (call,) = llm.calls
        assert call["system_prompt"].text == made.notes_instructions
        assert call["user_prompt"].text.startswith(
            "Text:\nCurrent notes:\n{}\n\nConversation since the last update:\nContext:\nI want to go to Kyoto.\n\n"
        )

    def test_the_next_update_gets_only_the_entries_since_the_last_one(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"), notes_answer(destination="Kyoto", budget=3000))
        made = noting(llm)
        made.add(note("a", "I want to go to Kyoto."))
        made.update_notes()
        made.add(note("b", "My budget is 3000."))
        made.update_notes()
        prompt = llm.calls[1]["user_prompt"].text
        assert "Current notes:\n" + json.dumps({**EMPTY, "destination": "Kyoto"}, indent=2) in prompt
        assert "My budget is 3000." in prompt and "I want to go to Kyoto." not in prompt

    def test_a_version_set_by_the_app_is_not_an_update(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto", budget=3000))
        made = noting(llm)
        made.add(note("a", "I want to go to Kyoto."))
        made.set_notes("budget", 3000)
        made.update_notes()
        prompt = llm.calls[0]["user_prompt"].text
        assert '"budget": 3000' in prompt and "I want to go to Kyoto." in prompt

    def test_without_notes_settings_it_is_refused(self):
        with pytest.raises(NoNotesSettingsError, match="no notes settings"):
            manager().update_notes()

    def test_with_nothing_new_it_is_refused(self):
        made = noting(ScriptedLLM(notes_answer(destination="Kyoto")))
        with pytest.raises(ValueError, match="nothing arrived since the last notes update"):
            made.update_notes()
        made.add(note("a"))
        made.update_notes()
        with pytest.raises(ValueError, match="nothing arrived"):
            made.update_notes()

    def test_after_a_bad_answer_the_notes_and_the_entries_wait_for_the_next_update(self):
        llm = ScriptedLLM('{"destination": "Kyoto"}', notes_answer(destination="Kyoto"))
        made = noting(llm)
        made.add(note("a", "I want to go to Kyoto."))
        with pytest.raises(ValueError, match="leaves out fields"):
            made.update_notes()
        assert made.notes is None
        made.update_notes()
        assert "I want to go to Kyoto." in llm.calls[1]["user_prompt"].text

    def test_a_repeated_answer_is_not_another_finished_turn(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm, notes_every=2)
        made.add(note("q1"))
        made.add(answer("a1"))
        assert made.add(answer("a1")) is False  # a retried add
        assert llm.calls == []
        made.add(note("q2"))
        made.add(answer("a2"))  # the second turn finishes
        assert len(llm.calls) == 1

    def test_every_n_finished_turns_the_notes_are_updated(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm, notes_every=2)
        made.add(note("q1"))
        made.add(answer("a1", calls=["c1"]))  # asks for a tool: the turn goes on
        made.add(answer("a2"))  # the first turn finishes
        made.add(note("q2"))
        assert llm.calls == []
        made.add(answer("a3"))  # the second turn finishes
        assert len(llm.calls) == 1
        assert made.notes is not None
        made.add(answer("a4"))  # the count starts again
        assert len(llm.calls) == 1


class Recorder(SummaryWorker):
    """Answers "summary 1", "summary 2"... and records the texts it got; fails when ``fail`` is set."""

    def __init__(self):
        self.texts, self.fail = [], False

    def _summarize(self, text):
        if self.fail:
            raise ValueError("the model is down")
        self.texts.append(text)
        return f"summary {len(self.texts)}"


def summarizing(policy, **settings):
    recorder = Recorder()
    return ShortTermMemoryManager(InMemoryStorage(), session_id="trip", policy=policy, summarizer=recorder,
                                  **settings), recorder


class TestSummarize:
    def test_entries_that_leave_the_window_are_summarized(self):
        made, recorder = summarizing(LastNPolicy(2))
        made.add(note("a"))
        made.add(note("b"))
        assert made.summary is None
        made.add(note("c"))
        assert recorder.texts == ["Context:\na"]
        assert made.summary is not None and made.summary.payload == Summary(text="summary 1", newly_covered=["a"])
        assert ids(made.active) == [made.summary.id_, "b", "c"]

    def test_the_next_summary_starts_from_the_previous_one(self):
        made, recorder = summarizing(LastNPolicy(2))
        for name in "abcd":
            made.add(note(name))
        assert recorder.texts[1] == "Summary of the earlier conversation:\nsummary 1\n\nContext:\nb"
        assert made.summary is not None and made.summary.payload == Summary(text="summary 2", newly_covered=["b"])

    def test_entries_leaving_together_are_summarized_together(self):
        made, recorder = summarizing(HeadTailPolicy(head=1, tail=1))
        for name in "abc":
            made.add(note(name))
        made.add(note("d"))
        assert recorder.texts == ["Context:\nb", "Summary of the earlier conversation:\nsummary 1\n\nContext:\nc"]

    def test_a_summarized_entry_stays_in_the_history_but_leaves_the_pool(self):
        seen = []

        def keep(entry):
            seen.append(entry.id_)
            return True

        made, _ = summarizing(PredicateFilterPolicy(keep).then(LastNPolicy(1)))
        made.add(note("a"))
        made.add(note("b"))
        seen.clear()
        made.add(note("c"))
        assert seen == ["b", "c"]  # "a" is summarized: the policy no longer sees it
        assert ids(made.entries) == ["a", "b", "c"]

    def test_pinned_and_unwanted_entries_are_not_summarized(self):
        made, recorder = summarizing(PredicateFilterPolicy(lambda entry: entry.id_ != "x").then(LastNPolicy(1)))
        made.add(note("rule"), pinned=True)
        made.add(note("x"))
        made.add(note("a"))
        made.add(note("b"))
        assert recorder.texts == ["Context:\na"]

    def test_without_a_summarizer_entries_that_leave_are_forgotten(self):
        made = manager(policy=LastNPolicy(1))
        made.add(note("a"))
        made.add(note("b"))
        assert made.summary is None
        assert ids(made.active) == ["b"]

    def test_when_the_summarizer_fails_the_entries_wait_for_the_next_add(self):
        made, recorder = summarizing(LastNPolicy(1))
        made.add(note("a"))
        recorder.fail = True
        with pytest.raises(ValueError, match="the model is down"):
            made.add(note("b"))
        assert made.summary is None
        assert ids(made.entries) == ["a", "b"]  # "b" is stored all the same
        recorder.fail = False
        made.add(note("c"))
        assert recorder.texts == ["Context:\na\n\nContext:\nb"]


class TestCoverage:
    def test_each_summary_version_lists_the_entries_it_added(self):
        made, _ = summarizing(HeadTailPolicy(head=1, tail=1))
        for name in "abcd":
            made.add(note(name))
        versions = [entry.payload for entry in made.storage.entries("trip") if isinstance(entry.payload, Summary)]
        assert [version.newly_covered for version in versions] == [["b"], ["c"]]

    def test_a_manager_on_a_stored_session_keeps_summarized_entries_out(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip", policy=LastNPolicy(1), summarizer=Recorder())
        for name in "abc":
            first.add(note(name))  # "a" and "b" are summarized, in two versions
        seen, recorder = [], Recorder()

        def keep(entry):
            seen.append(entry.id_)
            return True

        second = ShortTermMemoryManager(storage, session_id="trip", policy=PredicateFilterPolicy(keep).then(LastNPolicy(1)),
                                        summarizer=recorder)
        assert seen == ["c"]
        second.add(note("d"))
        assert recorder.texts == ["Summary of the earlier conversation:\nsummary 2\n\nContext:\nc"]


class TestRemove:
    def test_the_entry_leaves_the_storage_and_the_history(self):
        made = manager(session_id="trip")
        made.add(note("a"))
        made.add(note("b"))
        made.remove("a")
        assert ids(made.entries) == ["b"]
        assert ids(made.storage.entries("trip")) == ["b"]

    def test_an_entry_that_is_not_stored_is_refused(self):
        with pytest.raises(ValueError, match="entry 'x' is not stored in session 'trip'"):
            manager(session_id="trip").remove("x")

    def test_summary_and_notes_entries_are_refused(self):
        earlier, kyoto = summary("Earlier."), notes("Kyoto")
        made = opened(earlier, kyoto)
        for entry in (earlier, kyoto):
            with pytest.raises(ValueError, match="come from the manager"):
                made.remove(entry.id_)
        assert (made.summary, made.notes) == (earlier, kyoto)

    def test_a_pin_record_is_refused(self):
        made = manager(session_id="trip")
        made.add(note("rule"), pinned=True)
        (record_id,) = [entry_id for entry_id, kind in made.storage.index("trip") if kind is EntryKind.PIN]
        with pytest.raises(ValueError, match="pin entries come from the manager"):
            made.remove(record_id)
        assert ids(made.pinned) == ["rule"]

    def test_an_entry_the_notes_already_read_leaves_the_notes_as_they_are(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm, notes_every=1)
        made.add(note("q1", "I want to go to Kyoto."))
        made.add(answer("a1"))  # the turn finishes: the notes are updated from q1 and a1
        before = made.notes
        made.remove("q1")
        assert made.notes == before and len(llm.calls) == 1
        with pytest.raises(ValueError, match="nothing arrived"):
            made.update_notes()

    def test_a_pinned_entry_is_no_longer_pinned(self):
        made = manager()
        made.add(note("rule"), pinned=True)
        made.remove("rule")
        assert made.pinned == [] and made.active == []
        assert made.storage.entries("trip") == []  # the pin record is gone too

    def test_the_window_is_chosen_again(self):
        made = manager(policy=LastNPolicy(2))
        for name in "abc":
            made.add(note(name))
        made.remove("c")
        assert ids(made.window) == ["a", "b"]  # "a" was dropped without a summary: it comes back

    def test_a_summarized_entry_does_not_come_back_and_the_summary_stays(self):
        made, _ = summarizing(LastNPolicy(2))
        for name in "abc":
            made.add(note(name))
        before = made.summary
        made.remove("c")
        assert ids(made.window) == ["b"]
        made.remove("a")
        assert made.summary == before
        assert ids(made.entries) == ["b"]

    def test_an_entry_waiting_for_the_notes_is_no_longer_sent(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm)
        made.add(note("a", "I want to go to Paris."))
        made.add(note("b", "I want to go to Kyoto."))
        made.remove("a")
        made.update_notes()
        prompt = llm.calls[0]["user_prompt"].text
        assert "Kyoto" in prompt and "Paris" not in prompt

    def test_a_removed_finished_turn_no_longer_counts(self):
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        made = noting(llm, notes_every=2)
        made.add(answer("a1"))
        made.remove("a1")
        made.add(answer("a2"))
        assert llm.calls == []
        made.add(answer("a3"))
        assert len(llm.calls) == 1

    def test_the_archive_keeps_its_copy_and_the_next_dump_starts_at_the_right_place(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive)
        made.add(note("a"))
        made.add(note("b"))
        made.dump()
        made.remove("a")
        made.add(note("c"))
        assert ids(made.dump()) == ["c"]
        assert archived(archive) == ["a", "b", "c"]


class TestCompact:
    @pytest.fixture(autouse=True)
    def words(self, monkeypatch):
        """Count tokens as words: each ``note`` reads "Context:" and one word, so it is 2 tokens."""
        monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))

    def test_without_a_summarizer_it_is_refused(self):
        with pytest.raises(ValueError, match="no summarizer"):
            manager().compact()

    @pytest.mark.parametrize("max_tokens", [0, -1, 2.5])
    def test_max_tokens_must_be_a_positive_integer(self, max_tokens):
        made, _ = summarizing(None)
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            made.compact(max_tokens)

    def test_an_empty_window_is_left_as_it_is(self):
        made, recorder = summarizing(LastNPolicy(5))
        assert made.compact() is None
        made.add(note("rule"), pinned=True)  # pinned: not in the window
        assert made.compact() is None and recorder.texts == [] and made.summary is None

    def test_with_no_budget_the_whole_window_moves_into_the_summary(self):
        made, recorder = summarizing(None)
        for name in "abc":
            made.add(note(name))
        summary_entry = made.compact()
        assert recorder.texts == ["Context:\na\n\nContext:\nb\n\nContext:\nc"]
        assert made.window == []
        assert made.active == [summary_entry]

    def test_with_a_budget_the_oldest_move_until_the_window_fits(self):
        made, _ = summarizing(None)
        for name in "abcd":
            made.add(note(name))
        summary_entry = made.compact(max_tokens=5)  # c and d: 4 tokens; with b: 6
        assert summary_entry is not None and summary_entry.payload == Summary(text="summary 1", newly_covered=["a", "b"])
        assert ids(made.window) == ["c", "d"]

    def test_a_window_that_fits_is_left_as_it_is(self):
        made, recorder = summarizing(None)
        made.add(note("a"))
        assert made.compact(max_tokens=100) is None
        assert recorder.texts == [] and ids(made.window) == ["a"]

    def test_when_even_the_newest_does_not_fit_the_whole_window_moves(self):
        made, _ = summarizing(None)
        made.add(note("a"))
        made.compact(max_tokens=1)
        assert made.window == []

    def test_the_window_grows_again_but_what_moved_never_comes_back(self):
        made, _ = summarizing(LastNPolicy(3))
        for name in "abc":
            made.add(note(name))
        made.compact(max_tokens=2)
        assert ids(made.window) == ["c"]
        made.add(note("d"))
        assert ids(made.window) == ["c", "d"]

    def test_pinned_entries_stay_and_do_not_count(self):
        made, _ = summarizing(None)
        made.add(note("rule"), pinned=True)
        made.add(note("a"))
        made.add(note("b"))
        made.compact(max_tokens=2)
        assert ids(made.pinned) == ["rule"]
        assert ids(made.window) == ["b"]

    def test_when_the_summarizer_fails_nothing_changes(self):
        made, recorder = summarizing(None)
        made.add(note("a"))
        recorder.fail = True
        with pytest.raises(ValueError, match="the model is down"):
            made.compact()
        assert made.summary is None and ids(made.window) == ["a"]


class TestSetPolicy:
    def test_the_window_is_chosen_again_with_the_new_policy(self):
        made = manager()
        for name in "abc":
            made.add(note(name))
        policy = LastNPolicy(1)
        made.set_policy(policy)
        assert made.policy is policy and ids(made.window) == ["c"]

    def test_what_the_new_policy_drops_is_summarized(self):
        made, recorder = summarizing(None)
        for name in "abc":
            made.add(note(name))
        made.set_policy(LastNPolicy(1))
        assert recorder.texts == ["Context:\na\n\nContext:\nb"]

    def test_entries_dropped_without_a_summary_can_come_back(self):
        made = manager(policy=LastNPolicy(1))
        for name in "ab":
            made.add(note(name))
        made.set_policy(None)
        assert ids(made.window) == ["a", "b"]

    def test_summarized_entries_never_come_back(self):
        made, _ = summarizing(LastNPolicy(1))
        for name in "ab":
            made.add(note(name))
        made.set_policy(None)
        assert ids(made.window) == ["b"]

    def test_only_a_policy_is_accepted(self):
        with pytest.raises(TypeError, match="policy must be a Policy, got str"):
            manager().set_policy("LastNPolicy(1)")  # type: ignore[arg-type]


class TestResume:
    """A manager opened on a stored session gets all its state back."""

    def test_the_pins(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip", policy=LastNPolicy(1))
        first.add(note("rule"), pinned=True)
        first.add(note("a"))
        first.add(note("b"))
        again = ShortTermMemoryManager(storage, session_id="trip", policy=LastNPolicy(1))
        assert ids(again.pinned) == ["rule"] and ids(again.window) == ["b"]

    def test_a_removed_pinned_entry_stays_unpinned(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip")
        first.add(note("rule"), pinned=True)
        first.remove("rule")
        assert ShortTermMemoryManager(storage, session_id="trip").pinned == []

    def test_the_entries_waiting_for_the_notes(self):
        storage = InMemoryStorage()
        first = noting(ScriptedLLM(notes_answer(destination="Paris")), storage=storage)
        first.add(note("a", "Paris, maybe."))
        first.update_notes()
        first.add(note("b", "No: Kyoto."))
        first.set_notes("budget", 3000)  # the app's version: "b" still waits
        llm = ScriptedLLM(notes_answer(destination="Kyoto", budget=3000))
        noting(llm, storage=storage).update_notes()
        prompt = llm.calls[0]["user_prompt"].text
        assert "No: Kyoto." in prompt and "Paris, maybe." not in prompt
        assert '"budget": 3000' in prompt

    def test_the_finished_turns_toward_notes_every(self):
        storage = InMemoryStorage()
        noting(ScriptedLLM(), storage=storage, notes_every=2).add(answer("a1"))
        llm = ScriptedLLM(notes_answer(destination="Kyoto"))
        noting(llm, storage=storage, notes_every=2).add(answer("a2"))
        assert len(llm.calls) == 1

    def test_a_dump_continues_where_the_archive_ends(self):
        storage, archive = InMemoryStorage(), InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip", dump_storage=archive)
        first.add(note("a"))
        first.dump()
        first.add(note("b"))
        assert ids(ShortTermMemoryManager(storage, session_id="trip", dump_storage=archive).dump()) == ["b"]

    def test_the_adds_since_the_dump(self):
        storage, archive = InMemoryStorage(), InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip", dump_storage=archive, dump_every=2)
        for name in "abc":
            first.add(note(name))  # a and b are dumped; c waits
        again = ShortTermMemoryManager(storage, session_id="trip", dump_storage=archive, dump_every=2)
        again.add(note("d"))  # c and d: two adds since the dump
        assert archived(archive) == ["a", "b", "c", "d"]


class TestRewind:
    def test_what_came_after_no_longer_counts(self):
        made = manager(session_id="trip")
        for name in "abc":
            made.add(note(name))
        made.rewind("a")
        assert ids(made.entries) == ["a"] and ids(made.window) == ["a"]
        made.add(note("d"))
        assert ids(made.entries) == ["a", "d"]

    def test_rewinding_to_the_newest_entry_changes_nothing(self):
        made = manager(session_id="trip", policy=LastNPolicy(1))
        for name in "ab":
            made.add(note(name))
        made.rewind("b")
        assert ids(made.entries) == ["a", "b"] and ids(made.window) == ["b"]

    def test_an_entry_that_is_not_stored_or_that_the_manager_made_is_refused(self):
        earlier = summary("Earlier.")
        made = opened(note("a"), earlier)
        with pytest.raises(ValueError, match="entry 'x' is not stored"):
            made.rewind("x")
        with pytest.raises(ValueError, match="come from the manager; rewind"):
            made.rewind(earlier.id_)

    def test_the_summary_goes_back_to_the_version_the_entrys_add_made(self):
        made, _ = summarizing(LastNPolicy(1))
        for name in "abc":
            made.add(note(name))  # adding b summarizes a; adding c summarizes b
        at_b = next(entry for entry in made.storage.entries("trip") if isinstance(entry.payload, Summary))
        made.rewind("b")
        assert made.summary == at_b
        assert ids(made.window) == ["b"]

    def test_pins_after_the_entry_go_and_its_own_pin_stays(self):
        made = manager()
        made.add(note("a"), pinned=True)
        made.add(note("b"), pinned=True)
        made.rewind("a")
        assert ids(made.pinned) == ["a"] and made.window == []

    def test_the_notes_go_back_and_what_comes_next_waits_for_them(self):
        llm = ScriptedLLM(notes_answer(destination="Paris"), notes_answer(destination="Kyoto"),
                          notes_answer(destination="Rome"))
        made = noting(llm)
        made.add(note("a", "Paris."))
        paris = made.update_notes()
        made.add(note("b", "Kyoto."))
        made.update_notes()
        made.rewind("a")
        assert made.notes == paris
        with pytest.raises(ValueError, match="nothing arrived"):
            made.update_notes()
        made.add(note("c", "Rome."))
        made.update_notes()
        prompt = llm.calls[2]["user_prompt"].text
        assert "Rome." in prompt and "Kyoto." not in prompt

    def test_the_archive_keeps_its_copies_and_the_dump_goes_on(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive)
        for name in "abc":
            made.add(note(name))
        made.dump()
        made.rewind("a")
        made.add(note("d"))
        assert ids(made.dump()) == ["d"]
        assert archived(archive) == ["a", "b", "c", "d"]


class TestUnpin:
    def test_the_entry_goes_back_to_the_window_in_its_place(self):
        made = manager(session_id="trip")
        made.add(note("rule"), pinned=True)
        made.add(note("a"))
        made.unpin("rule")
        assert made.pinned == []
        assert ids(made.window) == ["rule", "a"]
        assert ids(made.storage.entries("trip")) == ["rule", "a"]  # the pin record is gone

    def test_an_old_entry_leaves_at_once_and_is_summarized(self):
        made, recorder = summarizing(LastNPolicy(1))
        made.add(note("rule"), pinned=True)
        made.add(note("a"))
        made.add(note("b"))  # summarizes "a"
        made.unpin("rule")
        assert recorder.texts[-1] == "Summary of the earlier conversation:\nsummary 1\n\nContext:\nrule"
        assert ids(made.window) == ["b"]

    def test_an_entry_that_is_not_pinned_is_refused(self):
        made = manager()
        made.add(note("a"))
        with pytest.raises(ValueError, match="entry 'a' is not pinned"):
            made.unpin("a")

    def test_a_manager_opened_later_sees_it_unpinned(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="trip")
        first.add(note("rule"), pinned=True)
        first.unpin("rule")
        assert ShortTermMemoryManager(storage, session_id="trip").pinned == []

    def test_the_next_dump_starts_at_the_right_place(self):
        archive = InMemoryStorage()
        made = manager(session_id="trip", dump_storage=archive)
        made.add(note("rule"), pinned=True)
        made.add(note("a"))
        made.dump()
        made.unpin("rule")
        made.add(note("b"))
        assert ids(made.dump()) == ["b"]


def system(name, text=None):
    """A system prompt entry with ID ``name``."""
    return Entry.of(SystemPrompt(text=text or name)).model_copy(update={"id_": name})


def system_id(made):
    return made.system_prompt.id_ if made.system_prompt is not None else None


class TestSystemPrompt:
    def test_without_one_there_is_none(self):
        made = manager()
        made.add(note("a"))
        assert system_id(made) is None
        assert ids(made.active) == ["a"]

    def test_the_latest_leads_what_goes_to_the_model(self):
        made = opened(note("a"), summary("Earlier: a."), system("s"), note("b"))
        assert ids(made.active)[0] == "s"
        assert made.active[1].payload == Summary(text="Earlier: a.")

    def test_a_new_one_replaces_the_previous_which_stays_in_the_history(self):
        made = manager()
        for entry in (system("s1"), note("a"), system("s2"), note("b")):
            made.add(entry)
        assert system_id(made) == "s2"
        assert ids(made.active) == ["s2", "a", "b"]
        assert ids(made.entries) == ["s1", "a", "s2", "b"]

    def test_the_policy_never_sees_them_so_they_are_never_dropped_or_summarized(self):
        made, recorder = summarizing(LastNPolicy(2))
        made.add(system("s", "You plan trips."))
        for name in "abcd":
            made.add(note(name))
        assert ids(made.window) == ["c", "d"]
        assert ids(made.active)[0] == "s"
        assert recorder.texts and all("You plan trips." not in text for text in recorder.texts)

    def test_a_system_prompt_cannot_be_pinned(self):
        made = manager(session_id="trip")
        with pytest.raises(ValueError, match="system prompts are kept by rule"):
            made.add(system("s"), pinned=True)
        assert made.storage.entries("trip") == []

    def test_removing_the_latest_brings_back_the_one_before(self):
        made = manager()
        made.add(system("s1"))
        made.add(system("s2"))
        made.remove("s2")
        assert system_id(made) == "s1"
        made.remove("s1")
        assert system_id(made) is None

    def test_removing_an_older_one_changes_nothing(self):
        made = manager()
        made.add(system("s1"))
        made.add(system("s2"))
        made.remove("s1")
        assert system_id(made) == "s2"

    def test_a_session_opened_again_knows_the_latest(self):
        assert system_id(opened(system("s1"), note("a"), system("s2"), note("b"))) == "s2"

    def test_a_rewind_to_before_the_latest_brings_back_the_one_before(self):
        made = manager()
        for entry in (system("s1"), note("a"), system("s2"), note("b")):
            made.add(entry)
        made.rewind("a")
        assert system_id(made) == "s1"
        assert ids(made.active) == ["s1", "a"]



class TestSummaryKeeper:
    def test_without_a_summarizer_it_refuses_to_summarize(self):
        keeper = SummaryKeeper(InMemoryStorage(), "trip", None)
        with pytest.raises(ValueError, match="this manager has no summarizer"):
            keeper.summarize([note("a")])
        with pytest.raises(ValueError, match="this manager has no summarizer"):
            asyncio.run(keeper.summarize_async([note("a")]))
        assert keeper.current is None
