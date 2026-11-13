import asyncio
import json
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import (
    ChainPolicy,
    NoteTaker,
    WorkingNotesEntry,
    RebuildCompaction,
    RollingCompaction,
    SummaryEntry,
    ContextEntry,
    FileStorage,
    FullPolicy,
    InMemoryStorage,
    LastNPolicy,
    ShortTermMemoryManager,
    SlidingWindowPolicy,
    SQLiteStorage,
    ToolResultEntry,
    TypeFilterPolicy,
)
from hiyal.models.schemas import LLMResponse, ResponseMessage, ToolCallContent
from hiyal.prompts import SystemPrompt, UserPrompt
from hiyal.tools import ToolCall
from hiyal.utility import Tokens

from tests.memory.short_term.compaction.test_compaction import FakeSummarizer
from tests.memory.short_term.note_takers.test_note_taker import FakeLLM, TripNotes, notes_json



def _run_async(name):
    """A sync stand-in for ``name`` that runs its ``_async`` twin."""
    def method(self, *args, **kwargs):
        return asyncio.run(getattr(self, f"{name}_async")(*args, **kwargs))
    return method


def _read_async(name):
    """A sync stand-in for the property ``name`` that awaits ``name_async()``."""
    return property(lambda self: asyncio.run(getattr(self, f"{name}_async")()))


@pytest.fixture(autouse=True, params=["sync", "async"])
def mode(request, monkeypatch):
    """Run every test twice: through the sync API, then through the async one.

    In the async run, construction is ``open_async``, every operation its
    ``_async`` twin and every read its ``_async`` method. A sync call left
    inside the async path would then call ``asyncio.run`` inside the running
    loop, and fail the test.
    """
    if request.param == "async":
        manager = ShortTermMemoryManager

        def opened(self, *args, **kwargs):
            self.__dict__.update(asyncio.run(manager.open_async(*args, **kwargs)).__dict__)

        monkeypatch.setattr(manager, "__init__", opened)
        for name in ("add", "compact", "refresh_notes", "dump"):
            monkeypatch.setattr(manager, name, _run_async(name))
        for name in ("active", "window", "summary", "notes", "pinned", "entries"):
            monkeypatch.setattr(manager, name, _read_async(name))
        monkeypatch.setattr(manager, "policy", property(
            lambda self: self._policy, lambda self, policy: asyncio.run(self.set_policy_async(policy))))
    return request.param


def notes(*texts):
    return [ContextEntry(text=text) for text in texts]


def texts(entries):
    return [entry.text for entry in entries]


def covered(manager):
    """The IDs the current summary covers, read against the session's history."""
    if manager.summary is None:
        return []
    originals = [e.id_ for e in manager.entries if not isinstance(e, (SummaryEntry, WorkingNotesEntry))]
    return manager.summary.covered(originals)


def managed(entries, **settings):
    manager = ShortTermMemoryManager(InMemoryStorage(), **settings)
    for entry in entries:
        manager.add(entry)
    return manager


class TestSession:
    def test_a_new_session_gets_a_generated_id(self):
        first, second = ShortTermMemoryManager(InMemoryStorage()), ShortTermMemoryManager(InMemoryStorage())
        assert first.session_id and first.session_id != second.session_id

    def test_a_given_id_continues_that_session(self):
        storage = InMemoryStorage()
        ShortTermMemoryManager(storage, session_id="s1").add(ContextEntry(text="a"))
        assert texts(ShortTermMemoryManager(storage, session_id="s1").entries) == ["a"]
        assert ShortTermMemoryManager(storage, session_id="s2").entries == []

    @pytest.mark.parametrize("session_id, error", [("", ValueError), (7, TypeError)])
    def test_a_given_id_must_be_a_non_empty_string(self, session_id, error):
        with pytest.raises(error):
            ShortTermMemoryManager(InMemoryStorage(), session_id=session_id)


class TestPrimaryStorage:
    def test_entries_are_added_in_order(self):
        manager = ShortTermMemoryManager(InMemoryStorage())
        for entry in notes("a", "b", "c"):
            manager.add(entry)
        assert texts(manager.entries) == ["a", "b", "c"]

    def test_a_duplicate_id_is_rejected_also_after_a_restart(self):
        storage = InMemoryStorage()
        entry = ContextEntry(text="a")
        ShortTermMemoryManager(storage, session_id="s1").add(entry)
        with pytest.raises(ValueError, match="already in the session"):
            ShortTermMemoryManager(storage, session_id="s1").add(entry)


class TestDump:
    def test_dumps_the_entries_and_keeps_the_primary(self):
        dump = InMemoryStorage()
        manager = ShortTermMemoryManager(InMemoryStorage(), dump_storage=dump)
        for entry in notes("a", "b"):
            manager.add(entry)
        assert texts(manager.dump()) == ["a", "b"]
        assert texts(dump.entries(manager.session_id)) == ["a", "b"]
        assert texts(manager.entries) == ["a", "b"]

    def test_dumping_again_appends_only_the_new_entries(self):
        dump = InMemoryStorage()
        manager = ShortTermMemoryManager(InMemoryStorage(), dump_storage=dump)
        manager.add(ContextEntry(text="a"))
        manager.dump()
        manager.add(ContextEntry(text="b"))
        assert texts(manager.dump()) == ["b"]
        assert manager.dump() == []
        assert texts(dump.entries(manager.session_id)) == ["a", "b"]

    def test_a_continued_session_does_not_dump_twice(self, tmp_path):
        primary = InMemoryStorage()
        dump_path = tmp_path / "dump.jsonl"
        first = ShortTermMemoryManager(primary, dump_storage=FileStorage(dump_path), session_id="s1")
        first.add(ContextEntry(text="a"))
        first.dump()
        again = ShortTermMemoryManager(primary, dump_storage=FileStorage(dump_path, exist_ok=True), session_id="s1")
        again.add(ContextEntry(text="b"))
        assert texts(again.dump()) == ["b"]
        assert texts(FileStorage(dump_path, exist_ok=True).entries("s1")) == ["a", "b"]

    def test_sessions_sharing_a_dump_stay_apart(self):
        primary, dump = InMemoryStorage(), InMemoryStorage()
        one = ShortTermMemoryManager(primary, dump_storage=dump, session_id="s1")
        two = ShortTermMemoryManager(primary, dump_storage=dump, session_id="s2")
        one.add(ContextEntry(text="a"))
        two.add(ContextEntry(text="b"))
        one.dump()
        two.dump()
        assert texts(dump.entries("s1")) == ["a"] and texts(dump.entries("s2")) == ["b"]

    def test_needs_a_dump_storage(self):
        with pytest.raises(ValueError, match="no dump_storage"):
            ShortTermMemoryManager(InMemoryStorage()).dump()


class TestActiveWindow:
    def test_a_new_session_starts_empty_and_every_added_entry_joins(self):
        manager = ShortTermMemoryManager(InMemoryStorage())
        assert manager.active == []
        for entry in notes("a", "b"):
            manager.add(entry)
        assert texts(manager.active) == ["a", "b"]

    def test_holds_ids_and_reads_the_entries_from_the_primary(self):
        storage = InMemoryStorage()
        manager = ShortTermMemoryManager(storage, session_id="s1")
        manager.add(ContextEntry(text="a"))
        storage.add("s1", ContextEntry(text="written behind the manager's back"))
        assert texts(manager.active) == ["a"]  # the window is the manager's IDs, not the whole primary

    def test_a_continued_session_starts_with_its_stored_entries(self):
        storage = InMemoryStorage()
        ShortTermMemoryManager(storage, session_id="s1").add(ContextEntry(text="a"))
        again = ShortTermMemoryManager(storage, session_id="s1")
        again.add(ContextEntry(text="b"))
        assert texts(again.active) == ["a", "b"]

    def test_a_returned_list_cannot_change_the_window(self):
        manager = managed(notes("a"))
        manager.active.clear()
        assert texts(manager.active) == ["a"]

    def test_a_rejected_duplicate_does_not_join(self):
        entry = ContextEntry(text="a")
        manager = managed([entry])
        with pytest.raises(ValueError):
            manager.add(entry)
        assert texts(manager.active) == ["a"]

    def test_dumping_leaves_the_window_as_it_is(self):
        manager = managed(notes("a", "b"), dump_storage=InMemoryStorage())
        manager.dump()
        assert texts(manager.active) == ["a", "b"]


def ids(entries):
    return [entry.id_ for entry in entries]


def answer(name, calls=()):
    messages = [ResponseMessage(id=call, role="assistant", content=ToolCallContent(
        data=ToolCall(name="search", arguments={}, status="complete", raw=None))) for call in calls]
    return LLMResponse(text=name, messages=messages, provider="openai", id_=f"resp-{name}", model="m",
                       created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")


def result(call):
    return ToolResultEntry(message_id=call, name="search", arguments={}, output="found")



def conversation():
    """System prompts, questions, answers, tool calls with results, context: every pair to repair."""
    return [
        SystemPrompt(text="s1"), UserPrompt(text="q1"), answer("r1"), ContextEntry(text="c1"),
        UserPrompt(text="q2"), answer("r2", calls=["m1", "m2"]), result("m1"), result("m2"), answer("r2b"),
        ContextEntry(text="c2"), SystemPrompt(text="s2"), UserPrompt(text="q3"), answer("r3", calls=["m3"]),
        result("m3"), UserPrompt(text="q4"), answer("r4"), ContextEntry(text="c3"), UserPrompt(text="q5"),
        answer("r5", calls=["m4"]), result("m4"), answer("r5b"), UserPrompt(text="q6"), answer("r6"),
    ]


POLICIES = {
    "full": lambda: FullPolicy(),
    **{f"last-{n}": (lambda n=n: LastNPolicy(n)) for n in (1, 2, 3, 4, 5, 7)},
    "sliding-4-2": lambda: SlidingWindowPolicy(4, 2),
    "sliding-6-3": lambda: SlidingWindowPolicy(6, 3),
    "sliding-5-1": lambda: SlidingWindowPolicy(5, 1),
    "no-context": lambda: TypeFilterPolicy(exclude=[ContextEntry]),
    "no-results": lambda: TypeFilterPolicy(exclude=[ToolResultEntry]),
    "no-questions": lambda: TypeFilterPolicy(exclude=[UserPrompt]),
    "filter-then-last": lambda: ChainPolicy([TypeFilterPolicy(exclude=[ContextEntry]), LastNPolicy(3)]),
    "filter-then-sliding": lambda: ChainPolicy([TypeFilterPolicy(exclude=[ContextEntry]), SlidingWindowPolicy(4, 2)]),
    # Written window first: the chain runs the filter first anyway, so no drift.
    "sliding-then-filter": lambda: ChainPolicy([SlidingWindowPolicy(6, 3), TypeFilterPolicy(exclude=[ContextEntry])]),
    "last-then-filter": lambda: ChainPolicy([LastNPolicy(3), TypeFilterPolicy(exclude=[ContextEntry])]),
}


class TestPolicyWindow:
    @pytest.mark.parametrize("name", POLICIES)
    def test_after_every_add_the_same_as_choosing_from_the_whole_history(self, name):
        # Without system prompts nothing is pinned, so the window is the policy's choice alone.
        manager, policy, added = ShortTermMemoryManager(InMemoryStorage(), policy=POLICIES[name]()), POLICIES[name](), []
        for entry in [e for e in conversation() if not isinstance(e, SystemPrompt)]:
            manager.add(entry)
            added.append(entry)
            # The reference: the policy's choice from everything added so far.
            assert ids(manager.active) == ids(policy.select(added)), f"after adding {entry.to_text()!r}"

    def test_the_default_policy_keeps_everything(self):
        manager = managed(notes("a", "b", "c"))
        assert isinstance(manager.policy, FullPolicy) and texts(manager.active) == ["a", "b", "c"]

    def test_pairs_are_not_repaired_the_window_is_the_policys_choice(self):
        q, r, done = UserPrompt(text="q"), answer("r", calls=["c1"]), result("c1")
        manager = managed([q, r, done, *notes("a")], policy=LastNPolicy(2))
        assert ids(manager.active) == [done.id_, manager.entries[-1].id_]  # the result, without its call

    def test_replacing_the_policy_chooses_again_from_the_whole_history(self):
        manager = managed(notes("a", "b", "c", "d"), policy=LastNPolicy(1))
        manager.policy = LastNPolicy(3)  # entries that left the window come back
        assert texts(manager.active) == ["b", "c", "d"]

    def test_a_continued_session_chooses_its_window_from_the_stored_entries(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "history.db")
        first = ShortTermMemoryManager(storage, session_id="s1")
        for entry in notes("a", "b", "c"):
            first.add(entry)
        again = ShortTermMemoryManager(storage, policy=LastNPolicy(2), session_id="s1")
        assert texts(again.active) == ["b", "c"]
        again.add(ContextEntry(text="d"))
        assert texts(again.active) == ["c", "d"]
        storage.close()


class TestPinning:
    def test_a_pinned_entry_stays_in_its_place_and_does_not_count_toward_the_window(self):
        manager = managed(notes("a", "b", "c"), policy=LastNPolicy(2))
        manager.pin(manager.entries[1].id_)  # b
        for entry in notes("d", "e"):
            manager.add(entry)
        assert texts(manager.active) == ["b", "d", "e"]  # two chosen by the policy, plus b, still first
        assert texts(manager.pinned) == ["b"]

    def test_unpinning_removes_nothing_until_the_next_add(self):
        manager = managed(notes("a", "b", "c"), policy=LastNPolicy(2))
        b = manager.entries[1].id_
        manager.pin(b)
        manager.add(ContextEntry(text="d"))
        manager.unpin(b)
        assert texts(manager.active) == ["b", "c", "d"]
        manager.add(ContextEntry(text="e"))
        assert texts(manager.active) == ["d", "e"]

    def test_only_an_active_entry_can_be_pinned(self):
        manager = managed(notes("a", "b"), policy=LastNPolicy(1))
        with pytest.raises(ValueError, match="not in the window"):
            manager.pin(manager.entries[0].id_)

    def test_pin_and_unpin_take_an_id(self):
        manager = managed(notes("a"))
        with pytest.raises(TypeError, match="entry_id must be a string"):
            manager.pin(manager.entries[0])

    def test_the_system_prompt_is_pinned(self):
        manager = managed([SystemPrompt(text="sys"), *notes("a", "b", "c")], policy=LastNPolicy(2))
        assert texts(manager.active) == ["sys", "b", "c"] and texts(manager.pinned) == ["sys"]

    def test_a_new_system_prompt_takes_the_pin_and_the_old_one_leaves_when_the_policy_drops_it(self):
        manager = managed([SystemPrompt(text="s1"), *notes("a", "b", "c")], policy=LastNPolicy(2))
        manager.add(SystemPrompt(text="s2"))
        assert texts(manager.active) == ["s1", "b", "c", "s2"]  # s1 unpinned, but still active
        assert texts(manager.pinned) == ["s2"]
        manager.add(ContextEntry(text="d"))
        assert texts(manager.active) == ["c", "s2", "d"]  # the policy dropped s1; s2 keeps its place
        manager.add(ContextEntry(text="e"))
        assert texts(manager.active) == ["s2", "d", "e"]  # everything older left: s2 is first

    def test_an_old_system_prompt_that_left_never_comes_back(self):
        storage = InMemoryStorage()
        manager = ShortTermMemoryManager(storage, compaction=RollingCompaction(FakeSummarizer()), session_id="s1")
        for entry in [SystemPrompt(text="s1"), *notes("a"), SystemPrompt(text="s2"), *notes("b")]:
            manager.add(entry)
        manager.compact()  # a and b go into the summary; s1 just leaves (not summarized)
        assert texts(manager.window) == ["s2"]
        manager.policy = FullPolicy()  # the window is chosen again from the whole history
        assert texts(manager.window) == ["s2"]
        assert texts(ShortTermMemoryManager(storage, session_id="s1").window) == ["s2"]  # and on resume

    def test_choosing_again_considers_only_the_latest_system_prompt(self):
        storage = InMemoryStorage()
        manager = ShortTermMemoryManager(storage, session_id="s1")  # FullPolicy: s1 stays in the window
        for entry in [SystemPrompt(text="s1"), *notes("a"), SystemPrompt(text="s2"), *notes("b")]:
            manager.add(entry)
        assert texts(manager.active) == ["s1", "a", "s2", "b"]
        assert texts(ShortTermMemoryManager(storage, session_id="s1").active) == ["a", "s2", "b"]  # a resume leaves s1 out
        manager.policy = FullPolicy()
        assert texts(manager.active) == ["a", "s2", "b"]  # so does a new policy

    def test_a_sliding_window_still_jumps_by_step_around_a_pinned_system_prompt(self):
        policy = SlidingWindowPolicy(4, 2)
        manager = managed([SystemPrompt(text="sys")], policy=SlidingWindowPolicy(4, 2))
        added = []
        for entry in notes(*"abcdefghij"):
            manager.add(entry)
            added.append(entry)
            assert texts(manager.active) == ["sys", *texts(policy.select(added))]

    def test_system_prompts_are_pinned_by_rule_only(self):
        manager = managed([SystemPrompt(text="s1"), *notes("a"), SystemPrompt(text="s2")])
        old, new = manager.entries[0].id_, manager.entries[2].id_
        with pytest.raises(ValueError, match="pinned by rule"):
            manager.pin(old)
        with pytest.raises(ValueError, match="stay pinned"):
            manager.unpin(new)

    def test_replacing_the_policy_keeps_the_pins(self):
        manager = managed(notes("a", "b", "c"))
        manager.pin(manager.entries[0].id_)
        manager.policy = LastNPolicy(1)
        assert texts(manager.active) == ["a", "c"]

    def test_a_continued_session_pins_its_latest_system_prompt_again(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, session_id="s1")
        for entry in [SystemPrompt(text="s1"), *notes("a"), SystemPrompt(text="s2"), *notes("b", "c")]:
            first.add(entry)
        again = ShortTermMemoryManager(storage, policy=LastNPolicy(1), session_id="s1")
        assert texts(again.active) == ["s2", "c"] and texts(again.pinned) == ["s2"]


def compacting(entries, summarizer=None, kind=RollingCompaction, **settings):
    """A manager with a compaction over a FakeSummarizer ('summary #N of <texts>'), with ``entries`` added."""
    return managed(entries, compaction=kind(summarizer or FakeSummarizer()), **settings)


class FlakySummarizer(FakeSummarizer):
    """Fails while ``failing`` is true, like a cut-off response or an API error."""

    failing = True

    def summarize(self, entries, previous=None):
        if self.failing:
            raise ValueError("the summary response is 'incomplete', not 'completed'")
        return super().summarize(entries, previous)


class TextSummarizer(FakeSummarizer):
    """Like FakeSummarizer, but labels any entry, a tool result too, by its ``to_text()``."""

    def summarize(self, entries, previous=None):
        return super().summarize([ContextEntry(text=entry.to_text()) for entry in entries], previous)


class TestCompaction:
    def test_what_leaves_the_window_is_summarized_in_the_same_add(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b", "c"), summarizer, policy=LastNPolicy(2))
        assert summarizer.calls == [(["a"], None)]
        assert texts(manager.active) == ["summary #1 of a", "b", "c"]
        manager.add(ContextEntry(text="d"))
        assert summarizer.calls[-1] == (["b"], "summary #1 of a")  # the current summary and only the new drop
        assert texts(manager.active) == ["summary #2 of b", "c", "d"]
        assert len(manager.entries) == 6  # nothing removed from the primary; each summary is added

    def test_the_summary_covers_everything_that_left(self):
        manager = compacting(notes("a", "b", "c", "d"), policy=LastNPolicy(2))
        by_text = {entry.text: entry.id_ for entry in manager.entries if isinstance(entry, ContextEntry)}
        assert covered(manager) == [by_text["a"], by_text["b"]]

    def test_a_sliding_window_gives_one_summary_per_jump(self):
        summarizer = FakeSummarizer()
        compacting(notes(*"abcdefgh"), summarizer, policy=SlidingWindowPolicy(4, 2))
        assert summarizer.calls == [(["a", "b"], None), (["c", "d"], "summary #1 of a b")]

    @pytest.mark.parametrize("name", POLICIES)
    def test_every_entry_is_in_the_window_the_summary_or_excluded_by_a_filter(self, name):
        policy = POLICIES[name]()
        manager = ShortTermMemoryManager(InMemoryStorage(), policy=policy, compaction=RollingCompaction(TextSummarizer()))
        for entry in [e for e in conversation() if not isinstance(e, SystemPrompt)]:
            manager.add(entry)
            added = [e for e in manager.entries if not isinstance(e, SummaryEntry)]
            excluded = {e.id_ for e in added} - {e.id_ for e in filters_keep(policy, added)}
            summarized = set(covered(manager))
            window = {e.id_ for e in manager.active if not isinstance(e, SummaryEntry)}
            assert window | summarized | excluded == {e.id_ for e in added}
            assert not window & summarized and not window & excluded and not summarized & excluded

    def test_the_summary_lists_only_what_it_skips_however_long_the_history(self):
        manager = compacting([SystemPrompt(text="S"), *notes(*map(str, range(100)))], policy=SlidingWindowPolicy(10, 5))
        system = manager.entries[0]
        assert manager.summary.covers_except == [system.id_]  # the system prompt, still in the window
        assert manager.summary.covers_count == len(covered(manager)) > 80

    def test_a_pinned_entry_is_an_exception_until_it_leaves(self):
        manager = compacting(notes("a", "b"), policy=LastNPolicy(1))
        b = manager.entries[1]
        manager.pin(b.id_)
        for entry in notes("c", "d"):
            manager.add(entry)
        by_id = {entry.id_: entry for entry in manager.entries}
        assert manager.summary.covers_except == [b.id_]
        assert texts(by_id[i] for i in covered(manager)) == ["a", "c"]
        manager.unpin(b.id_)
        manager.add(ContextEntry(text="e"))
        by_id = {entry.id_: entry for entry in manager.entries}
        assert manager.summary.covers_except == []
        assert texts(by_id[i] for i in covered(manager)) == ["a", "b", "c", "d"]

    def test_without_a_compaction_what_leaves_is_forgotten(self):
        manager = managed(notes("a", "b", "c"), policy=LastNPolicy(2))
        assert manager.summary is None and texts(manager.active) == ["b", "c"]

    def test_the_summary_is_first_even_before_the_pinned_system_prompt(self):
        manager = compacting([SystemPrompt(text="S"), *notes("a", "b")], policy=LastNPolicy(1))
        assert texts(manager.active) == ["summary #1 of a", "S", "b"]
        assert texts(manager.pinned) == ["summary #1 of a", "S"]

    def test_pinned_entries_are_never_summarized(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b"), summarizer, policy=LastNPolicy(2))
        manager.pin(manager.entries[0].id_)  # a
        for entry in notes("c", "d"):
            manager.add(entry)
        assert [call[0] for call in summarizer.calls] == [["b"]]
        assert texts(manager.active) == ["summary #1 of b", "a", "c", "d"]

    def test_an_old_system_prompt_is_not_summarized(self):
        summarizer = FakeSummarizer()
        compacting([SystemPrompt(text="S1"), *notes("a"), SystemPrompt(text="S2"), *notes("b", "c")],
                   summarizer, policy=LastNPolicy(1))
        assert [call[0] for call in summarizer.calls] == [["a"], ["b"]]

    def test_entries_a_filter_excludes_are_forgotten_not_summarized(self):
        summarizer = FakeSummarizer()
        manager = compacting([*notes("a"), UserPrompt(text="q")], summarizer,
                             policy=TypeFilterPolicy(exclude=[ContextEntry]))
        assert summarizer.calls == [] and manager.summary is None and texts(manager.active) == ["q"]

    def test_in_a_chain_the_window_drops_are_summarized_and_the_filter_drops_are_not(self):
        summarizer = FakeSummarizer()
        entries = [UserPrompt(text="q1"), *notes("c1"), UserPrompt(text="q2"), UserPrompt(text="q3")]
        compacting(entries, summarizer, policy=ChainPolicy([LastNPolicy(2), TypeFilterPolicy(exclude=[ContextEntry])]))
        assert summarizer.calls == [(["q1"], None)]  # c1 was excluded, q1 left the window

    def test_a_rebuild_resends_the_originals(self):
        summarizer = FakeSummarizer()
        compacting(notes("a", "b", "c"), summarizer, kind=RebuildCompaction, policy=LastNPolicy(1))
        assert summarizer.calls[-1] == (["a", "b"], None)

    def test_if_summarizing_fails_nothing_leaves_and_the_next_add_retries(self):
        summarizer = FlakySummarizer()
        manager = compacting(notes("a", "b"), summarizer, policy=LastNPolicy(2))
        with pytest.raises(ValueError, match="incomplete"):
            manager.add(ContextEntry(text="c"))
        assert texts(manager.active) == ["a", "b", "c"] and manager.summary is None
        summarizer.failing = False
        manager.add(ContextEntry(text="d"))
        assert summarizer.calls == [(["a", "b"], None)]
        assert texts(manager.active) == ["summary #1 of a b", "c", "d"]

    def test_a_smaller_policy_summarizes_what_it_leaves_out_and_a_larger_one_never_brings_back_the_summarized(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b", "c", "d"), summarizer)  # FullPolicy: nothing summarized yet
        manager.policy = LastNPolicy(1)
        assert summarizer.calls == [(["a", "b", "c"], None)] and texts(manager.active) == ["summary #1 of a b c", "d"]
        manager.policy = FullPolicy()
        assert texts(manager.active) == ["summary #1 of a b c", "d"]

    def test_a_continued_session_restores_the_summary_and_the_window(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "history.db")
        first = ShortTermMemoryManager(storage, policy=LastNPolicy(2), compaction=RollingCompaction(FakeSummarizer()),
                              session_id="s1")
        for entry in [SystemPrompt(text="S"), *notes("a", "b", "c")]:
            first.add(entry)
        again = ShortTermMemoryManager(storage, policy=LastNPolicy(2), session_id="s1")
        assert texts(again.active) == ["summary #1 of a", "S", "b", "c"]
        storage.close()

    def test_a_session_resumed_with_a_compaction_summarizes_what_was_never_summarized(self):
        storage = InMemoryStorage()
        for entry in notes("a", "b", "c"):
            storage.add("s1", entry)
        summarizer = FakeSummarizer()
        manager = ShortTermMemoryManager(storage, policy=LastNPolicy(1), compaction=RollingCompaction(summarizer), session_id="s1")
        assert summarizer.calls == [(["a", "b"], None)] and texts(manager.active) == ["summary #1 of a b", "c"]

    def test_refused(self):
        manager = compacting(notes("a", "b"), policy=LastNPolicy(1))
        with pytest.raises(ValueError, match="summaries come from compaction"):
            manager.add(SummaryEntry(text="s", covers_through="x", covers_count=1))
        with pytest.raises(ValueError, match="pinned by rule"):
            manager.pin(manager.summary.id_)
        with pytest.raises(ValueError, match="stay pinned"):
            manager.unpin(manager.summary.id_)


def size(*entries):
    """The tokens of ``entries`` as ``count_words`` counts them: the words of each one's ``to_text()``."""
    return sum(len(entry.to_text().split()) for entry in entries)


class TestCompactOnDemand:
    @pytest.fixture(autouse=True)
    def count_words(self, monkeypatch):
        """Count words instead of tokens, so no test loads a tiktoken encoding (that needs network)."""
        monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))

    def test_without_a_budget_the_whole_unpinned_window_goes_to_the_summary(self):
        summarizer = FakeSummarizer()
        manager = compacting([SystemPrompt(text="S"), *notes("a", "b", "c")], summarizer)
        update = manager.compact()
        assert summarizer.calls == [(["a", "b", "c"], None)] and update.summary.id_ == manager.summary.id_
        assert texts(manager.window) == ["S"] and texts(manager.active) == ["summary #1 of a b c", "S"]

    def test_with_a_budget_the_newest_entries_that_fit_stay(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b", "c", "d"), summarizer)
        a, b, c, d = manager.entries
        manager.compact(max_tokens=size(c, d))
        assert summarizer.calls == [(["a", "b"], None)] and covered(manager) == [a.id_, b.id_]
        assert texts(manager.window) == ["c", "d"]

    def test_it_stops_at_the_first_entry_that_does_not_fit(self):
        summarizer = FakeSummarizer()
        long = " ".join(["b"] * 20)
        manager = compacting(notes("a", long, "c"), summarizer)
        a, _, c = manager.entries
        manager.compact(max_tokens=size(a, c))  # a and c would fit together, but the long one is between them
        assert summarizer.calls == [(["a", long], None)] and texts(manager.window) == ["c"]

    def test_pinned_entries_count_toward_the_budget_and_stay(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b", "c", "d"), summarizer)
        a, _, _, d = manager.entries
        manager.pin(a.id_)
        manager.compact(max_tokens=size(a, d))
        assert summarizer.calls == [(["b", "c"], None)] and texts(manager.window) == ["a", "d"]

    def test_pinned_entries_alone_over_the_budget_change_nothing(self):
        summarizer = FakeSummarizer()
        manager = compacting([SystemPrompt(text="Be brief and kind."), *notes("a")], summarizer)
        with pytest.raises(ValueError, match="pinned entries alone take 5 tokens, more than max_tokens=1"):
            manager.compact(max_tokens=1)
        assert summarizer.calls == [] and texts(manager.window) == ["Be brief and kind.", "a"]

    def test_when_the_window_already_fits_nothing_is_summarized(self):
        summarizer = FakeSummarizer()
        manager = compacting(notes("a", "b"), summarizer)
        assert manager.compact(max_tokens=size(*manager.entries)) is None
        assert summarizer.calls == [] and texts(manager.window) == ["a", "b"]

    def test_an_old_system_prompt_leaves_without_being_summarized(self):
        summarizer = FakeSummarizer()
        manager = compacting([SystemPrompt(text="S1"), *notes("a"), SystemPrompt(text="S2"), *notes("b")], summarizer)
        manager.compact()
        assert summarizer.calls == [(["a", "b"], None)] and texts(manager.window) == ["S2"]

    def test_the_window_grows_again_and_the_summarized_never_come_back(self):
        manager = compacting(notes("a", "b", "c"), policy=LastNPolicy(3))
        manager.compact(max_tokens=size(manager.entries[2]))
        manager.add(ContextEntry(text="d"))
        assert texts(manager.active) == ["summary #1 of a b", "c", "d"]
        manager.policy = FullPolicy()
        assert texts(manager.active) == ["summary #1 of a b", "c", "d"]

    def test_a_continued_session_keeps_what_was_compacted_out(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, compaction=RollingCompaction(FakeSummarizer()), session_id="s1")
        for entry in notes("a", "b", "c"):
            first.add(entry)
        first.compact(max_tokens=size(first.entries[2]))
        assert texts(ShortTermMemoryManager(storage, session_id="s1").active) == ["summary #1 of a b", "c"]

    def test_if_summarizing_fails_the_window_stays_as_it_was(self):
        manager = compacting(notes("a", "b"), FlakySummarizer())
        with pytest.raises(ValueError, match="incomplete"):
            manager.compact()
        assert texts(manager.window) == ["a", "b"] and manager.summary is None

    @pytest.mark.parametrize("max_tokens", [-1, 1.5, True, "10"])
    def test_max_tokens_must_be_a_non_negative_integer(self, max_tokens):
        with pytest.raises(ValueError, match="max_tokens must be a non-negative integer"):
            compacting(notes("a")).compact(max_tokens=max_tokens)

    def test_refused(self):
        with pytest.raises(ValueError, match="no compaction"):
            managed(notes("a")).compact()
        with pytest.raises(ValueError, match="unknown encoding"):
            compacting(notes("a")).compact(encoding="nope")


def noting(entries, *answers, **settings):
    """A manager with an NoteTaker whose fake LLM answers ``answers`` in turn; returns it and the LLM."""
    llm = FakeLLM(*answers)
    return managed(entries, note_taker=NoteTaker(llm, schema=TripNotes), **settings), llm


def sent(llm, call=-1):
    """The conversation part of an update prompt: what the updater was given as new."""
    return llm.calls[call]["user_prompt"].text.split("Conversation since the last update:\n")[1]


class TestNotes:
    def test_updated_once_notes_every_turns_finished_with_only_the_entries_since(self):
        manager, llm = noting([UserPrompt(text="q1"), answer("a1"), *notes("c1"), UserPrompt(text="q2"), answer("a2")],
                              notes_json(destination="Kyoto"), notes_json(destination="Tokyo"), notes_every=2)
        assert len(llm.calls) == 1 and sent(llm) == "User: q1\n\nAssistant: a1\n\nContext:\nc1\n\nUser: q2\n\nAssistant: a2"
        assert manager.notes == {"destination": "Kyoto", "constraints": [], "next_step": None}
        for entry in [*notes("c2"), UserPrompt(text="q3"), answer("a3"), UserPrompt(text="q4")]:
            manager.add(entry)
        assert len(llm.calls) == 1  # one finished turn since, and a question still waiting for its answer: not yet
        manager.add(answer("a4"))
        assert sent(llm) == "Context:\nc2\n\nUser: q3\n\nAssistant: a3\n\nUser: q4\n\nAssistant: a4"  # the new ones only
        assert '"destination": "Kyoto"' in llm.calls[1]["user_prompt"].text  # with the current notes
        assert manager.notes["destination"] == "Tokyo"

    def test_a_question_does_not_finish_a_turn_its_answer_does(self):
        manager, llm = noting([UserPrompt(text="q1"), *notes("a")], notes_json(destination="x"), notes_every=1)
        assert llm.calls == []
        manager.add(answer("r1"))
        assert sent(llm) == "User: q1\n\nContext:\na\n\nAssistant: r1"  # the question and its answer together

    def test_an_answer_that_calls_a_tool_is_the_middle_of_a_turn(self):
        manager, llm = noting([UserPrompt(text="q1"), answer("r1", calls=["m1"]), result("m1")],
                              notes_json(destination="x"), notes_every=1)
        assert llm.calls == []
        manager.add(answer("r1b"))
        assert len(llm.calls) == 1 and sent(llm).endswith("Assistant: r1b")

    def test_active_is_the_summary_the_notes_then_the_window(self):
        llm = FakeLLM(notes_json(destination="Kyoto"))
        manager = managed([SystemPrompt(text="S"), UserPrompt(text="q1"), answer("r1"), *notes("a", "b")],
                          policy=LastNPolicy(1),
                          compaction=RollingCompaction(FakeSummarizer()),
                          note_taker=NoteTaker(llm, schema=TripNotes), notes_every=1)
        active = manager.active
        assert isinstance(active[0], SummaryEntry) and isinstance(active[1], WorkingNotesEntry)
        assert texts(active[2:]) == ["S", "b"]
        assert [type(e).__name__ for e in manager.pinned] == ["SummaryEntry", "WorkingNotesEntry", "SystemPrompt"]

    def test_the_notes_are_never_in_the_window_nor_summarized(self):
        summarizer = FakeSummarizer()
        manager = managed([UserPrompt(text="q1"), answer("r1"), *notes("a", "b", "c")], policy=LastNPolicy(1),
                          compaction=RollingCompaction(summarizer),
                          note_taker=NoteTaker(FakeLLM(notes_json(destination="x")), schema=TripNotes),
                          notes_every=1)
        assert all(not any("Working notes" in label for label in call[0]) for call in summarizer.calls)
        assert isinstance(manager.active[1], WorkingNotesEntry)

    def test_without_notes_every_only_on_demand(self):
        manager, llm = noting([UserPrompt(text="q1"), UserPrompt(text="q2")], notes_json(destination="Kyoto"))
        assert llm.calls == [] and manager.notes is None
        update = manager.refresh_notes()
        assert update.notes == manager.notes and update.response.text == notes_json(destination="Kyoto")

    def test_an_answer_that_does_not_fit_keeps_the_notes_and_the_entries_for_the_next_update(self):
        manager, llm = noting([UserPrompt(text="q1")], notes_json(destination="Kyoto"),
                              json.dumps({"destination": ["not", "a", "string"]}), notes_json(destination="Tokyo"))
        manager.refresh_notes()
        manager.add(UserPrompt(text="q2"))
        with pytest.raises(ValueError, match="do not fit TripNotes"):
            manager.refresh_notes()
        assert manager.notes["destination"] == "Kyoto"
        manager.add(UserPrompt(text="q3"))
        manager.refresh_notes()
        assert sent(llm) == "User: q2\n\nUser: q3" and manager.notes["destination"] == "Tokyo"

    def test_an_answer_that_leaves_out_sections_keeps_the_notes_and_the_entries_for_the_next_update(self):
        kept = notes_json(destination="Kyoto", constraints=["Direct flights only"])
        manager, llm = noting([UserPrompt(text="q1")], kept, json.dumps({"destination": "Kyoto"}),
                              notes_json(destination="Kyoto", constraints=["Direct flights only", "Vegetarian"]))
        manager.refresh_notes()
        manager.add(UserPrompt(text="q2"))
        with pytest.raises(ValueError, match="leave out sections that had content: constraints"):
            manager.refresh_notes()
        assert manager.notes["constraints"] == ["Direct flights only"]  # not wiped
        manager.add(UserPrompt(text="q3"))
        manager.refresh_notes()
        assert sent(llm) == "User: q2\n\nUser: q3"  # the refused update's entries were not lost
        assert manager.notes["constraints"] == ["Direct flights only", "Vegetarian"]

    def test_a_continued_session_restores_the_notes_and_what_arrived_since(self):
        storage = InMemoryStorage()
        first = ShortTermMemoryManager(storage, note_taker=NoteTaker(FakeLLM(notes_json(destination="Kyoto")),
                                                                     schema=TripNotes), session_id="s1")
        first.add(UserPrompt(text="q1"))
        first.refresh_notes()
        first.add(UserPrompt(text="q2"))
        first.add(answer("r2"))
        llm = FakeLLM(notes_json(destination="Tokyo"))
        again = ShortTermMemoryManager(storage, note_taker=NoteTaker(llm, schema=TripNotes), notes_every=2,
                              session_id="s1")
        assert again.notes["destination"] == "Kyoto" and llm.calls == []
        again.add(UserPrompt(text="q3"))
        again.add(answer("r3"))  # r2 and r3: two finished turns since the stored version
        assert sent(llm) == "User: q2\n\nAssistant: r2\n\nUser: q3\n\nAssistant: r3"
        assert again.notes["destination"] == "Tokyo"

    def test_refused(self):
        manager, _ = noting([UserPrompt(text="q1")], notes_json(destination="Kyoto"))
        manager.refresh_notes()
        with pytest.raises(ValueError, match="no entries"):
            manager.refresh_notes()  # nothing arrived since
        with pytest.raises(ValueError, match="no note_taker"):
            managed(notes("a")).refresh_notes()
        with pytest.raises(ValueError, match="notes come from the note taker"):
            manager.add(WorkingNotesEntry(notes={"destination": "x"}))
        notes_id = manager.active[0].id_
        with pytest.raises(ValueError, match="pinned by rule"):
            manager.pin(notes_id)
        with pytest.raises(ValueError, match="stay pinned"):
            manager.unpin(notes_id)

    @pytest.mark.parametrize("settings, message", [
        ({"notes_every": 2}, "notes_every needs a note_taker"),
        ({"note_taker": NoteTaker(FakeLLM(), schema=TripNotes), "notes_every": 0},
         "notes_every must be a positive"),
    ])
    def test_invalid_settings(self, settings, message):
        with pytest.raises(ValueError, match=message):
            ShortTermMemoryManager(InMemoryStorage(), **settings)


def filters_keep(policy, entries):
    """What the filter part of ``policy`` keeps: all of a filter, the leading filters of a chain, else everything."""
    if policy.is_filter:
        return policy.select(entries)
    if isinstance(policy, ChainPolicy):
        for part in [p for p in policy.policies if p.is_filter]:
            entries = part.select(entries)
    return entries


class TestDumpEvery:
    def test_dumps_by_itself_after_every_n_adds(self):
        dump = InMemoryStorage()
        manager = managed(notes("a", "b", "c", "d", "e"), dump_storage=dump, dump_every=2)
        assert texts(dump.entries(manager.session_id)) == ["a", "b", "c", "d"]  # after b, then after d
        manager.add(ContextEntry(text="f"))
        assert texts(dump.entries(manager.session_id)) == ["a", "b", "c", "d", "e", "f"]

    def test_a_manual_dump_starts_the_count_again(self):
        dump = InMemoryStorage()
        manager = managed(notes("a"), dump_storage=dump, dump_every=2)
        manager.dump()
        manager.add(ContextEntry(text="b"))
        assert texts(dump.entries(manager.session_id)) == ["a"]  # one add since the dump, not two

    def test_each_dump_includes_the_summary_made_in_the_same_add(self):
        dump = InMemoryStorage()
        manager = compacting(notes("a", "b", "c"), policy=LastNPolicy(2), dump_storage=dump, dump_every=3)
        dumped = dump.entries(manager.session_id)
        assert [type(e).__name__ for e in dumped] == ["ContextEntry", "ContextEntry", "ContextEntry", "SummaryEntry"]

    @pytest.mark.parametrize("settings, message", [
        ({"dump_every": 2}, "dump_every needs a dump_storage"),
        ({"dump_storage": InMemoryStorage(), "dump_every": 0}, "dump_every must be a positive integer"),
    ])
    def test_invalid_settings(self, settings, message):
        with pytest.raises(ValueError, match=message):
            ShortTermMemoryManager(InMemoryStorage(), **settings)


class TestAsyncConcurrency:
    def test_concurrent_adds_run_one_at_a_time_in_the_order_they_were_started(self):
        summarizer = FakeSummarizer()
        manager = ShortTermMemoryManager(InMemoryStorage(), policy=LastNPolicy(2),
                                         compaction=RollingCompaction(summarizer))
        entries = notes(*"abcdef")

        async def session():
            await asyncio.gather(*(manager.add_async(entry) for entry in entries))

        asyncio.run(session())
        assert texts([e for e in manager.entries if not isinstance(e, SummaryEntry)]) == list("abcdef")
        assert texts(manager.window) == ["e", "f"]
        assert [call[0] for call in summarizer.calls] == [["a"], ["b"], ["c"], ["d"]]  # one drop per add, in order

    def test_an_add_and_a_compact_started_together_do_not_interleave(self):
        manager = ShortTermMemoryManager(InMemoryStorage(), compaction=RollingCompaction(FakeSummarizer()))
        for entry in notes("a", "b"):
            manager.add(entry)

        async def session():
            await asyncio.gather(manager.compact_async(), manager.add_async(ContextEntry(text="c")))

        asyncio.run(session())
        assert texts(manager.window) == ["c"] and covered(manager) == [e.id_ for e in manager.entries[:2]]
