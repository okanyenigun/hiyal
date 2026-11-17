import asyncio
import threading

import pytest

from hiyal.memory.short_term import (
    Context,
    Entry,
    EntryKind,
    FilterPolicy,
    HeadTailPolicy,
    KindFilterPolicy,
    LastNPolicy,
    Policy,
    PredicateFilterPolicy,
    Selection,
    SlidingWindowPolicy,
    TokenWindowPolicy,
    ToolResult,
    WindowPolicy,
)
from hiyal.utility import Tokens


def note(name):
    """A context entry with ID ``name``."""
    return Entry(id_=name, kind=EntryKind.CONTEXT, payload=Context(text=name))


def result(name):
    """A tool result entry with ID ``name``."""
    return Entry(id_=name, kind=EntryKind.TOOL_RESULT,
                 payload=ToolResult(message_id="m", name="search", arguments={}, output=name))


def ids(entries):
    return [entry.id_ for entry in entries]


def conversation():
    """q1 t1 q2 t2 q3 t3: questions as context entries, each followed by a tool result."""
    return [entry for n in (1, 2, 3) for entry in (note(f"q{n}"), result(f"t{n}"))]


def entries(count):
    return [note(f"e{index}") for index in range(count)]


NO_TOOLS = [EntryKind.TOOL_RESULT]


@pytest.fixture(params=["sync", "async"])
def choose(request):
    """A test taking ``choose`` runs twice: once with ``choose``, once with ``choose_async``."""

    def run(policy, entries):
        if request.param == "sync":
            return policy.choose(entries)
        return asyncio.run(policy.choose_async(entries))

    return run


class TestChain:
    def test_then_returns_the_first_policy(self):
        first = KindFilterPolicy(exclude=NO_TOOLS)
        assert first.then(LastNPolicy(2)) is first

    def test_a_filter_then_a_window(self, choose):
        chosen = choose(KindFilterPolicy(exclude=NO_TOOLS).then(LastNPolicy(2)), conversation())
        assert ids(chosen.kept) == ["q2", "q3"]
        assert ids(chosen.too_old) == ["q1"]
        assert ids(chosen.unwanted) == ["t1", "t2", "t3"]

    def test_a_window_then_a_filter_is_another_choice(self, choose):
        chosen = choose(LastNPolicy(2).then(KindFilterPolicy(exclude=NO_TOOLS)), conversation())
        assert ids(chosen.kept) == ["q3"]
        assert ids(chosen.too_old) == ["q1", "t1", "q2", "t2"]
        assert ids(chosen.unwanted) == ["t3"]

    def test_drops_from_several_steps_keep_the_history_order(self, choose):
        chain = LastNPolicy(5).then(KindFilterPolicy(exclude=NO_TOOLS)).then(LastNPolicy(1))
        chosen = choose(chain, conversation())
        assert ids(chosen.kept) == ["q3"]
        assert ids(chosen.too_old) == ["q1", "q2"]
        assert ids(chosen.unwanted) == ["t1", "t2", "t3"]

    def test_a_policy_alone_is_a_chain(self, choose):
        given = entries(3)
        assert choose(LastNPolicy(5), given) == Selection(kept=given, too_old=[], unwanted=[])

    def test_the_entries_given_are_not_changed(self, choose):
        given = conversation()
        before = list(given)
        choose(KindFilterPolicy(exclude=NO_TOOLS).then(LastNPolicy(1)), given)
        assert given == before

    def test_a_policy_already_in_the_chain_is_refused(self):
        a, b = LastNPolicy(5), LastNPolicy(3)
        with pytest.raises(ValueError, match="only once"):
            a.then(a)
        a.then(b)
        with pytest.raises(ValueError, match="only once"):
            a.then(b)
        with pytest.raises(ValueError, match="only once"):
            b.then(a)

    def test_only_policies_can_be_linked(self):
        with pytest.raises(TypeError, match="not a str"):
            LastNPolicy(1).then("LastNPolicy(2)")  # type: ignore[arg-type]

    def test_a_policy_that_adds_or_reorders_entries_is_refused(self, choose):
        class Reverse(WindowPolicy):
            def select(self, entries):
                return entries[::-1]

        class Invent(WindowPolicy):
            def select(self, entries):
                return [*entries, note("new")]

        class Twice(WindowPolicy):
            def select(self, entries):
                return [entries[0], entries[0]]

        class Copies(WindowPolicy):
            def select(self, entries):
                return [entry.model_copy() for entry in entries]  # equal, but not the entries given

        for policy in (Reverse(), Invent(), Twice(), Copies()):
            with pytest.raises(ValueError, match="a policy only selects"):
                choose(policy, entries(3))

    def test_a_linked_chain_is_added_whole(self, choose):
        first, second, third = KindFilterPolicy(exclude=NO_TOOLS), LastNPolicy(2), LastNPolicy(1)
        assert first.then(second.then(third)) is first
        chosen = choose(first, conversation())
        assert ids(chosen.kept) == ["q3"] and ids(chosen.too_old) == ["q1", "q2"]

    @pytest.mark.parametrize("policy", [
        KindFilterPolicy(exclude=NO_TOOLS), PredicateFilterPolicy(lambda entry: True), LastNPolicy(2),
        SlidingWindowPolicy(size=2, step=1), HeadTailPolicy(head=1, tail=1), TokenWindowPolicy(10),
    ], ids=lambda policy: type(policy).__name__)
    def test_no_entries_give_an_empty_choice(self, choose, policy):
        assert choose(policy, []) == Selection(kept=[], too_old=[], unwanted=[])

    def test_a_policy_may_keep_nothing(self, choose):
        chosen = choose(KindFilterPolicy(exclude=[EntryKind.CONTEXT, EntryKind.TOOL_RESULT]), conversation())
        assert chosen.kept == [] and ids(chosen.unwanted) == ids(conversation())

    def test_a_policy_must_be_a_filter_or_a_window(self, choose):
        class Neither(Policy):
            def select(self, entries):
                return entries

        with pytest.raises(TypeError, match="must be a FilterPolicy or a WindowPolicy"):
            choose(Neither(), entries(2))

    def test_by_default_select_async_runs_select_directly(self):
        class Threads(WindowPolicy):
            def __init__(self):
                self.threads = []

            def select(self, entries):
                self.threads.append(threading.get_ident())
                return entries

        policy = Threads()
        asyncio.run(policy.choose_async(entries(2)))
        assert policy.threads == [threading.get_ident()]

    def test_a_policy_can_wait_in_its_own_select_async(self):
        class NoToolsLater(FilterPolicy):
            def select(self, entries):
                raise AssertionError("the async chain runs select_async")

            async def select_async(self, entries):
                await asyncio.sleep(0)
                return [entry for entry in entries if entry.kind != EntryKind.TOOL_RESULT]

        chosen = asyncio.run(NoToolsLater().then(LastNPolicy(2)).choose_async(conversation()))
        assert ids(chosen.kept) == ["q2", "q3"]
        assert ids(chosen.too_old) == ["q1"]
        assert ids(chosen.unwanted) == ["t1", "t2", "t3"]


class TestKindFilterPolicy:
    def test_entries_of_the_excluded_kinds_are_dropped(self):
        assert ids(KindFilterPolicy(exclude=NO_TOOLS).select(conversation())) == ["q1", "q2", "q3"]

    def test_kinds_can_be_given_by_value(self):
        assert KindFilterPolicy(exclude=["tool_result"]).exclude == {EntryKind.TOOL_RESULT}

    def test_exclude_can_be_any_iterable_and_repeats_are_fine(self):
        policy = KindFilterPolicy(exclude=(kind for kind in ["tool_result", EntryKind.TOOL_RESULT, "context"]))
        assert policy.exclude == {EntryKind.TOOL_RESULT, EntryKind.CONTEXT}

    def test_an_empty_or_unknown_exclude_is_refused(self):
        with pytest.raises(ValueError, match="at least one entry kind"):
            KindFilterPolicy(exclude=[])
        with pytest.raises(ValueError, match="'tool'"):
            KindFilterPolicy(exclude=["tool"])


class TestPredicateFilterPolicy:
    def test_entries_keep_says_yes_to_are_kept(self):
        policy = PredicateFilterPolicy(lambda entry: entry.id_.endswith("2"))
        assert ids(policy.select(conversation())) == ["q2", "t2"]

    def test_any_truthy_answer_keeps(self):
        policy = PredicateFilterPolicy(lambda entry: {"q1": 1, "q2": "yes", "q3": []}.get(entry.id_, 0))
        assert ids(policy.select(conversation())) == ["q1", "q2"]

    def test_keep_must_be_callable(self):
        with pytest.raises(TypeError, match="keep must be callable"):
            PredicateFilterPolicy("q2")  # type: ignore[arg-type]


class TestLastNPolicy:
    def test_the_last_n_are_kept(self):
        assert ids(LastNPolicy(2).select(entries(5))) == ["e3", "e4"]

    def test_fewer_than_n_are_all_kept(self):
        assert ids(LastNPolicy(5).select(entries(2))) == ["e0", "e1"]

    @pytest.mark.parametrize("n", [0, -1, 1.5])
    def test_n_must_be_a_positive_integer(self, n):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            LastNPolicy(n)


class TestSlidingWindowPolicy:
    @pytest.mark.parametrize("count, start", [(3, 0), (4, 0), (5, 2), (6, 2), (7, 4)])
    def test_the_start_moves_in_steps(self, count, start):
        policy = SlidingWindowPolicy(size=4, step=2)
        assert policy.start(count) == start
        assert ids(policy.select(entries(count))) == [f"e{index}" for index in range(start, count)]

    def test_step_cannot_be_larger_than_size(self):
        with pytest.raises(ValueError, match="cannot be larger than size"):
            SlidingWindowPolicy(size=2, step=3)

    @pytest.mark.parametrize("count, start", [(4, 0), (5, 4), (8, 4), (9, 8)])
    def test_a_step_as_large_as_the_size_drops_the_whole_window(self, count, start):
        assert SlidingWindowPolicy(size=4, step=4).start(count) == start

    @pytest.mark.parametrize("size, step, message", [
        (0, 1, "size must be a positive integer"), (2, 0, "step must be a positive integer"),
        (2.5, 1, "size must be a positive integer"),
    ])
    def test_size_and_step_must_be_positive_integers(self, size, step, message):
        with pytest.raises(ValueError, match=message):
            SlidingWindowPolicy(size=size, step=step)


class TestHeadTailPolicy:
    def test_the_first_and_the_last_entries_are_kept(self):
        assert ids(HeadTailPolicy(head=1, tail=2).select(entries(5))) == ["e0", "e3", "e4"]

    def test_with_no_more_than_head_and_tail_all_are_kept(self):
        assert ids(HeadTailPolicy(head=1, tail=2).select(entries(3))) == ["e0", "e1", "e2"]

    def test_head_and_tail_must_be_positive_integers(self):
        with pytest.raises(ValueError, match="head must be a positive integer"):
            HeadTailPolicy(head=0, tail=2)
        with pytest.raises(ValueError, match="tail must be a positive integer"):
            HeadTailPolicy(head=1, tail=-1)

    def test_one_more_than_head_and_tail_drops_one(self):
        assert ids(HeadTailPolicy(head=1, tail=2).select(entries(4))) == ["e0", "e2", "e3"]


def sized(*sizes):
    """Entries whose text is their size, for ``by_text``."""
    return [note(str(size)) for size in sizes]


def by_text(entry):
    return int(entry.payload.text)


class TestTokenWindowPolicy:
    def test_nothing_is_cut_within_max_tokens(self):
        assert ids(TokenWindowPolicy(25, 10, count=by_text).select(sized(10, 10))) == ["10", "10"]

    def test_passing_max_tokens_cuts_down_to_target_tokens(self):
        given = [note("10"), note("11"), note("12")]
        policy = TokenWindowPolicy(25, 12, count=by_text)
        # 10, 21, then 33 > 25: drop 10 (23 > 12), drop 11 (12): only the newest is left.
        assert ids(policy.select(given)) == ["12"]
        # The start stays while the window grows again: 12 + 13 = 25.
        assert ids(policy.select([*given, note("13")])) == ["12", "13"]

    def test_the_newest_entry_stays_even_alone_over_max_tokens(self):
        assert ids(TokenWindowPolicy(25, count=by_text).select(sized(5, 40))) == ["40"]

    def test_target_tokens_defaults_to_half(self):
        assert TokenWindowPolicy(25).target_tokens == 12

    def test_target_tokens_is_at_least_1(self):
        assert TokenWindowPolicy(1).target_tokens == 1

    def test_reaching_max_tokens_exactly_is_not_passing_it(self):
        assert ids(TokenWindowPolicy(20, 5, count=by_text).select(sized(10, 10))) == ["10", "10"]

    def test_a_cut_stops_as_soon_as_the_window_is_down_to_target_tokens(self):
        # 5, 10, 20, then 30 > 25: drop 5 (25), drop 5 (20 = target): the rest stay.
        assert ids(TokenWindowPolicy(25, 20, count=by_text).select(sized(5, 5, 10, 10))) == ["10", "10"]

    def test_target_equal_to_max_trims_just_enough(self):
        # 10, 20, then 30 > 20: drop one 10 (20); then 30 again: drop the next.
        assert ids(TokenWindowPolicy(20, 20, count=by_text).select(sized(10, 10, 10, 10))) == ["10", "10"]

    def test_entries_counted_as_0_never_cause_a_cut(self):
        assert len(TokenWindowPolicy(1, count=lambda entry: 0).select(entries(50))) == 50

    def test_by_default_entries_are_counted_by_their_text(self, monkeypatch):
        monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))
        # Each context entry reads "Context:" and one word: 2 tokens; 6 > 5 cuts down to 2.
        assert ids(TokenWindowPolicy(5, 2).select(entries(3))) == ["e2"]

    def test_invalid_settings_are_refused(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            TokenWindowPolicy(0)
        with pytest.raises(ValueError, match="target_tokens must be a positive integer"):
            TokenWindowPolicy(10, 0)
        with pytest.raises(ValueError, match="cannot be larger than max_tokens"):
            TokenWindowPolicy(10, 20)
        with pytest.raises(TypeError, match="count must be callable"):
            TokenWindowPolicy(10, count=5)  # type: ignore[arg-type]
