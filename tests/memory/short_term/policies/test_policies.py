import pytest

from hiyal.memory.short_term import (
    ChainPolicy,
    ContextEntry,
    FullPolicy,
    LastNPolicy,
    SlidingWindowPolicy,
    SummaryEntry,
    ToolResultEntry,
    TypeFilterPolicy,
)
from hiyal.prompts import UserPrompt


def numbered(count):
    """Context entries whose text is their position, starting at 1."""
    return [ContextEntry(text=str(number)) for number in range(1, count + 1)]


def texts(entries):
    return [int(entry.text) for entry in entries]


class TestFullPolicy:
    def test_everything_is_active(self):
        entries = numbered(5)
        assert FullPolicy().select(entries) == entries

    def test_returns_a_new_list(self):
        entries = numbered(3)
        assert FullPolicy().select(entries) is not entries


class TestLastNPolicy:
    def test_keeps_the_last_n(self):
        assert texts(LastNPolicy(3).select(numbered(10))) == [8, 9, 10]

    def test_fewer_entries_than_n(self):
        assert texts(LastNPolicy(5).select(numbered(2))) == [1, 2]

    def test_empty_history(self):
        assert LastNPolicy(5).select([]) == []

    @pytest.mark.parametrize("n", [0, -1, 2.5, True, "3"])
    def test_n_must_be_a_positive_integer(self, n):
        with pytest.raises(ValueError, match="positive integer"):
            LastNPolicy(n)


class TestSlidingWindowPolicy:
    @pytest.mark.parametrize("count, first", [(250, 1), (251, 51), (300, 51), (301, 101), (351, 151)])
    def test_the_docstring_example(self, count, first):
        active = SlidingWindowPolicy(size=250, step=50).select(numbered(count))
        assert texts(active)[0] == first and texts(active)[-1] == count

    def test_never_more_than_size(self):
        policy = SlidingWindowPolicy(size=10, step=3)
        assert all(len(policy.select(numbered(count))) <= 10 for count in range(40))

    def test_start_moves_only_in_steps(self):
        policy = SlidingWindowPolicy(size=10, step=4)
        starts = [policy.start(count) for count in range(30)]
        assert set(starts) == {0, 4, 8, 12, 16, 20}
        assert starts == sorted(starts)

    def test_step_one_behaves_like_last_n(self):
        window, last = SlidingWindowPolicy(size=7, step=1), LastNPolicy(7)
        for count in range(20):
            entries = numbered(count)
            assert window.select(entries) == last.select(entries)

    def test_step_equal_to_size_jumps_a_whole_window(self):
        policy = SlidingWindowPolicy(size=5, step=5)
        assert texts(policy.select(numbered(6))) == [6]
        assert texts(policy.select(numbered(10))) == [6, 7, 8, 9, 10]

    @pytest.mark.parametrize("size, step, message", [
        (0, 1, "size must be"), (5, 0, "step must be"), (5, 6, "cannot be larger"), (5, 1.5, "step must be"),
    ])
    def test_invalid_settings(self, size, step, message):
        with pytest.raises(ValueError, match=message):
            SlidingWindowPolicy(size=size, step=step)


class TestTypeFilterPolicy:
    def test_removes_excluded_types_and_keeps_the_order(self):
        question = UserPrompt(text="q")
        context = ContextEntry(text="c")
        result = ToolResultEntry(message_id="m", name="f", arguments={}, output=1)
        summary = SummaryEntry(text="s", covers_through="x", covers_count=1)
        policy = TypeFilterPolicy(exclude=[ToolResultEntry, SummaryEntry])
        assert policy.select([question, result, context, summary]) == [question, context]

    def test_exclude_cannot_be_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            TypeFilterPolicy(exclude=[])


class TestChainPolicy:
    @staticmethod
    def mixed():
        """Twenty entries: every fourth is a tool result."""
        return [ToolResultEntry(message_id="m", name="f", arguments={}, output=n) if n % 4 == 0 else ContextEntry(text=str(n))
                for n in range(1, 21)]

    def test_filter_then_window_gives_a_full_window(self):
        chain = ChainPolicy([TypeFilterPolicy(exclude=[ToolResultEntry]), LastNPolicy(5)])
        assert texts(chain.select(self.mixed())) == [14, 15, 17, 18, 19]

    def test_filters_run_first_whatever_the_order(self):
        chain = ChainPolicy([LastNPolicy(5), TypeFilterPolicy(exclude=[ToolResultEntry])])
        assert texts(chain.select(self.mixed())) == [14, 15, 17, 18, 19]
        assert [type(policy) for policy in chain.policies] == [TypeFilterPolicy, LastNPolicy]

    def test_filters_and_windows_each_keep_their_order(self):
        first, second = TypeFilterPolicy(exclude=[ToolResultEntry]), TypeFilterPolicy(exclude=[SummaryEntry])
        wide, narrow = LastNPolicy(6), LastNPolicy(2)
        chain = ChainPolicy([wide, first, narrow, second])
        assert chain.policies == [first, second, wide, narrow]

    def test_which_policies_are_filters(self):
        assert TypeFilterPolicy(exclude=[ToolResultEntry]).is_filter and FullPolicy().is_filter
        assert not LastNPolicy(3).is_filter and not SlidingWindowPolicy(5, 2).is_filter
        assert ChainPolicy([FullPolicy(), TypeFilterPolicy(exclude=[ToolResultEntry])]).is_filter
        assert not ChainPolicy([FullPolicy(), LastNPolicy(3)]).is_filter

    def test_a_single_policy_chain_is_that_policy(self):
        entries = numbered(10)
        assert ChainPolicy([LastNPolicy(3)]).select(entries) == LastNPolicy(3).select(entries)

    def test_chains_can_be_nested(self):
        inner = ChainPolicy([LastNPolicy(6)])
        assert texts(ChainPolicy([inner, LastNPolicy(2)]).select(numbered(10))) == [9, 10]

    def test_needs_at_least_one_policy(self):
        with pytest.raises(ValueError, match="at least one policy"):
            ChainPolicy([])


@pytest.mark.parametrize("policy", [
    FullPolicy(), LastNPolicy(3), SlidingWindowPolicy(5, 2), TypeFilterPolicy([SummaryEntry]),
    ChainPolicy([FullPolicy(), LastNPolicy(2)]),
], ids=lambda policy: type(policy).__name__)
def test_no_policy_changes_its_input(policy):
    entries = numbered(10)
    before = list(entries)
    policy.select(entries)
    assert entries == before



class TestFilterPart:
    def test_a_filter_keeps_its_choice_a_window_keeps_everything(self):
        entries = [UserPrompt(text="1"), ContextEntry(text="2"), UserPrompt(text="3")]
        assert texts(TypeFilterPolicy(exclude=[ContextEntry]).filter(entries)) == [1, 3]
        assert texts(LastNPolicy(1).filter(entries)) == [1, 2, 3]

    def test_a_chain_runs_only_its_filters(self):
        entries = [UserPrompt(text="1"), ContextEntry(text="2"), UserPrompt(text="3")]
        chain = ChainPolicy([LastNPolicy(1), TypeFilterPolicy(exclude=[ContextEntry])])
        assert texts(chain.filter(entries)) == [1, 3]  # the window cuts nothing here
