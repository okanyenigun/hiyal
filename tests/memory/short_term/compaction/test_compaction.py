import asyncio
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import (
    BaseCompaction,
    HybridCompaction,
    RebuildCompaction,
    RollingCompaction,
    SummaryEntry,
)
from hiyal.memory.short_term.summarizers import BaseSummarizer
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import UserPrompt


@pytest.fixture(autouse=True, params=["sync", "async"])
def mode(request, monkeypatch):
    """Run every test twice: through ``compact``, then through ``compact_async``, which must behave the same."""
    if request.param == "async":
        for cls in (RollingCompaction, RebuildCompaction, HybridCompaction):
            original = cls.compact_async
            monkeypatch.setattr(cls, "compact", lambda self, newly, previous, covered, original=original:
                                asyncio.run(original(self, newly, previous, covered)))
    return request.param


class FakeSummarizer(BaseSummarizer):
    """Records what it is asked to summarize; answers 'summary #N of <texts>'."""

    def __init__(self):
        super().__init__(llm=None)
        self.calls = []

    def render(self, entries, previous=None):
        return UserPrompt(text="unused")

    def summarize(self, entries, previous=None):
        self.calls.append(([entry.text for entry in entries], previous.text if previous else None))
        text = f"summary #{len(self.calls)} of {' '.join(entry.text for entry in entries)}"
        return LLMResponse(text=text, messages=[], provider="openai", id_=f"sum-{len(self.calls)}", model="m",
                           created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")

    async def summarize_async(self, entries, previous=None):
        return self.summarize(entries, previous)


def ask(text):
    return UserPrompt(text=text)


def reply(text):
    return LLMResponse(text=text, messages=[], provider="openai", id_=f"resp-{text}", model="m",
                       created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")


def turns(count, start=1):
    """``count`` question/answer pairs: q1 a1 q2 a2 ..."""
    entries = []
    for number in range(start, start + count):
        entries += [ask(f"q{number}"), reply(f"a{number}")]
    return entries


def summary_of(entries, text="earlier", version=1):
    """A summary covering ``entries``, as the manager records it."""
    return SummaryEntry(text=text, covers_through=entries[-1].id_, covers_count=len(entries), version=version)


class TestRollingCompaction:
    def test_the_first_summary_is_the_summarizers_response(self):
        summarizer = FakeSummarizer()
        response = RollingCompaction(summarizer).compact(turns(2), None, [])
        assert isinstance(response, LLMResponse) and response.text
        assert summarizer.calls == [(["q1", "a1", "q2", "a2"], None)]

    def test_folds_only_the_new_entries_into_the_previous_summary(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        RollingCompaction(summarizer).compact(newly, summary_of(old), old)
        assert summarizer.calls == [(["q3", "a3"], "earlier")]  # the originals are not resent

    def test_nothing_new_to_summarize(self):
        with pytest.raises(ValueError, match="nothing to summarize"):
            RollingCompaction(FakeSummarizer()).compact([], None, [])


class TestRebuildCompaction:
    def test_rebuilds_from_the_originals_without_the_previous_summary(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        RebuildCompaction(summarizer).compact(newly, summary_of(old), old)
        assert summarizer.calls == [(["q1", "a1", "q2", "a2", "q3", "a3"], None)]

    def test_falls_back_on_the_previous_summary_for_originals_that_are_gone(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        RebuildCompaction(summarizer).compact(newly, summary_of(old), old[1:])  # q1 is gone
        assert summarizer.calls == [(["a1", "q2", "a2", "q3", "a3"], "earlier")]

    def test_nothing_new_to_summarize(self):
        with pytest.raises(ValueError, match="nothing to summarize"):
            RebuildCompaction(FakeSummarizer()).compact([], None, [])


class TestHybridCompaction:
    def test_rolls_between_rebuilds(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        compaction = HybridCompaction(summarizer, rebuild_every=3)
        previous = summary_of(old, version=1)  # the next is version 2: a roll
        assert not compaction.needs_covered(previous)
        compaction.compact(newly, previous, [])
        assert summarizer.calls == [(["q3", "a3"], "earlier")]

    def test_rebuilds_every_rebuild_every_versions(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        compaction = HybridCompaction(summarizer, rebuild_every=3)
        previous = summary_of(old, version=2)  # the next is version 3: a rebuild
        assert compaction.needs_covered(previous)
        compaction.compact(newly, previous, old)
        assert summarizer.calls == [(["q1", "a1", "q2", "a2", "q3", "a3"], None)]

    def test_the_first_summary_rolls_unless_every_version_is_rebuilt(self):
        assert not HybridCompaction(FakeSummarizer(), rebuild_every=3).needs_covered(None)
        assert HybridCompaction(FakeSummarizer(), rebuild_every=1).needs_covered(None)

    @pytest.mark.parametrize("rebuild_every", [0, -2, 1.5, True])
    def test_rebuild_every_must_be_a_positive_integer(self, rebuild_every):
        with pytest.raises(ValueError, match="rebuild_every must be a positive integer"):
            HybridCompaction(FakeSummarizer(), rebuild_every=rebuild_every)


class TestNeedsCovered:
    def test_rolling_never_and_rebuild_always(self):
        previous = summary_of(turns(1))
        assert not RollingCompaction(FakeSummarizer()).needs_covered(previous)
        assert RebuildCompaction(FakeSummarizer()).needs_covered(previous)


class TestAsync:
    def test_a_compaction_without_its_own_async_compact_runs_compact_in_a_thread(self):
        class Fixed(BaseCompaction):
            def compact(self, newly, previous, covered):
                return reply("s")

        response = asyncio.run(Fixed(FakeSummarizer()).compact_async(turns(1), None, []))
        assert response.text == "s"
