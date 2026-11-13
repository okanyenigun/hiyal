import asyncio
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import (
    BaseCompaction,
    RebuildCompaction,
    RollingCompaction,
    SummaryEntry,
    SummaryUpdate,
)
from hiyal.memory.short_term.summarizers import BaseSummarizer
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import UserPrompt


@pytest.fixture(autouse=True, params=["sync", "async"])
def mode(request, monkeypatch):
    """Run every test twice: through ``compact``, then through ``compact_async``, which must behave the same."""
    if request.param == "async":
        for cls in (RollingCompaction, RebuildCompaction):
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


def ids(entries):
    return [entry.id_ for entry in entries]


class TestRollingCompaction:
    def test_the_first_summary_covers_the_new_entries(self):
        summarizer = FakeSummarizer()
        newly = turns(2)
        update = RollingCompaction(summarizer).compact(newly, None, [])
        assert isinstance(update, SummaryUpdate) and update.response.text == update.summary.text
        assert summarizer.calls == [(["q1", "a1", "q2", "a2"], None)]
        assert update.summary.covers == ids(newly)

    def test_folds_only_the_new_entries_into_the_previous_summary(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        previous = SummaryEntry(text="earlier", covers=ids(old))
        update = RollingCompaction(summarizer).compact(newly, previous, old)
        assert summarizer.calls == [(["q3", "a3"], "earlier")]  # the originals are not resent
        assert update.summary.covers == ids(old) + ids(newly)

    def test_nothing_new_to_summarize(self):
        with pytest.raises(ValueError, match="nothing to summarize"):
            RollingCompaction(FakeSummarizer()).compact([], None, [])


class TestRebuildCompaction:
    def test_rebuilds_from_the_originals_without_the_previous_summary(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        previous = SummaryEntry(text="earlier", covers=ids(old))
        update = RebuildCompaction(summarizer).compact(newly, previous, old)
        assert summarizer.calls == [(["q1", "a1", "q2", "a2", "q3", "a3"], None)]
        assert update.summary.covers == ids(old) + ids(newly)

    def test_falls_back_on_the_previous_summary_for_originals_that_are_gone(self):
        summarizer = FakeSummarizer()
        old, newly = turns(2), turns(1, start=3)
        previous = SummaryEntry(text="earlier", covers=ids(old))
        update = RebuildCompaction(summarizer).compact(newly, previous, old[1:])  # q1 is gone
        assert summarizer.calls == [(["a1", "q2", "a2", "q3", "a3"], "earlier")]
        assert update.summary.covers == ids(old) + ids(newly)  # still covers q1

    def test_nothing_new_to_summarize(self):
        with pytest.raises(ValueError, match="nothing to summarize"):
            RebuildCompaction(FakeSummarizer()).compact([], None, [])


class TestAsync:
    def test_a_compaction_without_its_own_async_compact_runs_compact_in_a_thread(self):
        class Fixed(BaseCompaction):
            def compact(self, newly, previous, covered):
                return SummaryUpdate(SummaryEntry(text="s", covers=ids(newly)), reply("s"))

        newly = turns(1)
        update = asyncio.run(Fixed(FakeSummarizer()).compact_async(newly, None, []))
        assert update.summary.covers == ids(newly)
