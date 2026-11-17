import asyncio
import threading
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import (
    BudgetSummarizer,
    ChunkedSummarizer,
    SingleCallSummarizer,
    SummaryWorker,
)
from hiyal.models.schemas import LLMResponse
from hiyal.prompts.hub import hub
from hiyal.utility import Tokens

SUMMARIZE = hub.get("memory/summarize", "v1")
SHORTEN = hub.get("memory/shorten", "v1")


@pytest.fixture(autouse=True)
def count_words(monkeypatch):
    """Count words instead of tokens, so no test loads a tiktoken encoding (that needs network)."""
    monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))


class ScriptedLLM:
    """Answers each query with the next answer and records the calls.

    An answer is a text, or ``(text, status)`` for a response that is not completed.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
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
        answer = self.answers[len(self.calls) - 1]
        text, status = answer if isinstance(answer, tuple) else (answer, "completed")
        return LLMResponse(text=text, messages=[], provider="openai", id_=f"resp-{len(self.calls)}", model="m",
                           created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status=status)

    def systems(self):
        return [call["system_prompt"].text for call in self.calls]

    def users(self):
        return [call["user_prompt"].text for call in self.calls]


class Tag(SummaryWorker):
    """Adds its name to the text and records that it ran."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def _summarize(self, text):
        self.log.append(self.name)
        return f"{text}{self.name}"


class Threads(SummaryWorker):
    """Records the thread each ``_summarize`` runs in."""

    def __init__(self):
        self.threads = []

    def _summarize(self, text):
        self.threads.append(threading.get_ident())
        return text


class Shout(SummaryWorker):
    """Upper-cases the text; its async side runs directly, and says so."""

    def _summarize(self, text):
        return text.upper()

    async def _summarize_async(self, text):
        return f"{text.upper()}!"


def words(count, name="w"):
    """Text the patched counter counts as ``count`` tokens."""
    return " ".join(f"{name}{index}" for index in range(count))


def paragraphs(*sizes, names="abcdefghij"):
    """Paragraphs of ``sizes`` words, separated by blank lines; paragraph ``n`` uses name ``names[n]``."""
    return "\n\n".join(words(size, names[index]) for index, size in enumerate(sizes))


class TestChain:
    def test_then_returns_the_first_worker_and_summarize_runs_them_in_order(self):
        log = []
        a, b, c = Tag("a", log), Tag("b", log), Tag("c", log)
        chain = a.then(b).then(c)
        assert chain is a
        assert chain.summarize("t") == "tabc"
        assert log == ["a", "b", "c"]

    def test_a_linked_chain_is_added_whole(self):
        log = []
        chain = Tag("a", log).then(Tag("b", log).then(Tag("c", log)))
        assert chain.summarize("t") == "tabc"

    def test_a_worker_alone_is_a_chain(self):
        assert SingleCallSummarizer(ScriptedLLM("The summary.")).summarize("q and a") == "The summary."

    def test_a_worker_already_in_the_chain_is_refused(self):
        log = []
        a, b = Tag("a", log), Tag("b", log)
        with pytest.raises(ValueError, match="only once"):
            a.then(a)
        a.then(b)
        with pytest.raises(ValueError, match="only once"):
            a.then(b)
        with pytest.raises(ValueError, match="only once"):
            b.then(a)
        assert a.summarize("t") == "tab"

    def test_only_workers_can_be_linked(self):
        with pytest.raises(TypeError, match="not a str"):
            Tag("a", []).then("a worker")  # type: ignore[arg-type]

    def test_only_a_text_can_be_summarized(self):
        with pytest.raises(TypeError, match="takes a text, not a list"):
            Tag("a", []).summarize(["t1", "t2"])  # type: ignore[arg-type]

    def test_the_whole_chain(self):
        # ChunkedSummarizer's one chunk summary, the single call's summary, then one budget round.
        llm = ScriptedLLM(words(2, "s"), words(8, "f"), words(5, "g"))
        summarizer = (
            ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5)
            .then(SingleCallSummarizer(llm))
            .then(BudgetSummarizer(llm, max_tokens=5))
        )
        assert summarizer.summarize(paragraphs(4, 4, 4)) == words(5, "g")
        users = llm.users()
        # The first two paragraphs were one chunk; the third, alone and short, passed through.
        assert users[0].endswith(f"{words(4, 'a')}\n\n{words(4, 'b')}")
        assert users[1] == f"Conversation to summarize:\n{words(2, 's')}\n\n{words(4, 'c')}"
        assert users[2].startswith("This summary is 8 tokens.")


class TestAsyncChain:
    def test_summarize_async_runs_the_workers_in_order(self):
        log = []
        chain = Tag("a", log).then(Tag("b", log)).then(Tag("c", log))
        assert asyncio.run(chain.summarize_async("t")) == "tabc"
        assert log == ["a", "b", "c"]

    def test_by_default_a_workers_async_step_runs_its_step_in_a_worker_thread(self):
        worker = Threads()
        asyncio.run(worker.summarize_async("t"))
        assert worker.threads != [threading.get_ident()]

    def test_each_worker_runs_its_own_async_side(self):
        log = []
        chain = Tag("a", log).then(Shout()).then(Tag("b", log))
        assert asyncio.run(chain.summarize_async("t")) == "TA!b"

    def test_the_whole_chain(self):
        llm = ScriptedLLM(words(2, "s"), words(8, "f"), words(5, "g"))
        summarizer = (
            ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5)
            .then(SingleCallSummarizer(llm))
            .then(BudgetSummarizer(llm, max_tokens=5))
        )
        assert asyncio.run(summarizer.summarize_async(paragraphs(4, 4, 4))) == words(5, "g")
        assert llm.modes == ["async"] * 3

    def test_only_a_text_can_be_summarized(self):
        with pytest.raises(TypeError, match="takes a text, not a list"):
            asyncio.run(Tag("a", []).summarize_async(["t1", "t2"]))  # type: ignore[arg-type]


class TestSingleCallSummarizer:
    def test_one_call_over_the_text(self):
        llm = ScriptedLLM("The summary.")
        assert SingleCallSummarizer(llm).summarize("User: Plan a trip.\n\nAssistant: April.") == "The summary."
        assert llm.systems() == [SUMMARIZE]
        assert llm.users() == ["Conversation to summarize:\nUser: Plan a trip.\n\nAssistant: April."]

    def test_the_hub_prompt_can_be_chosen(self):
        llm = ScriptedLLM("s")
        SingleCallSummarizer(llm, prompt_name="memory/summarize_with_notes", prompt_version="v1").summarize("t")
        assert llm.systems() == [hub.get("memory/summarize_with_notes", "v1")]

    @pytest.mark.parametrize("text", ["", " \n "])
    def test_a_blank_text_is_refused(self, text):
        with pytest.raises(ValueError, match="nothing to summarize"):
            SingleCallSummarizer(ScriptedLLM()).summarize(text)

    def test_a_cut_off_answer_is_refused(self):
        with pytest.raises(ValueError, match="'incomplete', not 'completed'"):
            SingleCallSummarizer(ScriptedLLM(("Half a summ", "incomplete"))).summarize("t")

    def test_an_empty_answer_is_refused(self):
        with pytest.raises(ValueError, match="no text"):
            SingleCallSummarizer(ScriptedLLM("  ")).summarize("t")

    def test_summarize_async_makes_the_same_call_on_the_async_client(self):
        llm = ScriptedLLM("The summary.")
        summary = asyncio.run(SingleCallSummarizer(llm).summarize_async("User: Plan a trip."))
        assert summary == "The summary."
        assert llm.modes == ["async"]
        assert llm.systems() == [SUMMARIZE]
        assert llm.users() == ["Conversation to summarize:\nUser: Plan a trip."]

    def test_async_refuses_a_blank_text_with_no_call(self):
        llm = ScriptedLLM()
        with pytest.raises(ValueError, match="nothing to summarize"):
            asyncio.run(SingleCallSummarizer(llm).summarize_async(" \n "))
        assert llm.calls == []

    @pytest.mark.parametrize(
        "answer, message",
        [(("Half a summ", "incomplete"), "'incomplete', not 'completed'"), ("  ", "no text")],
    )
    def test_async_refuses_a_cut_off_or_empty_answer(self, answer, message):
        with pytest.raises(ValueError, match=message):
            asyncio.run(SingleCallSummarizer(ScriptedLLM(answer)).summarize_async("t"))


class Side:
    """Runs a worker on one side, as an app does: ``summarize`` (``"sync"``) or ``summarize_async`` (``"async"``)."""

    def __init__(self, name):
        self.name = name

    def __call__(self, worker, text):
        return worker.summarize(text) if self.name == "sync" else asyncio.run(worker.summarize_async(text))


@pytest.fixture(params=["sync", "async"])
def summarize(request):
    """A test taking ``summarize`` runs twice: once on each side, with the same expectations."""
    return Side(request.param)


class SlowLLM(ScriptedLLM):
    """A ``ScriptedLLM`` whose async answers take a moment; records the most calls waiting at once."""

    def __init__(self, *answers):
        super().__init__(*answers)
        self.waiting = 0
        self.most_waiting = 0

    async def query_async(self, **parameters):
        self.waiting += 1
        self.most_waiting = max(self.most_waiting, self.waiting)
        await asyncio.sleep(0)  # let the other calls start
        self.waiting -= 1
        return await super().query_async(**parameters)


class TestChunkedSummarizer:
    def test_a_text_that_fits_passes_on_with_no_call(self, summarize):
        llm = ScriptedLLM()
        text = paragraphs(4, 6)
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10), text) == text
        assert llm.calls == []

    def test_each_chunk_is_summarized_and_the_summaries_joined(self, summarize):
        llm = ScriptedLLM("s1 s1", "s2 s2")
        result = summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(4, 4, 4, 4))
        assert result == "s1 s1\n\ns2 s2"
        assert llm.systems() == [f"{SUMMARIZE} Aim for about 3 words."] * 2
        assert llm.users() == [
            "Part 1 of 2 of a longer conversation; the other parts are summarized separately. "
            f"Conversation to summarize:\n{paragraphs(4, 4)}",
            "Part 2 of 2 of a longer conversation; the other parts are summarized separately. "
            f"Conversation to summarize:\n{paragraphs(4, 4, names='cd')}",
        ]

    def test_summaries_still_too_big_go_another_round(self, summarize):
        llm = ScriptedLLM(words(4, "x"), words(4, "y"), words(4, "z"), "c c c")
        result = summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(4, 4, 4, 4, 4, 4))
        # Round 2 chunks the joined summaries: the first two are summarized together; the third, short, is kept.
        assert result == f"c c c\n\n{words(4, 'z')}"
        assert llm.systems()[3] == f"{SUMMARIZE} Aim for about 3 words."
        assert llm.users()[3] == (
            "Part 1 of 2 of a longer conversation; the other parts are summarized separately. "
            f"Conversation to summarize:\n{words(4, 'x')}\n\n{words(4, 'y')}"
        )

    def test_a_paragraph_too_big_for_a_chunk_is_split_at_its_lines(self, summarize):
        llm = ScriptedLLM("s1", "s2")
        text = f"{words(6, 'a')}\n{words(6, 'b')}"
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), text) == "s1\n\ns2"
        assert llm.users()[0].endswith(f"Conversation to summarize:\n{words(6, 'a')}")
        assert llm.users()[1].endswith(f"Conversation to summarize:\n{words(6, 'b')}")

    def test_a_line_too_big_for_a_chunk_is_split_at_its_words(self, summarize):
        llm = ScriptedLLM("s", "s", "s")
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10), words(30)) == "s\n\ns\n\ns"
        assert llm.users()[0].endswith(f"Conversation to summarize:\n{words(10)}")

    def test_a_run_with_no_spaces_is_cut_anywhere(self, monkeypatch, summarize):
        monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text)))
        llm = ScriptedLLM("s", "s", "s")
        summarize(ChunkedSummarizer(llm, chunk_tokens=10), "abcdefghijklmnopqrstuvwxy")
        assert [user.rsplit("\n", 1)[1] for user in llm.users()] == ["abcdefghij", "klmnopqrst", "uvwxy"]

    def test_the_separators_can_be_chosen(self, summarize):
        llm = ScriptedLLM("s1", "s2")
        worker = ChunkedSummarizer(llm, chunk_tokens=10, separators=["|"])
        assert summarize(worker, "a b c d e f|g h i j k l") == "s1\n\ns2"
        assert worker.separators == ("|",)
        assert llm.users()[0].endswith("Conversation to summarize:\na b c d e f|")
        assert llm.users()[1].endswith("Conversation to summarize:\ng h i j k l")

    def test_by_default_it_splits_at_paragraphs_lines_and_words(self):
        assert ChunkedSummarizer(ScriptedLLM(), chunk_tokens=10).separators == ("\n\n", "\n", " ")

    @pytest.mark.parametrize("separators, error, message", [
        ("\n", TypeError, "separators must be a tuple or list of strings, got str"),
        (("\n", 1), TypeError, "a separator must be a string, got int"),
        (("\n", ""), ValueError, "a separator must not be empty"),
    ])
    def test_invalid_separators_are_refused(self, separators, error, message):
        with pytest.raises(error, match=message):
            ChunkedSummarizer(ScriptedLLM(), chunk_tokens=10, separators=separators)

    def test_a_text_exactly_chunk_tokens_long_fits(self, summarize):
        llm = ScriptedLLM()
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10), words(10)) == words(10)
        assert llm.calls == []

    def test_empty_pieces_and_blank_edges_make_no_chunk(self, summarize):
        llm = ScriptedLLM("s1 s1", "s2 s2")
        text = f"\n\n{paragraphs(4, 4)}\n\n\n\n{paragraphs(4, 4, names='cd')}\n\n  \n\n"
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), text) == "s1 s1\n\ns2 s2"
        assert [user.split("Conversation to summarize:\n")[1] for user in llm.users()] == [
            paragraphs(4, 4), paragraphs(4, 4, names="cd")]

    def test_a_chunk_exactly_at_the_aim_is_kept(self, summarize):
        llm = ScriptedLLM("s1")
        # chunk_aim 0.5 of 10 is 5: the 5-word paragraph is kept, the 9-word one summarized.
        result = summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(5, 9))
        assert result == f"{words(5, 'a')}\n\ns1" and len(llm.calls) == 1

    def test_a_chunk_summary_is_trimmed(self, summarize):
        llm = ScriptedLLM("  s1 \n", "\ns2  ")
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(4, 4, 4, 4)) == "s1\n\ns2"

    def test_a_failed_chunk_call_fails_the_whole_summary(self, summarize):
        llm = ScriptedLLM("s1", ("half a summ", "incomplete"))
        with pytest.raises(ValueError, match="'incomplete', not 'completed'"):
            summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(4, 4, 4, 4))

    def test_a_chunk_already_short_is_kept(self, summarize):
        llm = ScriptedLLM("s1", "s2")
        result = summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(9, 2, 9))
        assert result == f"s1\n\n{words(2, 'b')}\n\ns2"
        assert len(llm.calls) == 2

    def test_each_side_calls_its_own_client(self, summarize):
        llm = ScriptedLLM("s1 s1", "s2 s2")
        summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(4, 4, 4, 4))
        assert llm.modes == [summarize.name] * 2

    def test_async_summarizes_the_chunks_of_a_round_at_the_same_time(self):
        llm = SlowLLM("s1", "s2", "s3")
        worker = ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5)
        assert asyncio.run(worker.summarize_async(paragraphs(8, 8, 8))) == "s1\n\ns2\n\ns3"
        assert llm.most_waiting == 3

    def test_a_round_that_does_not_shrink_returns_the_text_as_it_was(self, summarize):
        llm = ScriptedLLM(words(6, "s"), words(6, "z"))
        text = paragraphs(6, 6)
        assert summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), text) == text
        assert len(llm.calls) == 2

    def test_a_round_that_barely_shrinks_is_the_last_and_its_text_is_returned(self, summarize):
        # 12 words -> 11: less than the default 10% shorter, so no other round, though 11 > 10.
        llm = ScriptedLLM(words(5, "s"), words(6, "z"))
        result = summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5), paragraphs(6, 6))
        assert result == f"{words(5, 's')}\n\n{words(6, 'z')}"
        assert len(llm.calls) == 2

    def test_with_min_decrease_0_it_goes_on_while_the_text_shrinks(self, summarize):
        # Round 2: the 5-word summary is within the aim and kept; the 6-word one is summarized again.
        llm = ScriptedLLM(words(5, "s"), words(6, "z"), words(4, "y"))
        worker = ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5, min_decrease=0)
        assert summarize(worker, paragraphs(6, 6)) == f"{words(5, 's')}\n\n{words(4, 'y')}"
        assert len(llm.calls) == 3

    def test_the_hub_prompt_can_be_chosen(self, summarize):
        llm = ScriptedLLM("s1", "s2")
        summarize(ChunkedSummarizer(llm, chunk_tokens=10, chunk_aim=0.5, prompt_version="v2"), paragraphs(4, 4, 4, 4))
        assert llm.systems()[0].startswith(hub.get("memory/summarize", "v2"))

    @pytest.mark.parametrize("arguments, message", [
        ({"chunk_tokens": 0}, "chunk_tokens must be a positive integer"),
        ({"chunk_tokens": 10, "chunk_aim": 0}, r"chunk_aim must be a number in \(0, 1\]"),
        ({"chunk_tokens": 10, "chunk_aim": 1.5}, r"chunk_aim must be a number in \(0, 1\]"),
        ({"chunk_tokens": 10, "min_decrease": 1}, r"min_decrease must be a number in \[0, 1\)"),
        ({"chunk_tokens": 10, "encoding": "nope"}, "unknown encoding"),
    ])
    def test_invalid_settings_are_refused(self, arguments, message):
        with pytest.raises(ValueError, match=message):
            ChunkedSummarizer(ScriptedLLM(), **arguments)

    @pytest.mark.parametrize("value", [True, "0.5", None])
    def test_a_share_must_be_a_number(self, value):
        with pytest.raises(ValueError, match="chunk_aim must be a number"):
            ChunkedSummarizer(ScriptedLLM(), chunk_tokens=10, chunk_aim=value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="min_decrease must be a number"):
            ChunkedSummarizer(ScriptedLLM(), chunk_tokens=10, min_decrease=value)  # type: ignore[arg-type]

    def test_the_boundary_shares_are_accepted(self):
        worker = ChunkedSummarizer(ScriptedLLM(), chunk_tokens=10, chunk_aim=1, min_decrease=0)
        assert (worker.chunk_aim, worker.min_decrease) == (1, 0)


class TestBudgetSummarizer:
    def test_a_text_within_the_budget_passes_on_with_no_call(self, summarize):
        llm = ScriptedLLM()
        assert summarize(BudgetSummarizer(llm, max_tokens=10), words(10)) == words(10)
        assert llm.calls == []

    def test_one_round_that_fits(self, summarize):
        llm = ScriptedLLM(words(8, "s"))
        assert summarize(BudgetSummarizer(llm, max_tokens=10), words(20)) == words(8, "s")
        assert llm.systems() == [f"{SHORTEN} Aim for about 5 words; the summary must be at most 10 tokens."]
        assert llm.users() == [
            f"This summary is 20 tokens. Shorten it to about 5 words (hard limit: 10 tokens):\n\n{words(20)}"]

    def test_after_an_overshoot_the_next_round_asks_for_less(self, summarize):
        llm = ScriptedLLM(words(15, "s"), words(9, "z"))
        assert summarize(BudgetSummarizer(llm, max_tokens=10), words(20)) == words(9, "z")
        # 7.5 tokens aimed, 15 reached: the next round aims at 7.5 * 10 / 15 = 5 tokens, about 3 words.
        assert "about 3 words" in llm.systems()[1]
        assert llm.users()[1].startswith("This summary is 15 tokens.")

    def test_after_max_rounds_the_shortest_is_kept(self, summarize):
        llm = ScriptedLLM(words(18, "s"), words(16, "z"))
        assert summarize(BudgetSummarizer(llm, max_tokens=10, max_rounds=2), words(20)) == words(16, "z")

    def test_a_round_that_barely_shortens_stops(self, summarize):
        llm = ScriptedLLM(words(19, "s"))
        assert summarize(BudgetSummarizer(llm, max_tokens=10), words(20)) == words(19, "s")
        assert len(llm.calls) == 1

    def test_a_longer_answer_is_not_kept(self, summarize):
        llm = ScriptedLLM(words(25, "s"))
        assert summarize(BudgetSummarizer(llm, max_tokens=10), words(20)) == words(20)

    def test_each_side_calls_its_own_client(self, summarize):
        llm = ScriptedLLM(words(15, "s"), words(9, "z"))
        summarize(BudgetSummarizer(llm, max_tokens=10), words(20))
        assert llm.modes == [summarize.name] * 2

    def test_the_hub_prompt_version_can_be_chosen(self, summarize):
        llm = ScriptedLLM("s")
        summarize(BudgetSummarizer(llm, max_tokens=10, prompt_version="v2"), words(20))
        assert llm.systems()[0].startswith(hub.get("memory/shorten", "v2"))

    @pytest.mark.parametrize("arguments, message", [
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"max_tokens": 10, "max_rounds": 0}, "max_rounds must be a positive integer"),
        ({"max_tokens": 10, "aim": 0}, r"aim must be a number in \(0, 1\]"),
        ({"max_tokens": 10, "min_decrease": -0.1}, r"min_decrease must be a number in \[0, 1\)"),
        ({"max_tokens": 10, "encoding": "nope"}, "unknown encoding"),
    ])
    def test_invalid_settings_are_refused(self, arguments, message):
        with pytest.raises(ValueError, match=message):
            BudgetSummarizer(ScriptedLLM(), **arguments)
