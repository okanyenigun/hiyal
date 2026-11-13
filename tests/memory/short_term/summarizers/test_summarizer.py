import asyncio
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import ContextEntry, Summarizer, SummaryEntry
from hiyal.memory.short_term.summarizers import BaseSummarizer
from hiyal.models.schemas import LLMResponse
from hiyal.prompts import Image, SystemPrompt, UserPrompt
from hiyal.prompts.hub import hub
from hiyal.utility import Tokens

PNG = b"\x89PNG\r\n\x1a\n"
FOCUS = "choosing how to travel between cities"
FOCUS_SENTENCE = (
    f" Focus: {FOCUS}. Keep what serves this focus, plus anything that limits it "
    "(for example a budget), and leave out the rest."
)


DEFAULT_INSTRUCTIONS = hub.get("memory/summarize", "v1")
COMBINE_INSTRUCTIONS = hub.get("memory/combine", "v1")
SHORTEN_INSTRUCTIONS = hub.get("memory/shorten", "v1")


@pytest.fixture(autouse=True)
def count_words(monkeypatch):
    """Count words instead of tokens, so no test loads a tiktoken encoding (that needs network)."""
    monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))


@pytest.fixture(autouse=True, params=["sync", "async"])
def mode(request, monkeypatch):
    """Run every test twice: through ``summarize``, then through ``summarize_async``.

    The async run must make the same calls, in the same order, and return the
    same summary: the scripted LLM answers without waiting, so calls run at
    the same time still reach it in order.
    """
    if request.param == "async":
        monkeypatch.setattr(Summarizer, "summarize",
                            lambda self, entries, previous=None: asyncio.run(self.summarize_async(entries, previous)))
    return request.param


class ScriptedLLM:
    """Answers each query with the next answer and records the calls.

    An answer is a text, or ``(text, status)`` for a response that is not completed.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def query(self, **parameters):
        self.calls.append(parameters)
        answer = self.answers[len(self.calls) - 1]
        text, status = answer if isinstance(answer, tuple) else (answer, "completed")
        return LLMResponse(text=text, messages=[], provider="openai", id_=f"resp-{len(self.calls)}", model="m",
                           created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status=status)

    async def query_async(self, **parameters):
        return self.query(**parameters)


class FakeLLM(ScriptedLLM):
    """Answers every query with the same summary."""

    def __init__(self, answer="The summary."):
        super().__init__()
        self.answer = answer

    def query(self, **parameters):
        self.answers.append(self.answer)
        return super().query(**parameters)


def conversation():
    question = UserPrompt(text="Plan a Japan trip.", images=[
        Image(data=PNG, mime_type="image/png", filename="map.png")])
    reply = LLMResponse(text="April is best.", messages=[], provider="openai", id_="r1", model="m",
                        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), status="completed")
    return [question, reply, ContextEntry(text="Prefers direct flights.", source="long_term_memory")]


RENDERED = (
    "Conversation to summarize:\n"
    "User: Plan a Japan trip.\n[image: map.png (image/png)]\n\n"
    "Assistant: April is best.\n\n"
    "Context (long_term_memory):\nPrefers direct flights."
)


def words(count):
    """Text that the patched counter counts as ``count`` tokens."""
    return " ".join(["word"] * count)


def notes(*sizes):
    """Context entries whose text counts as ``size`` + 1 tokens ("Context:" plus the words)."""
    return [ContextEntry(text=f"n{index} " + words(size - 1)) for index, size in enumerate(sizes)]


def systems(llm):
    return [call["system_prompt"].text for call in llm.calls]


class TestRender:
    def test_entries_become_their_text_with_media_as_placeholders(self):
        prompt = Summarizer(FakeLLM()).render(conversation())
        assert prompt.text == RENDERED and not prompt.has_media

    def test_a_previous_summary_comes_first(self):
        previous = SummaryEntry(text="Trip planning started.", covers=["x"])
        prompt = Summarizer(FakeLLM()).render(conversation(), previous)
        assert prompt.text == f"Previous summary:\nTrip planning started.\n\n{RENDERED}"

    def test_media_descriptions_are_in_the_text_and_no_media_is_sent(self):
        photo = UserPrompt(text="Our hotel.", images=[
            Image(data=PNG, mime_type="image/png", filename="booking.png", description="Hotel Sakura, 4-8 April")])
        prompt = Summarizer(FakeLLM()).render([photo])
        assert "[image: booking.png (image/png): Hotel Sakura, 4-8 April]" in prompt.text
        assert not prompt.has_media


class TestPlain:
    def test_with_no_options_one_call_with_the_instructions_and_the_rendered_entries(self):
        llm = FakeLLM()
        response = Summarizer(llm).summarize(conversation())
        assert response.text == "The summary." and len(llm.calls) == 1
        call = llm.calls[0]
        assert call["system_prompt"] == SystemPrompt(text=DEFAULT_INSTRUCTIONS, id_=call["system_prompt"].id_)
        assert call["user_prompt"].text == RENDERED

    def test_custom_instructions(self):
        llm = FakeLLM()
        Summarizer(llm, "One sentence.").summarize(conversation())
        assert llm.calls[0]["system_prompt"].text == "One sentence."

    def test_nothing_to_summarize(self):
        llm = FakeLLM()
        with pytest.raises(ValueError, match="no entries"):
            Summarizer(llm).summarize([])
        assert llm.calls == []


class TestFocus:
    def test_goes_into_the_system_prompt_not_the_text(self):
        llm = FakeLLM()
        summarizer = Summarizer(llm, focus=f"  {FOCUS}  ")
        summarizer.summarize(conversation())
        assert summarizer.focus == FOCUS
        assert llm.calls[0]["system_prompt"].text == DEFAULT_INSTRUCTIONS + FOCUS_SENTENCE
        assert llm.calls[0]["user_prompt"].text == RENDERED

    def test_can_be_changed_and_switched_off(self):
        llm = FakeLLM()
        summarizer = Summarizer(llm, focus=FOCUS)
        summarizer.focus = None
        summarizer.summarize(conversation())
        assert llm.calls[0]["system_prompt"].text == DEFAULT_INSTRUCTIONS

    @pytest.mark.parametrize("focus", ["", "   ", 7])
    def test_must_be_none_or_a_non_blank_string(self, focus):
        with pytest.raises(ValueError, match="focus must be None or a non-blank string"):
            Summarizer(FakeLLM(), focus=focus)


class TestBudget:
    def test_one_call_with_the_aim_when_it_fits(self):
        llm = ScriptedLLM(words(10))
        response = Summarizer(llm, max_tokens=20).summarize(conversation())
        assert response.text == words(10) and len(llm.calls) == 1
        # Aim 75% of 20 = 15 tokens, about 11 words.
        assert systems(llm) == [f"{DEFAULT_INSTRUCTIONS} Aim for about 11 words; the summary must be at most 20 tokens."]

    def test_shortens_until_it_fits_aiming_lower_each_time(self):
        llm = ScriptedLLM(words(30), words(12))
        response = Summarizer(llm, max_tokens=20).summarize(conversation())
        assert response.text == words(12) and len(llm.calls) == 2
        # Overshot 30/20: aim 15 * 20/30 = 10 tokens, about 7 words.
        assert systems(llm)[1] == f"{SHORTEN_INSTRUCTIONS} Aim for about 7 words; the summary must be at most 20 tokens."
        assert llm.calls[1]["user_prompt"].text.startswith(
            "This summary is 30 tokens. Shorten it to about 7 words (hard limit: 20 tokens):\n\nword")

    def test_the_focus_is_kept_when_shortening(self):
        llm = ScriptedLLM(words(30), words(12))
        Summarizer(llm, focus=FOCUS, max_tokens=20).summarize(conversation())
        assert systems(llm)[1].startswith(SHORTEN_INSTRUCTIONS + FOCUS_SENTENCE)

    def test_stops_when_a_round_stalls_and_returns_the_shortest(self):
        llm = ScriptedLLM(words(30), words(29))
        response = Summarizer(llm, max_tokens=20).summarize(conversation())
        assert response.text == words(29) and len(llm.calls) == 2

    def test_stops_after_max_rounds(self):
        llm = ScriptedLLM(words(80), words(60), words(40))
        response = Summarizer(llm, max_tokens=20, max_rounds=2).summarize(conversation())
        assert response.text == words(40) and len(llm.calls) == 3


class TestChunking:
    def test_one_call_without_a_target_when_everything_fits(self):
        llm, entries = ScriptedLLM("The summary."), notes(2, 2)
        summarizer = Summarizer(llm, chunk_tokens=10)
        summarizer.summarize(entries)
        assert systems(llm) == [DEFAULT_INSTRUCTIONS]
        assert llm.calls[0]["user_prompt"].text == summarizer.render(entries).text

    def test_chunks_aim_at_a_quarter_of_chunk_tokens_and_the_final_combine_has_no_target(self):
        # 16 + 16 fit 40; the third goes alone: 2 chunks, then one combine.
        llm = ScriptedLLM(words(5), words(5), "Combined.")
        response = Summarizer(llm, chunk_tokens=40).summarize(notes(15, 15, 15))
        assert response.text == "Combined." and len(llm.calls) == 3
        # A quarter of 40 = 10 tokens, about 7 words.
        assert systems(llm) == [f"{DEFAULT_INSTRUCTIONS} Aim for about 7 words."] * 2 + [COMBINE_INSTRUCTIONS]
        assert llm.calls[0]["user_prompt"].text.startswith("Part 1 of 2 of a longer conversation")
        assert llm.calls[2]["user_prompt"].text == (
            f"Summaries of consecutive parts of the conversation, oldest first:\n\nPart 1:\n{words(5)}\n\nPart 2:\n{words(5)}"
        )

    def test_the_previous_summary_is_the_oldest_part(self):
        llm = ScriptedLLM(words(5), words(5), "Combined.")
        previous = SummaryEntry(text="Earlier.", covers=["x"])
        Summarizer(llm, chunk_tokens=40).summarize(notes(15, 15, 15), previous)
        assert llm.calls[2]["user_prompt"].text.split("\n\n")[1] == "Part 1:\nEarlier."

    def test_combines_level_by_level_with_intermediate_targets(self):
        # 3 chunks of 15 each: 15 + 15 fit 40, the third passes through; then the final combine.
        llm = ScriptedLLM(words(15), words(15), words(15), words(10), "Combined.")
        response = Summarizer(llm, chunk_tokens=40).summarize(notes(30, 30, 30))
        assert response.text == "Combined." and len(llm.calls) == 5
        assert systems(llm)[3] == f"{COMBINE_INSTRUCTIONS} Aim for about 7 words."
        assert systems(llm)[4] == COMBINE_INSTRUCTIONS

    def test_shortens_the_pieces_when_no_two_fit_one_combine(self):
        # 30 + 30 is over 40: each is shortened, aiming at 10 * 10/30 = 3.3 tokens (2 words), then combined.
        llm = ScriptedLLM(words(30), words(30), words(8), words(8), "Combined.")
        response = Summarizer(llm, chunk_tokens=40).summarize(notes(30, 30))
        assert response.text == "Combined." and len(llm.calls) == 5
        assert systems(llm)[2:4] == [f"{SHORTEN_INSTRUCTIONS} Aim for about 2 words."] * 2
        assert llm.calls[2]["user_prompt"].text == f"This summary is 30 tokens. Shorten it to about 2 words:\n\n{words(30)}"
        assert systems(llm)[4] == COMBINE_INSTRUCTIONS

    def test_a_piece_within_the_target_is_not_shortened(self):
        # 30 + 8 is over 40 only through the first: only it is shortened.
        llm = ScriptedLLM(words(35), words(8), words(8), "Combined.")
        Summarizer(llm, chunk_tokens=40).summarize(notes(30, 30))
        assert systems(llm)[2].startswith(SHORTEN_INSTRUCTIONS) and systems(llm)[3] == COMBINE_INSTRUCTIONS
        assert len(llm.calls) == 4

    def test_raises_when_shortening_stalls(self):
        llm = ScriptedLLM(words(30), words(30), words(29), words(29))
        with pytest.raises(ValueError, match="too long to combine within chunk_tokens=40, and shortening them stalled"):
            Summarizer(llm, chunk_tokens=40).summarize(notes(30, 30))


class TestChunkingWithBudget:
    def test_intermediate_calls_take_the_smaller_aim_then_the_result_is_fitted(self):
        llm = ScriptedLLM(words(5), words(5), words(25), words(12))
        response = Summarizer(llm, chunk_tokens=40, max_tokens=20, focus=FOCUS).summarize(notes(15, 15, 15))
        assert response.text == words(12) and len(llm.calls) == 4
        limit = "the summary must be at most 20 tokens."
        assert systems(llm) == [
            # Chunks: min(10, 15) = 10 tokens, about 7 words.
            f"{DEFAULT_INSTRUCTIONS}{FOCUS_SENTENCE} Aim for about 7 words; {limit}",
            f"{DEFAULT_INSTRUCTIONS}{FOCUS_SENTENCE} Aim for about 7 words; {limit}",
            # Final combine: the budget's aim, 15 tokens, about 11 words.
            f"{COMBINE_INSTRUCTIONS}{FOCUS_SENTENCE} Aim for about 11 words; {limit}",
            # 25 is over 20: shorten, aiming at 15 * 20/25 = 12 tokens, about 9 words.
            f"{SHORTEN_INSTRUCTIONS}{FOCUS_SENTENCE} Aim for about 9 words; {limit}",
        ]


class TestSettings:
    @pytest.mark.parametrize("settings, message", [
        ({"chunk_tokens": 0}, "chunk_tokens must be a positive integer"),
        ({"chunk_tokens": True}, "chunk_tokens must be a positive integer"),
        ({"max_tokens": -1}, "max_tokens must be a positive integer"),
        ({"max_tokens": 1.5}, "max_tokens must be a positive integer"),
        ({"max_rounds": -1}, "max_rounds must be a non-negative integer"),
        ({"min_decrease": 1}, r"min_decrease must be a number in \[0, 1\)"),
        ({"budget_aim": 0}, r"budget_aim must be a number in \(0, 1\]"),
        ({"chunk_aim": 1.5}, r"chunk_aim must be a number in \(0, 1\]"),
    ])
    def test_invalid_settings(self, settings, message):
        with pytest.raises(ValueError, match=message):
            Summarizer(FakeLLM(), **settings)

    def test_an_unknown_kind_of_call(self):
        with pytest.raises(ValueError, match="kind must be"):
            Summarizer(FakeLLM()).system("translate")

    def test_a_cut_off_response_is_refused(self):
        with pytest.raises(ValueError, match="'incomplete', not 'completed'.*max_output_tokens"):
            Summarizer(ScriptedLLM(("The user plans a trip to", "incomplete"))).summarize(conversation())


def test_a_summarizer_must_say_how_to_render():
    with pytest.raises(TypeError, match="render"):
        BaseSummarizer(FakeLLM())  # type: ignore[abstract]


class TestAsync:
    def test_the_chunks_are_summarized_at_the_same_time(self):
        class SlowLLM(FakeLLM):
            in_flight = most = 0

            async def query_async(self, **parameters):
                SlowLLM.in_flight += 1
                SlowLLM.most = max(SlowLLM.most, SlowLLM.in_flight)
                await asyncio.sleep(0.01)
                SlowLLM.in_flight -= 1
                return self.query(**parameters)

        llm = SlowLLM(words(2))
        asyncio.run(Summarizer(llm, chunk_tokens=10).summarize_async(notes(9, 9, 9)))
        assert len(llm.calls) == 4 and SlowLLM.most == 3  # three chunks at once, then one combine

    def test_the_base_class_makes_one_async_call(self):
        class Plain(BaseSummarizer):
            def render(self, entries, previous=None):
                return UserPrompt(text="rendered")

        llm = FakeLLM()
        response = asyncio.run(Plain(llm).summarize_async(conversation()))
        assert response.text == "The summary." and llm.calls[0]["user_prompt"].text == "rendered"
