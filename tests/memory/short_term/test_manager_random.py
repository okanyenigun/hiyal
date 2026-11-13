"""Random sessions against ShortTermMemoryManager.

Each session adds random entries under a random policy, with compactions on
demand (with and without a token budget) and policy changes along the way.
After every step:

- the invariants hold: nothing is lost or summarized twice, ``active`` is
  the summary, the notes, then the window, and exactly the latest system
  prompt is pinned;
- a manager resumed from a copy of the storage agrees: the same window,
  summary and notes, with no LLM call at resume. The one difference by
  design: a resume considers only the latest system prompt, so an old one
  still waiting in the live window for the policy to drop it is left out.

User pins are left out: they live only in memory, so a resume does not
restore them.
"""

import asyncio
import random

import pytest

from hiyal.memory.short_term import (
    ContextEntry,
    InMemoryStorage,
    NoteTaker,
    ShortTermMemoryManager,
    RollingCompaction,
    SummaryEntry,
    WorkingNotesEntry,
)
from hiyal.prompts import SystemPrompt, UserPrompt
from hiyal.utility import Tokens

from tests.memory.short_term.note_takers.test_note_taker import FakeLLM, TripNotes, notes_json
from tests.memory.short_term.test_manager import POLICIES, TextSummarizer, answer, filters_keep, result

SEEDS = range(150)
STEPS = 30


@pytest.fixture(autouse=True)
def count_words(monkeypatch):
    """Count words instead of tokens, so no test loads a tiktoken encoding (that needs network)."""
    monkeypatch.setattr(Tokens, "count", staticmethod(lambda text, encoding="o200k_base": len(text.split())))


class EndlessLLM(FakeLLM):
    """Answers every notes update with a new version."""

    def query(self, **parameters):
        self.answers = [notes_json(destination=f"version {len(self.calls) + 1}")]
        return super().query(**parameters)


def settings(policy_name, storage=None, session_id=None):
    return dict(storage=storage or InMemoryStorage(), policy=POLICIES[policy_name](),
                compaction=RollingCompaction(TextSummarizer()),
                note_taker=NoteTaker(EndlessLLM(), schema=TripNotes), notes_every=2, session_id=session_id)


def make(policy_name, storage=None, session_id=None):
    return ShortTermMemoryManager(**settings(policy_name, storage, session_id))


async def make_async(policy_name, storage=None, session_id=None):
    return await ShortTermMemoryManager.open_async(**settings(policy_name, storage, session_id))


def random_entries(rng, several_system_prompts):
    """A system prompt, then random questions, answers, tool calls with results, context and (maybe) system prompts."""
    kinds = ["user", "user", "answer", "answer", "tool", "context"] + (["system"] if several_system_prompts else [])
    entries: list = [SystemPrompt(text="s0")]
    calls = 0
    while len(entries) < STEPS:
        kind = rng.choice(kinds)
        if kind == "user":
            entries.append(UserPrompt(text=f"q{len(entries)} " + "w " * rng.randint(0, 6)))
        elif kind == "answer":
            entries.append(answer(f"a{len(entries)}"))
        elif kind == "tool":
            calls += 1
            entries += [answer(f"t{len(entries)}", calls=[f"m{calls}"]), result(f"m{calls}")]
        elif kind == "context":
            entries.append(ContextEntry(text=f"c{len(entries)}"))
        else:
            entries.append(SystemPrompt(text=f"s{len(entries)}"))
    return entries


def tokens(entries):
    return sum(Tokens.count(entry.to_text()) for entry in entries)


def check_invariants(manager, where):
    added = [e for e in manager.entries if not isinstance(e, (SummaryEntry, WorkingNotesEntry))]
    window = [e.id_ for e in manager.window]
    covers = manager.summary.covered([e.id_ for e in added]) if manager.summary else []
    covered = set(covers)
    if manager.summary:
        assert len(covers) == manager.summary.covers_count, f"{where}: covers_count is off"
    assert not covered & set(window), f"{where}: an entry is both in the window and in the summary"

    leading = [e.id_ for e in (manager.summary, ) if e] + ([e.id_ for e in manager.active[:2]
                                                              if isinstance(e, WorkingNotesEntry)])
    assert [e.id_ for e in manager.active] == leading + window, f"{where}: active is not summary, notes, window"

    pinned = [e for e in manager.pinned if not isinstance(e, (SummaryEntry, WorkingNotesEntry))]
    assert {e.id_ for e in pinned} <= set(window), f"{where}: a pinned entry is not in the window"
    systems = [e.id_ for e in added if isinstance(e, SystemPrompt)]
    assert [e.id_ for e in pinned if isinstance(e, SystemPrompt)] == systems[-1:], \
        f"{where}: the pinned system prompt is not exactly the latest one"

    excluded = {e.id_ for e in added} - {e.id_ for e in filters_keep(manager.policy, added)}
    replaced = set(systems[:-1])
    lost = [e.text for e in added if e.id_ not in set(window) | covered | excluded | replaced]
    assert not lost, f"{where}: neither in the window, the summary, nor excluded by a filter: {lost}"
    if manager.summary:
        assert set(manager.summary.covers_except) <= set(window) | excluded | replaced, \
            f"{where}: covers_except lists an entry that is neither in the window, excluded, nor replaced"


def storage_copy(manager):
    """A copy of the session's storage: if a resume summarizes anything, the live session stays as it was."""
    copy = InMemoryStorage()
    for entry in manager.entries:
        copy.add(manager.session_id, entry)
    return copy


def check_resume(manager, again, where):
    latest = [e.id_ for e in manager.entries if isinstance(e, SystemPrompt)][-1]
    live = [e.id_ for e in manager.window if not isinstance(e, SystemPrompt) or e.id_ == latest]  # see the docstring
    assert [e.id_ for e in again.window] == live, (
        f"{where}: the resumed window differs\n  live:    {[label(e) for e in manager.window]}\n"
        f"  resumed: {[label(e) for e in again.window]}"
    )
    summary_id = manager.summary.id_ if manager.summary else None
    assert (again.summary.id_ if again.summary else None) == summary_id, f"{where}: the resume made a new summary"
    assert again.notes == manager.notes, f"{where}: the resumed notes differ"


def label(entry):
    return getattr(entry, "text", None) or type(entry).__name__


def budget(rng, manager):
    """A random token budget between what the pinned entries take and what the window takes."""
    pinned = tokens(e for e in manager.pinned if not isinstance(e, (SummaryEntry, WorkingNotesEntry)))
    return rng.randint(pinned, max(pinned, tokens(manager.window)))


def run_session(seed, several_system_prompts, resume):
    rng = random.Random(seed)
    policy_name = rng.choice(list(POLICIES))
    manager = make(policy_name)
    for step, entry in enumerate(random_entries(rng, several_system_prompts)):
        manager.add(entry)
        roll = rng.random()
        if roll < 0.08:
            manager.compact()
        elif roll < 0.2:
            manager.compact(max_tokens=budget(rng, manager))
        elif roll < 0.25:
            policy_name = rng.choice(list(POLICIES))
            manager.policy = POLICIES[policy_name]()
        where = f"seed {seed}, step {step}, policy {policy_name}"
        check_invariants(manager, where)
        if resume:
            check_resume(manager, make(policy_name, storage_copy(manager), manager.session_id), where)


async def run_session_async(seed, several_system_prompts, resume):
    """``run_session`` through the async API, in one event loop.

    The checks read the in-memory storage with the sync properties, which
    needs no loop; the async reads are covered in ``test_manager``.
    """
    rng = random.Random(seed)
    policy_name = rng.choice(list(POLICIES))
    manager = await make_async(policy_name)
    for step, entry in enumerate(random_entries(rng, several_system_prompts)):
        await manager.add_async(entry)
        roll = rng.random()
        if roll < 0.08:
            await manager.compact_async()
        elif roll < 0.2:
            await manager.compact_async(max_tokens=budget(rng, manager))
        elif roll < 0.25:
            policy_name = rng.choice(list(POLICIES))
            await manager.set_policy_async(POLICIES[policy_name]())
        where = f"seed {seed}, step {step}, policy {policy_name} (async)"
        check_invariants(manager, where)
        if resume:
            check_resume(manager, await make_async(policy_name, storage_copy(manager), manager.session_id), where)


def run(seed, several_system_prompts, resume, asynchronous):
    if asynchronous:
        asyncio.run(run_session_async(seed, several_system_prompts, resume))
    else:
        run_session(seed, several_system_prompts, resume)


SYSTEM_PROMPTS = pytest.mark.parametrize(
    "several_system_prompts", [False, True], ids=["one-system-prompt", "several-system-prompts"])
API = pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])


@API
@SYSTEM_PROMPTS
def test_the_invariants_hold_after_every_step(several_system_prompts, asynchronous):
    for seed in SEEDS:
        run(seed, several_system_prompts, resume=False, asynchronous=asynchronous)


@API
@SYSTEM_PROMPTS
def test_a_resumed_manager_agrees_after_every_step(several_system_prompts, asynchronous):
    for seed in SEEDS:
        run(seed, several_system_prompts, resume=True, asynchronous=asynchronous)
