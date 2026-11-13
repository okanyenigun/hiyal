import pytest

from hiyal.memory.short_term import Summarizer
from hiyal.prompts.hub import PromptHub, hub


@pytest.fixture
def library(tmp_path):
    (tmp_path / "memory" / "summarize").mkdir(parents=True)
    (tmp_path / "memory" / "summarize" / "v1.md").write_text("First version.\n")
    (tmp_path / "memory" / "summarize" / "v2.md").write_text("Second version.\n")
    return PromptHub(tmp_path)


class TestPromptHub:
    def test_get_reads_a_version_without_surrounding_whitespace(self, library):
        assert library.get("memory/summarize", "v2") == "Second version."

    def test_versions_are_listed_sorted(self, library):
        assert library.versions("memory/summarize") == ["v1", "v2"]
        assert library.versions("nothing") == []

    def test_an_unknown_prompt_or_version_says_what_exists(self, library):
        with pytest.raises(ValueError, match="there is no prompt 'memory/other'"):
            library.get("memory/other", "v1")
        with pytest.raises(ValueError, match="has no version 'v9'; it has v1, v2"):
            library.get("memory/summarize", "v9")

    def test_the_shipped_summarize_prompt_is_there(self):
        assert hub.get("memory/summarize", "v1").startswith("You summarize a conversation")

    def test_summarize_v2_asks_for_fixed_headings(self):
        v2 = hub.get("memory/summarize", "v2")
        headings = ["LANGUAGE:", "INTENT:", "SUMMARY:", "TOOLS:", "ARTIFACTS:", "NEXT STEPS:"]
        assert [v2.index(h) for h in headings] == sorted(v2.index(h) for h in headings)
        assert 'Write "None"' in v2 and Summarizer(llm=None, prompt_version="v2").instructions == v2


class TestSummarizerUsesTheHub:
    def test_by_default_the_hub_prompt(self):
        assert Summarizer(llm=None).instructions == hub.get("memory/summarize", "v1")

    def test_another_version_or_given_instructions(self, monkeypatch, library):
        monkeypatch.setattr("hiyal.memory.short_term.summarizers.base.hub", library)
        assert Summarizer(llm=None, prompt_version="v2").instructions == "Second version."
        assert Summarizer(llm=None, instructions="Mine.").instructions == "Mine."
