import json
from datetime import datetime, timezone

import pytest

from hiyal.memory.short_term import (
    ContextEntry,
    FileStorage,
    SummaryEntry,
    ToolResultEntry,
    WorkingNotesEntry,
)
from hiyal.models.schemas import LLMResponse, ResponseContent, ResponseMessage
from hiyal.prompts import Prompt, Image, SystemPrompt, UserPrompt

S = "session-1"
PNG = b"\x89PNG\r\n\x1a\n\x00\xff\xfe\xfb"


def response(**overrides):
    values = {"text": "Hi.", "messages": [], "provider": "openai", "id_": "resp-1", "model": "m",
              "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "status": "completed", "raw": {"sdk": "object"}}
    return LLMResponse(**{**values, **overrides})


def every_entry():
    return [
        UserPrompt(text="Hi", images=[Image(data=PNG, mime_type="image/png")]),
        SystemPrompt(text="Be brief."),
        response(),
        ContextEntry(data={"budget": 3000}, source="long_term_memory"),
        ToolResultEntry(message_id="item_1", call_id="call_1", name="get_weather", arguments={"city": "Kyoto"},
                        output={"c": 21}),
        SummaryEntry(text="Trip to Japan.", covers_through="a", covers_count=1),
        WorkingNotesEntry(notes={"objective": "Compare", "decisions": ["Exclude large-scale"]}),
    ]


def dumped(entries):
    """Comparable form: LLMResponse.raw is not stored, so compare dumps rather than objects."""
    return [entry.model_dump() for entry in entries]


@pytest.fixture(params=[".jsonl", ".txt", ".md"])
def path(request, tmp_path):
    return tmp_path / "nested" / f"history{request.param}"


@pytest.fixture
def storage(path):
    return FileStorage(path)


class TestFileStorage:
    def test_missing_file_reads_as_empty_and_is_not_created(self, storage, path):
        assert storage.entries(S) == [] and not path.exists()

    def test_every_entry_type_round_trips_in_order(self, storage):
        entries = every_entry()
        for entry in entries:
            storage.add(S, entry)
        reloaded = storage.entries(S)
        assert [type(entry) for entry in reloaded] == [type(entry) for entry in entries]
        assert dumped(reloaded) == dumped(entries)

    def test_entries_survive_a_new_storage_object(self, storage, path):
        entries = every_entry()
        for entry in entries:
            storage.add(S, entry)
        reopened = FileStorage(path, exist_ok=True)
        assert dumped(reopened.entries(S)) == dumped(entries)

    def test_tricky_text_survives(self, storage):
        # Text that looks like each format's markers, code fences, and the
        # "-->" that would close a Markdown comment.
        tricky = "@entry {}\n<!-- hiyal-entry {} -->\n```json\n{}\n```\na --> b\n\n> quoted"
        storage.add(S, ContextEntry(text=tricky))
        assert storage.entries(S)[0].text == tricky

    def test_media_bytes_survive(self, storage):
        storage.add(S, every_entry()[0])
        assert storage.entries(S)[0].images[0].data == PNG

    def test_response_image_output_survives(self, storage):
        image = ResponseMessage(id="m1", role="assistant", content=ResponseContent(
            content_type="image", data={"mime_type": "image/png", "data": PNG}))
        storage.add(S, response(messages=[image]))
        assert storage.entries(S)[0].images[0].data == PNG

    def test_remove_keeps_the_rest_in_order(self, storage):
        a, b, c = ContextEntry(text="a"), response(), ContextEntry(text="c")
        for entry in (a, b, c):
            storage.add(S, entry)
        storage.remove(S, b.id_)
        assert [entry.id_ for entry in storage.entries(S)] == [a.id_, c.id_]

    def test_removing_an_entry_that_is_not_stored_raises(self, storage):
        storage.add(S, ContextEntry(text="a"))
        with pytest.raises(ValueError, match="is not stored"):
            storage.remove(S, "no-such-id")

    def test_clear_empties_the_file(self, storage, path):
        for entry in every_entry():
            storage.add(S, entry)
        storage.clear(S)
        assert storage.entries(S) == [] and path.read_text() == ""

    def test_no_temporary_file_is_left_behind(self, storage, path):
        storage.add(S, ContextEntry(text="a"))
        storage.remove(S, storage.entries(S)[0].id_)
        assert [p.name for p in path.parent.iterdir()] == [path.name]


class TestWorkingNotes:
    def test_notes_round_trip_as_plain_data(self, storage, path):
        notes = WorkingNotesEntry(notes={"objective": "Compare", "decisions": ["Exclude large-scale"]})
        storage.add(S, notes)
        assert FileStorage(path, exist_ok=True).entries(S) == [notes]


class TestUnknownEntries:
    def test_a_non_entry_cannot_be_stored(self, storage):
        with pytest.raises(TypeError, match="cannot store"):
            storage.add(S, Prompt(text="the base prompt is not an entry"))

    def test_an_unknown_type_in_the_file_raises(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text('{"session": "session-1", "type": "Mystery", "entry": {}}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="unknown entry type 'Mystery'"):
            FileStorage(path, exist_ok=True).entries(S)


class TestExistingFiles:
    def test_an_existing_file_is_refused_by_default(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(FileExistsError, match="exist_ok=True"):
            FileStorage(path)

    def test_exist_ok_continues_the_file(self, tmp_path):
        path = tmp_path / "history.md"
        first = ContextEntry(text="first")
        FileStorage(path).add(S, first)
        reopened = FileStorage(path, exist_ok=True)
        reopened.add(S, ContextEntry(text="second"))
        assert [entry.text for entry in reopened.entries(S)] == ["first", "second"]

    def test_a_missing_file_is_fine_either_way(self, tmp_path):
        assert FileStorage(tmp_path / "a.txt").entries(S) == []
        assert FileStorage(tmp_path / "b.txt", exist_ok=True).entries(S) == []


class TestFormats:
    @pytest.mark.parametrize("name", ["history.json", "history.csv", "history"])
    def test_unsupported_extensions_are_refused(self, tmp_path, name):
        with pytest.raises(ValueError, match="unsupported file extension"):
            FileStorage(tmp_path / name)

    def test_extension_case_does_not_matter(self, tmp_path):
        storage = FileStorage(tmp_path / "HISTORY.MD")
        storage.add(S, ContextEntry(text="a"))
        assert storage.entries(S)[0].text == "a"

    def test_jsonl_has_one_record_per_line(self, tmp_path):
        path = tmp_path / "history.jsonl"
        storage = FileStorage(path)
        storage.add(S, ContextEntry(text="a"))
        storage.add(S, response())
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["type"] for line in lines] == ["ContextEntry", "LLMResponse"]

    @pytest.mark.parametrize("name", ["history.txt", "history.md"])
    def test_readable_formats_show_each_entrys_text(self, tmp_path, name):
        storage = FileStorage(tmp_path / name)
        storage.add(S, SummaryEntry(text="Trip to Japan in April.", covers_through="a", covers_count=1))
        content = (tmp_path / name).read_text(encoding="utf-8")
        assert "Summary of the earlier conversation:" in content and "Trip to Japan in April." in content

    def test_markdown_hides_the_record_in_a_comment(self, tmp_path):
        path = tmp_path / "history.md"
        FileStorage(path).add(S, ContextEntry(text="a > b"))
        record_lines = [line for line in path.read_text().splitlines() if line.startswith("<!--")]
        assert len(record_lines) == 1 and ">" not in record_lines[0][4:-3]


class TestToolCalls:
    def test_response_tool_calls_survive_storage(self, storage):
        from hiyal.models.schemas import ToolCallContent
        from hiyal.tools import ToolCall

        call = ToolCall(raw={"sdk": "object"}, name="get_weather", arguments={"city": "Kyoto"}, status="complete")
        message = ResponseMessage(id="m1", role="assistant", content=ToolCallContent(data=call))
        storage.add(S, response(messages=[message]))
        reloaded = storage.entries(S)[0]
        assert [(c.name, c.arguments) for c in reloaded.tool_calls] == [("get_weather", {"city": "Kyoto"})]
