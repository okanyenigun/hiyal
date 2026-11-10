import re
from datetime import datetime, timedelta, timezone

import pytest

from hiyal.utility import Dicts, Identifiers, Parsing, Timing


class TestDicts:
    def test_returns_the_stored_dict_itself_by_default(self):
        inner = {"a": 1}
        params = {"key": inner}
        assert Dicts.get(params, "key") is inner
        assert params == {"key": inner}

    def test_copy_returns_an_independent_dict(self):
        inner = {"a": 1}
        result = Dicts.get({"key": inner}, "key", copy=True)
        assert result == inner and result is not inner
        result["b"] = 2
        assert inner == {"a": 1}

    def test_missing_key_gives_a_fresh_empty_dict_each_time(self):
        first, second = Dicts.get({}, "key"), Dicts.get({}, "key")
        assert first == {} and first is not second

    def test_pop_removes_the_key(self):
        params = {"key": {"a": 1}, "other": 1}
        assert Dicts.get(params, "key", pop=True) == {"a": 1}
        assert params == {"other": 1}

    def test_pop_of_a_missing_key_leaves_params_unchanged(self):
        params = {"other": 1}
        assert Dicts.get(params, "key", pop=True) == {}
        assert params == {"other": 1}

    @pytest.mark.parametrize("value", ["text", [1], 3, None, ("a",)])
    def test_non_dict_values_raise_type_error(self, value):
        with pytest.raises(TypeError, match="^key must be a dictionary$"):
            Dicts.get({"key": value}, "key")

    def test_owner_and_label_shape_the_message(self):
        with pytest.raises(TypeError, match=r"^Gemini 'http_options.retry' must be a dictionary$"):
            Dicts.get({"key": 1}, "key", owner="Gemini", label="http_options.retry")
        with pytest.raises(TypeError, match=r"^Gemini 'key' must be a dictionary$"):
            Dicts.get({"key": 1}, "key", owner="Gemini")
        with pytest.raises(TypeError, match=r"^custom must be a dictionary$"):
            Dicts.get({"key": 1}, "key", label="custom")

    def test_a_non_dict_is_still_removed_when_popping(self):
        params = {"key": 1}
        with pytest.raises(TypeError):
            Dicts.get(params, "key", pop=True)
        assert params == {}


class TestIdentifiers:
    def test_format_is_prefix_underscore_hex(self):
        assert re.fullmatch(r"item_[0-9a-f]{32}", Identifiers.generate_id("item"))

    def test_ids_are_unique(self):
        assert len({Identifiers.generate_id("x") for _ in range(1000)}) == 1000


class TestParsing:
    @pytest.mark.parametrize("text, expected", [
        ('{"a": 1}', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ("3", 3),
        ('"text"', "text"),
        ("null", None),
        ("true", True),
    ])
    def test_valid_json_is_parsed(self, text, expected):
        assert Parsing.parse_json(text) == expected

    def test_empty_input_gives_an_empty_object(self):
        assert Parsing.parse_json("") == {}

    @pytest.mark.parametrize("text", ['{"a":', "not json", "{'a': 1}", " "])
    def test_invalid_json_is_returned_unchanged(self, text):
        assert Parsing.parse_json(text) == text


class TestTiming:
    @pytest.mark.parametrize("nanoseconds, seconds", [(0, 0.0), (1_500_000_000, 1.5), (1, 1e-9)])
    def test_nanoseconds_to_seconds(self, nanoseconds, seconds):
        assert Timing.nanoseconds_to_seconds(nanoseconds) == pytest.approx(seconds)

    def test_none_duration_stays_none(self):
        assert Timing.nanoseconds_to_seconds(None) is None

    def test_parse_utc_z_timestamp(self):
        assert Timing.parse_datetime("2026-01-02T03:04:05Z") == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def test_parse_offset_timestamp(self):
        parsed = Timing.parse_datetime("2026-01-02T03:04:05+03:00")
        assert parsed is not None and parsed.utcoffset() == timedelta(hours=3)

    def test_parse_nanosecond_precision_as_ollama_sends_it(self):
        parsed = Timing.parse_datetime("2026-01-02T03:04:05.123456789Z")
        assert parsed == datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)

    @pytest.mark.parametrize("value", [None, "", "yesterday", "2026-13-01T00:00:00Z"])
    def test_empty_or_invalid_timestamps_give_none(self, value):
        assert Timing.parse_datetime(value) is None
