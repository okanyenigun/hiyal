import pytest
from pydantic import TypeAdapter, ValidationError

from hiyal.tools import ToolCall, ToolCallDelta, ToolCallIssue

RAW = object()


def normalize(**overrides):
    values = {"name": "get_weather", "arguments": {"city": "Istanbul"}, "raw": RAW, "complete": True}
    values.update(overrides)
    return ToolCall.normalize(**values)


class TestNormalizeNames:
    def test_valid_call_is_complete(self):
        call = normalize()
        assert (call.name, call.arguments, call.status, call.issue) == (
            "get_weather", {"city": "Istanbul"}, "complete", None)
        assert call.raw is RAW

    @pytest.mark.parametrize("name, code", [(None, "missing_name"), ("", "missing_name"),
                                            ("get weather", "invalid_name"), ("get\tweather", "invalid_name"),
                                            (" ", "invalid_name")])
    def test_missing_or_whitespace_names_are_invalid(self, name, code):
        call = normalize(name=name)
        assert (call.name, call.status, call.issue.code) == (None, "invalid", code)
        assert call.arguments == {"city": "Istanbul"}


class TestNormalizeArguments:
    def test_explicit_empty_object_is_valid(self):
        assert normalize(arguments={}).status == "complete"

    def test_missing_arguments_are_invalid(self):
        call = normalize(arguments=None)
        assert (call.arguments, call.status, call.issue.code) == (None, "invalid", "missing_arguments")

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_blank_json_text_counts_as_missing(self, text):
        call = normalize(arguments=text, json_encoded=True)
        assert (call.arguments, call.issue.code) == (None, "missing_arguments")

    def test_json_text_is_decoded(self):
        assert normalize(arguments='{"city": "Paris", "n": [1, 2.5, null]}', json_encoded=True).arguments == {
            "city": "Paris", "n": [1, 2.5, None]}

    @pytest.mark.parametrize("text, code", [
        ('{"city":', "invalid_json"),
        ("not json", "invalid_json"),
        ("[1, 2]", "arguments_not_object"),
        ('"text"', "arguments_not_object"),
        ("3", "arguments_not_object"),
        ('{"x": NaN}', "invalid_arguments"),
    ])
    def test_bad_json_text_is_invalid_without_arguments(self, text, code):
        call = normalize(arguments=text, json_encoded=True)
        assert (call.arguments, call.status, call.issue.code) == (None, "invalid", code)
        assert call.issue.message

    @pytest.mark.parametrize("arguments, code", [
        ([1, 2], "arguments_not_object"),
        ("text", "arguments_not_object"),
        ({"bad": {1, 2}}, "invalid_arguments"),
        ({"bad": float("nan")}, "invalid_arguments"),
    ])
    def test_non_json_python_arguments_are_invalid(self, arguments, code):
        call = normalize(arguments=arguments)
        assert (call.arguments, call.status, call.issue.code) == (None, "invalid", code)


class TestNormalizeStatus:
    def test_incomplete_generation_is_incomplete_even_when_valid(self):
        call = normalize(complete=False)
        assert (call.status, call.issue) == ("incomplete", None)

    def test_incomplete_generation_keeps_its_issue(self):
        call = normalize(complete=False, arguments='{"city":', json_encoded=True)
        assert (call.status, call.issue.code) == ("incomplete", "invalid_json")

    def test_a_given_issue_takes_precedence(self):
        given = ToolCallIssue(code="conflicting_arguments", message="Two sources.")
        call = normalize(name=None, arguments="not json", json_encoded=True, issue=given)
        assert call.issue == given and call.status == "invalid"

    def test_a_given_issue_makes_an_otherwise_valid_call_invalid(self):
        given = ToolCallIssue(code="unsupported_partial_arguments", message="Partial.")
        call = normalize(issue=given)
        assert (call.status, call.issue, call.arguments) == ("invalid", given, {"city": "Istanbul"})

    def test_normalize_never_raises_for_provider_output(self):
        for name in (None, "", "ok", "bad name"):
            for arguments in (None, "", "{", "[]", "{}", {}, [], {"a": {1}}):
                for complete in (True, False):
                    encoded = isinstance(arguments, str)
                    assert isinstance(normalize(name=name, arguments=arguments, json_encoded=encoded,
                                                complete=complete), ToolCall)


class TestToolCallIssueFromValidationError:
    def _error(self, schema, value):
        with pytest.raises(ValidationError) as caught:
            TypeAdapter(schema).validate_python(value, strict=True)
        return caught.value

    def test_dict_type_maps_to_arguments_not_object(self):
        issue = ToolCallIssue.from_validation_error(self._error(dict[str, int], [1]))
        assert issue.code == "arguments_not_object"

    def test_other_errors_map_to_invalid_arguments(self):
        issue = ToolCallIssue.from_validation_error(self._error(dict[str, int], {"a": "x"}))
        assert issue.code == "invalid_arguments" and issue.message

    def test_json_errors_map_to_invalid_json(self):
        with pytest.raises(ValidationError) as caught:
            TypeAdapter(dict).validate_json("{")
        assert ToolCallIssue.from_validation_error(caught.value).code == "invalid_json"


class TestToolCallDelta:
    def test_append_and_snapshot_forms(self):
        assert ToolCallDelta(arguments_delta='{"a":').arguments_delta == '{"a":'
        assert ToolCallDelta(arguments={"a": 1}).arguments == {"a": 1}
        assert ToolCallDelta(name="f").name == "f"
        assert ToolCallDelta() == ToolCallDelta(name=None, arguments=None, arguments_delta=None)

    def test_append_and_replace_together_are_rejected(self):
        with pytest.raises(ValidationError, match="cannot append and replace"):
            ToolCallDelta(arguments_delta="{", arguments={})

    def test_strict_types_and_no_extra_fields(self):
        with pytest.raises(ValidationError):
            ToolCallDelta(arguments_delta=1)
        with pytest.raises(ValidationError):
            ToolCallDelta(id="x")
        with pytest.raises(ValidationError):
            ToolCallDelta(arguments={"x": float("inf")})
