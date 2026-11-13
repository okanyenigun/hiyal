import json
import unittest
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

from hiyal.tools import ToolCall, ToolCallIssue


class ToolCallTests(unittest.TestCase):
    def test_complete_call_serializes_nested_json_without_raw(self):
        arguments = {
            "city": "Istanbul",
            "options": {"enabled": True, "limit": 0, "ratio": 1.5, "missing": None},
            "days": ["Monday", "Tuesday"],
        }
        raw = {"id": "fc_item", "call_id": "call_provider"}
        call = ToolCall(
            raw=raw, name="get_weather",
            arguments=arguments, status="complete",
        )
        serialized = json.loads(call.model_dump_json())
        self.assertNotIn("raw", serialized)
        self.assertNotIn("id", serialized)
        self.assertNotIn("call_id", serialized)
        self.assertNotIn("raw=", repr(call))
        restored = ToolCall.model_validate({**serialized, "raw": raw})
        self.assertEqual(restored, call)
        self.assertEqual(restored.arguments, arguments)
        self.assertEqual(call.raw, raw)
        self.assertEqual(restored.raw, raw)

    def test_provider_sdk_objects_become_json_dicts(self):
        class SdkCall(BaseModel):
            id: str
            signature: bytes

        call = ToolCall(raw=SdkCall(id="fc_item", signature=b"\x00\xff"), name="get_weather",
                        arguments={}, status="complete")
        self.assertEqual(call.raw, {"id": "fc_item", "signature": "AP8="})  # bytes as base64 text

    def test_sdk_objects_with_dates_become_json_dicts(self):
        from datetime import datetime, timezone

        class SdkEvent(BaseModel):
            at: datetime

        call = ToolCall(raw=SdkEvent(at=datetime(2026, 1, 1, tzinfo=timezone.utc)), name="f",
                        arguments={}, status="complete")
        self.assertEqual(call.raw, {"at": "2026-01-01T00:00:00Z"})
        self.assertNotIn("raw", call.model_dump())

    def test_dicts_and_none_are_kept(self):
        for raw in ({"function": {"name": "get_weather", "arguments": {}}}, None):
            with self.subTest(raw=raw):
                call = ToolCall(raw=raw, name="get_weather", arguments={}, status="complete")
                self.assertEqual(call.raw, raw)

    def test_other_objects_are_refused(self):
        for raw in (SimpleNamespace(id="x"), "text", object()):
            with self.subTest(raw=raw), self.assertRaises(TypeError):
                ToolCall(raw=raw, name="get_weather", arguments={}, status="complete")

    def test_raw_is_optional_so_a_serialized_call_can_be_reloaded(self):
        original = ToolCall(raw={"sdk": "object"}, name="get_weather", arguments={"city": "Kyoto"}, status="complete")
        reloaded = ToolCall.model_validate_json(original.model_dump_json())
        self.assertIsNone(reloaded.raw)
        self.assertEqual(reloaded.model_dump(), original.model_dump())

    def test_explicit_empty_arguments_are_valid_but_missing_arguments_are_not(self):
        call = ToolCall(raw=None, name="get_time", arguments={}, status="complete")
        self.assertEqual(call.arguments, {})
        for values in ({}, {"arguments": None}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ToolCall(raw=None, name="get_time", status="complete", **values)

    def test_complete_call_requires_name_and_no_issue(self):
        for updates in (
            {"name": None},
            {"issue": {"code": "invalid_json", "message": "Could not decode input."}},
        ):
            values = dict(raw=None, name="get_weather", arguments={}, status="complete")
            values.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                ToolCall(**values)

    def test_invalid_call_preserves_an_issue_and_absent_arguments(self):
        call = ToolCall(
            raw=None, status="invalid",
            issue=ToolCallIssue(code="missing_name", message="No name was returned."),
        )
        self.assertIsNone(call.name)
        self.assertIsNone(call.arguments)
        self.assertEqual(call.issue.code, "missing_name")
        with self.assertRaises(ValidationError):
            ToolCall(raw=None, status="invalid")

    def test_incomplete_call_does_not_infer_completion_from_parsed_arguments(self):
        call = ToolCall(raw=None, status="incomplete")
        self.assertIsNone(call.name)
        self.assertIsNone(call.arguments)
        call = ToolCall(
            raw=None, name="get_weather", arguments={"city": "Istanbul"},
            status="incomplete",
            issue={"code": "interrupted", "message": "Generation did not finish."},
        )
        self.assertEqual(call.status, "incomplete")

    def test_argument_strings_and_non_object_shapes_are_rejected(self):
        for arguments in ('{"city":"Istanbul"}', "", [], [1], 1, True):
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                ToolCall(
                    raw=None, name="get_weather", arguments=arguments,
                    status="complete",
                )

    def test_non_json_arguments_are_rejected_without_coercion(self):
        for arguments in (
            {"value": b"bytes"},
            {"value": (1, 2)},
            {b"key": 1},
            {"nested": [{"value": float("nan")}]},
            {"value": float("inf")},
            {"value": float("-inf")},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                ToolCall(raw=None, name="get_weather", arguments=arguments, status="complete")

    def test_blank_names_and_issues_are_rejected(self):
        for value in ("", " ", "bad value"):
            with self.subTest(name=value), self.assertRaises(ValidationError):
                ToolCall(raw=None, name=value, arguments={}, status="complete")
        for field in ("code", "message"):
            for value in ("", " "):
                values = dict(code="invalid_json", message="Cannot decode arguments.")
                values[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ToolCallIssue(**values)

    def test_removed_identifiers_and_undeclared_fields_are_rejected(self):
        for field in ("id", "call_id", "raw_data", "provider_data", "provider", "raw_arguments"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ToolCall(raw=None, status="incomplete", **{field: "unexpected"})
        with self.assertRaises(ValidationError):
            ToolCallIssue(code="invalid_json", message="Bad JSON.", raw="{")


if __name__ == "__main__":
    unittest.main()
