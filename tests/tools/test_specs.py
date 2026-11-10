import json

import pytest
from pydantic import ValidationError

from hiyal.tools import (
    AnthropicToolSpec,
    ArgumentType,
    GeminiToolSpec,
    OllamaToolSpec,
    OpenAIToolSpec,
    ToolArgument,
    ToolParameters,
    ToolSpec,
    Tools,
)

CITY = {"name": "city", "type": "string", "description": "City.", "required": True}
OUTPUT = {"type": "object", "properties": {"c": {"type": "number"}}, "required": ["c"], "additionalProperties": False}


def spec_values(**overrides):
    values = {"name": "get_weather", "description": "Get weather.", "parameters": {"properties": [CITY]}}
    values.update(overrides)
    return values


class TestToolArgument:
    def test_alias_and_field_name_are_both_accepted(self):
        assert ToolArgument(name="a", type="string", description="A.").type_ == ArgumentType.STRING
        assert ToolArgument(name="a", type_="string", description="A.").type_ == ArgumentType.STRING

    @pytest.mark.parametrize("bad", [{"name": ""}, {"name": "two words"}, {"description": ""},
                                     {"description": "   "}, {"type": "date"}, {"required": "yes"},
                                     {"enum": []}, {"unknown": 1}])
    def test_invalid_fields_are_rejected(self, bad):
        with pytest.raises(ValidationError):
            ToolArgument(**{"name": "a", "type": "string", "description": "A.", **bad})

    def test_type_list_must_be_nonempty_and_unique(self):
        with pytest.raises(ValidationError, match="at least one unique type"):
            ToolArgument(name="a", type=[], description="A.")
        with pytest.raises(ValidationError, match="at least one unique type"):
            ToolArgument(name="a", type=["string", "string"], description="A.")

    def test_array_needs_items_and_items_need_array(self):
        with pytest.raises(ValidationError, match="array arguments need an items schema"):
            ToolArgument(name="a", type="array", description="A.")
        with pytest.raises(ValidationError, match="items is only valid for an array"):
            ToolArgument(name="a", type="string", description="A.", items={"type": "string"})
        assert ToolArgument(name="a", type=["array", "null"], description="A.", items=True)

    def test_properties_need_an_object(self):
        with pytest.raises(ValidationError, match="require an object argument"):
            ToolArgument(name="a", type="string", description="A.", properties=[CITY])
        with pytest.raises(ValidationError, match="require an object argument"):
            ToolArgument(name="a", type="string", description="A.", additionalProperties=True)

    def test_nested_names_must_be_unique(self):
        with pytest.raises(ValidationError, match="nested property names must be unique"):
            ToolArgument(name="a", type="object", description="A.", properties=[CITY, CITY])

    def test_scalar_schema_drops_config_only_fields(self):
        argument = ToolArgument(name="unit", type=["string", "null"], description="Unit.",
                                enum=["c", "f", None], required=True)
        assert argument.to_json_schema() == {"type": ["string", "null"], "description": "Unit.",
                                             "enum": ["c", "f", None]}

    def test_object_schema_is_nested_with_required_and_closed_by_default(self):
        country = {"name": "country", "type": "string", "description": "Country."}
        argument = ToolArgument(name="place", type="object", description="Place.", properties=[CITY, country])
        assert argument.to_json_schema() == {
            "type": "object", "description": "Place.",
            "properties": {"city": {"type": "string", "description": "City."},
                           "country": {"type": "string", "description": "Country."}},
            "required": ["city"], "additionalProperties": False,
        }

    def test_object_without_properties_and_explicit_open_object(self):
        argument = ToolArgument(name="extra", type="object", description="Extra.", additionalProperties=True)
        assert argument.to_json_schema() == {"type": "object", "description": "Extra.", "properties": {},
                                             "required": [], "additionalProperties": True}

    def test_array_schema_keeps_items(self):
        argument = ToolArgument(name="tags", type="array", description="Tags.", items={"type": "string"})
        assert argument.to_json_schema() == {"type": "array", "description": "Tags.", "items": {"type": "string"}}


class TestToolParameters:
    def test_defaults_and_schema(self):
        assert ToolParameters().to_json_schema() == {"type": "object", "properties": {}, "required": [],
                                                     "additionalProperties": False}

    def test_required_list_follows_property_order(self):
        optional = {"name": "unit", "type": "string", "description": "Unit."}
        schema = ToolParameters(properties=[optional, CITY]).to_json_schema()
        assert list(schema["properties"]) == ["unit", "city"] and schema["required"] == ["city"]

    def test_duplicate_names_and_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError, match="property names must be unique"):
            ToolParameters(properties=[CITY, CITY])
        with pytest.raises(ValidationError):
            ToolParameters(type="array")
        with pytest.raises(ValidationError):
            ToolParameters(required=["city"])
        with pytest.raises(ValidationError):
            ToolParameters(additionalProperties="no")


class TestToolSpecBase:
    def test_base_declaration(self):
        assert ToolSpec(**spec_values()).to_sdk_tool() == {
            "name": "get_weather", "description": "Get weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City."}},
                           "required": ["city"], "additionalProperties": False},
        }

    def test_base_has_no_provider_for_result_behavior(self):
        spec = ToolSpec(**spec_values(result_behavior="blocking"))
        with pytest.raises(ValueError, match="needs a provider-specific ToolSpec adapter"):
            spec.to_sdk_tool()

    @pytest.mark.parametrize("bad", [{"name": "get weather"}, {"name": ""}, {"description": " "},
                                     {"strict": "yes"}, {"defer_loading": 1}, {"result_behavior": "later"},
                                     {"extra": 1}])
    def test_invalid_fields_are_rejected(self, bad):
        with pytest.raises(ValidationError):
            ToolSpec(**spec_values(**bad))


class TestOpenAIToolSpec:
    def test_defaults_to_strict_and_always_sends_the_flag(self):
        wire = OpenAIToolSpec(**spec_values()).to_sdk_tool()
        assert wire["type"] == "function" and wire["strict"] is True
        assert OpenAIToolSpec(**spec_values(strict=None)).to_sdk_tool()["strict"] is None

    def test_strict_needs_every_property_required(self):
        optional = {"name": "unit", "type": "string", "description": "Unit."}
        with pytest.raises(ValidationError, match="every property in required"):
            OpenAIToolSpec(**spec_values(parameters={"properties": [CITY, optional]}))
        assert OpenAIToolSpec(**spec_values(parameters={"properties": [CITY, optional]}, strict=False))

    def test_strict_checks_nested_objects(self):
        inner = {"name": "place", "type": "object", "description": "Place.",
                 "properties": [{"name": "zip", "type": "string", "description": "Zip."}], "required": True}
        with pytest.raises(ValidationError, match=r"parameters\.properties\.place"):
            OpenAIToolSpec(**spec_values(parameters={"properties": [inner]}))

    def test_strict_rejects_open_objects(self):
        inner = {"name": "extra", "type": "object", "description": "Extra.", "additionalProperties": True,
                 "required": True}
        with pytest.raises(ValidationError, match="additionalProperties=False"):
            OpenAIToolSpec(**spec_values(parameters={"properties": [inner]}))

    def test_strict_checks_array_item_objects(self):
        items = {"type": "object", "properties": {"a": {"type": "string"}}, "required": [],
                 "additionalProperties": False}
        argument = {"name": "rows", "type": "array", "description": "Rows.", "items": items, "required": True}
        with pytest.raises(ValidationError, match=r"items: strict mode requires every property"):
            OpenAIToolSpec(**spec_values(parameters={"properties": [argument]}))

    @pytest.mark.parametrize("behavior, flag", [("blocking", False), ("non_blocking", True)])
    def test_result_behavior_maps_to_async(self, behavior, flag):
        assert OpenAIToolSpec(**spec_values(result_behavior=behavior)).to_sdk_tool()["async"] is flag

    def test_non_blocking_programmatic_calls_are_rejected(self):
        with pytest.raises(ValidationError, match="require direct, not programmatic"):
            OpenAIToolSpec(**spec_values(result_behavior="non_blocking", allowed_callers=["programmatic"]))

    def test_output_schema_and_defer_loading_pass_through(self):
        wire = OpenAIToolSpec(**spec_values(output_schema=OUTPUT, defer_loading=True,
                                            allowed_callers=["direct"])).to_sdk_tool()
        assert (wire["output_schema"], wire["defer_loading"], wire["allowed_callers"]) == (OUTPUT, True, ["direct"])

    def test_allowed_callers_must_be_known_and_nonempty(self):
        with pytest.raises(ValidationError):
            OpenAIToolSpec(**spec_values(allowed_callers=[]))
        with pytest.raises(ValidationError):
            OpenAIToolSpec(**spec_values(allowed_callers=["code_execution_20250825"]))


class TestAnthropicToolSpec:
    def test_parameters_become_input_schema(self):
        wire = AnthropicToolSpec(**spec_values()).to_sdk_tool()
        assert "parameters" not in wire and wire["input_schema"]["required"] == ["city"]
        assert "strict" not in wire and "type" not in wire

    def test_input_schema_is_accepted_on_input(self):
        spec = AnthropicToolSpec(name="f", description="F.", input_schema={"properties": [CITY]})
        assert spec.parameters.properties[0].name == "city"

    def test_native_options_pass_through(self):
        wire = AnthropicToolSpec(**spec_values(
            strict=True, type="custom", cache_control={"ttl": "1h"}, eager_input_streaming=True,
            input_examples=[{"city": "Paris"}], defer_loading=False,
            allowed_callers=["direct", "code_execution_20260521"])).to_sdk_tool()
        assert wire["strict"] is True and wire["type"] == "custom"
        assert wire["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert (wire["eager_input_streaming"], wire["input_examples"], wire["defer_loading"]) == (
            True, [{"city": "Paris"}], False)

    def test_output_schema_is_appended_to_the_description(self):
        wire = AnthropicToolSpec(**spec_values(output_schema=OUTPUT)).to_sdk_tool()
        assert "output_schema" not in wire
        assert wire["description"].startswith("Get weather.\n\nTool output JSON Schema")
        assert wire["description"].endswith(json.dumps(OUTPUT, sort_keys=True))

    def test_result_behavior(self):
        assert "async" not in AnthropicToolSpec(**spec_values(result_behavior="blocking")).to_sdk_tool()
        with pytest.raises(ValueError, match="anthropic does not support non-blocking"):
            AnthropicToolSpec(**spec_values(result_behavior="non_blocking")).to_sdk_tool()

    def test_invalid_cache_ttl(self):
        with pytest.raises(ValidationError):
            AnthropicToolSpec(**spec_values(cache_control={"ttl": "2h"}))


class TestGeminiToolSpec:
    def test_wrapped_declaration_with_parameters_json_schema(self):
        wire = GeminiToolSpec(**spec_values(strict=True, defer_loading=False)).to_sdk_tool()
        assert list(wire) == ["function_declarations"]
        declaration = wire["function_declarations"][0]
        assert "parameters_json_schema" in declaration and "parameters" not in declaration
        assert "strict" not in declaration and "defer_loading" not in declaration

    def test_zero_argument_function_omits_parameters(self):
        wire = GeminiToolSpec(name="now", description="Current time.").to_sdk_tool()
        assert wire == {"function_declarations": [{"name": "now", "description": "Current time."}]}

    @pytest.mark.parametrize("name", ["travel.get_weather", "ns:tool", "a-b", "_x"])
    def test_extended_names_are_allowed(self, name):
        assert GeminiToolSpec(name=name, description="D.")

    @pytest.mark.parametrize("name", ["1tool", "a b", "x" * 129, "a/b"])
    def test_invalid_names_are_rejected(self, name):
        with pytest.raises(ValidationError):
            GeminiToolSpec(name=name, description="D.")

    def test_output_schema_becomes_response_json_schema(self):
        declaration = GeminiToolSpec(**spec_values(output_schema=OUTPUT)).to_sdk_tool()["function_declarations"][0]
        assert declaration["response_json_schema"] == OUTPUT and "output_schema" not in declaration

    @pytest.mark.parametrize("behavior, wire", [("blocking", "BLOCKING"), ("non_blocking", "NON_BLOCKING")])
    def test_result_behavior_maps_to_behavior(self, behavior, wire):
        declaration = GeminiToolSpec(**spec_values(result_behavior=behavior)).to_sdk_tool()["function_declarations"][0]
        assert declaration["behavior"] == wire

    def test_defer_loading_true_is_rejected(self):
        with pytest.raises(ValueError, match="gemini does not support defer_loading=True"):
            GeminiToolSpec(**spec_values(defer_loading=True)).to_sdk_tool()


class TestOllamaToolSpec:
    def test_function_wrapper_without_unsupported_flags(self):
        wire = OllamaToolSpec(**spec_values(strict=True, defer_loading=False)).to_sdk_tool()
        assert wire == {"type": "function", "function": {
            "name": "get_weather", "description": "Get weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City."}},
                           "required": ["city"], "additionalProperties": False}}}

    def test_output_schema_is_appended_to_the_description(self):
        wire = OllamaToolSpec(**spec_values(output_schema=OUTPUT)).to_sdk_tool()
        assert "Tool output JSON Schema" in wire["function"]["description"]

    def test_unsupported_policies_are_rejected(self):
        with pytest.raises(ValueError, match="ollama does not support defer_loading=True"):
            OllamaToolSpec(**spec_values(defer_loading=True)).to_sdk_tool()
        with pytest.raises(ValueError, match="ollama does not support non-blocking"):
            OllamaToolSpec(**spec_values(result_behavior="non_blocking")).to_sdk_tool()


class TestTools:
    def test_defaults(self):
        tools = Tools(tool_list=[ToolSpec(**spec_values())])
        assert (tools.tool_choice, tools.parallel_tool_calls) == ("auto", False)

    def test_specs_are_copied_and_subclasses_survive(self):
        spec = OpenAIToolSpec(**spec_values(allowed_callers=["direct"]))
        tools = Tools(tool_list=[spec])
        assert tools.tool_list[0] == spec and tools.tool_list[0] is not spec
        assert type(tools.tool_list[0]) is OpenAIToolSpec

    def test_dicts_are_validated_as_base_specs(self):
        tools = Tools(tool_list=[spec_values()])
        assert type(tools.tool_list[0]) is ToolSpec

    def test_required_needs_a_tool(self):
        with pytest.raises(ValidationError, match="needs at least one tool"):
            Tools(tool_list=[], tool_choice="required")
        assert Tools(tool_list=[], tool_choice="none").tool_list == []

    def test_names_must_be_unique(self):
        with pytest.raises(ValidationError, match="tool names must be unique"):
            Tools(tool_list=[spec_values(), spec_values(description="Other.")])

    @pytest.mark.parametrize("bad", [{"tool_choice": "any"}, {"parallel_tool_calls": "yes"}, {"extra": 1}])
    def test_invalid_controls_are_rejected(self, bad):
        with pytest.raises(ValidationError):
            Tools(tool_list=[spec_values()], **bad)

    def test_callable_objects_and_bad_tool_spec_attributes_are_rejected(self):
        class Callable:
            def __call__(self):
                pass

        with pytest.raises(ValidationError, match="must be plain functions"):
            Tools(tool_list=[Callable()])

        def fake():
            """Fake."""

        fake.tool_spec = {"name": "fake"}
        with pytest.raises(ValidationError, match="function.tool_spec must be a ToolSpec"):
            Tools(tool_list=[fake])

    def test_non_sequence_input_is_rejected(self):
        with pytest.raises(ValidationError):
            Tools(tool_list="get_weather")
        with pytest.raises(ValidationError):
            Tools(tool_list=ToolSpec(**spec_values()))


class TestStrictValidationOfNativeSchemas:
    def argument(self, items):
        return {"name": "rows", "type": "array", "description": "Rows.", "items": items, "required": True}

    @pytest.mark.parametrize("items, message", [
        ({"type": "object", "properties": ["a"], "required": [], "additionalProperties": False},
         "properties must be an object and required a list"),
        ({"type": "object", "properties": {}, "required": [1], "additionalProperties": False},
         "properties must be an object and required a list"),
        ({"$defs": ["x"]}, r"\$defs: must be an object"),
        ({"anyOf": {"type": "string"}}, "anyOf: must be a list"),
        ({"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}, "required": [],
                     "additionalProperties": False}]}, r"anyOf\[0\]: strict mode requires every property"),
        ({"$defs": {"Row": {"type": "object", "properties": {}, "required": []}}},
         r"\$defs\.Row: strict mode requires additionalProperties=False"),
    ])
    def test_malformed_or_non_strict_nested_schemas(self, items, message):
        with pytest.raises(ValidationError, match=message):
            OpenAIToolSpec(**spec_values(parameters={"properties": [self.argument(items)]}))

    def test_valid_nested_schemas_pass(self):
        items = {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"a": {"type": "string"}},
                                                "required": ["a"], "additionalProperties": False}]}
        assert OpenAIToolSpec(**spec_values(parameters={"properties": [self.argument(items)]}))


class TestCustomProviderSpec:
    def test_unknown_provider_result_behavior_is_rejected(self):
        class CustomSpec(ToolSpec):
            _provider = "custom"

        with pytest.raises(ValueError, match="Unknown tool provider: custom"):
            CustomSpec(**spec_values(result_behavior="blocking")).to_sdk_tool()
        assert "parameters" in CustomSpec(**spec_values()).to_sdk_tool()
