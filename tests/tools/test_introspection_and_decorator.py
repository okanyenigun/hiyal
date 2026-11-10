"""Function inference at definition time and Tools construction time."""

from __future__ import annotations

import asyncio
import inspect
import unittest
from collections import UserList
from enum import Enum, IntEnum
from functools import partial, wraps
from typing import Annotated, Any, Literal, Optional
from unittest.mock import patch

from pydantic import Field, ValidationError

from hiyal.models.anthropic import AnthropicLLM
from hiyal.models.gemini import GeminiLLM
from hiyal.models.ollama import OllamaLLM
from hiyal.models.openai import OpenAILLM
from hiyal.models.schemas import LLMConfig
from hiyal.prompts.messages import SystemPrompt, UserPrompt
from hiyal.tools import (
    AnthropicToolSpec,
    OpenAIToolSpec,
    ToolArgument,
    ToolParameters,
    ToolSpec,
    Tools,
    tool,
)
from hiyal.tools.introspection import parse_function

# Request translation needs an LLM instance; no request is sent.
OPENAI = OpenAILLM(model="model", api_key="test-key")


class Unit(str, Enum):
    C = "c"
    F = "f"


class Level(IntEnum):
    LOW = 1
    HIGH = 2


class Mixed(Enum):
    TEXT = "text"
    NUMBER = 1


def weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up.
    """
    raise AssertionError("inference must not execute the function")


def example_function(annotation, *, default=inspect.Parameter.empty):
    def sample(value):
        """Inspect a value.

        Args:
            value: Value from the docstring.
        """
        return value

    sample.__annotations__ = {"value": annotation}
    if default is not inspect.Parameter.empty:
        sample.__defaults__ = (default,)
    return sample


def manual_spec(kind=ToolSpec, *, name="manual", **options):
    return kind(
        name=name,
        description="Manual definition.",
        parameters=ToolParameters(),
        **options,
    )


class ToolDecoratorTests(unittest.TestCase):
    def test_plain_function_matches_handwritten_spec_without_attaching_metadata(self):
        result = Tools(tool_list=[weather]).tool_list[0]
        expected = ToolSpec(
            name="weather",
            description="Get the current weather for a city.",
            parameters=ToolParameters(
                properties=[
                    ToolArgument(
                        name="city",
                        type_="string",
                        description="The city to look up.",
                        required=True,
                    )
                ]
            ),
        )
        self.assertEqual(result, expected)
        self.assertFalse(hasattr(weather, "tool_spec"))

    def test_decorator_preserves_identity_signature_and_normal_calls(self):
        function = example_function(str)
        signature = inspect.signature(function)
        decorated = tool(function)
        self.assertIs(decorated, function)
        self.assertEqual(inspect.signature(decorated), signature)
        self.assertEqual(decorated("Istanbul"), "Istanbul")
        self.assertIsInstance(decorated.tool_spec, ToolSpec)
        self.assertIsNone(decorated.tool_spec._provider)

    def test_eager_path_parses_once_and_collection_uses_the_attached_snapshot(self):
        with patch(
            "hiyal.tools.decorator.parse_function", wraps=parse_function
        ) as parse:
            decorated = tool(example_function(str))
            self.assertEqual(parse.call_count, 1)
        decorated.__doc__ = None
        decorated.__annotations__ = {"value": Any}
        with patch(
            "hiyal.tools.introspection.parse_function",
            side_effect=AssertionError("reparsed"),
        ):
            first = Tools(tool_list=[decorated])
            second = Tools(tool_list=[decorated])
        self.assertEqual(first.tool_list[0], decorated.tool_spec)
        self.assertIsNot(first.tool_list[0], second.tool_list[0])

    def test_lazy_path_parses_each_new_collection_without_modifying_function(self):
        function = example_function(str)
        with patch(
            "hiyal.tools.introspection.parse_function", wraps=parse_function
        ) as parse:
            first = Tools(tool_list=[function])
            second = Tools(tool_list=[function])
            self.assertEqual(parse.call_count, 2)
        self.assertEqual(first.tool_list, second.tool_list)
        self.assertFalse(hasattr(function, "tool_spec"))
        function.__annotations__ = {"value": int}
        third = Tools(tool_list=[function])
        self.assertEqual(first.tool_list[0].parameters.properties[0].type_, "string")
        self.assertEqual(third.tool_list[0].parameters.properties[0].type_, "integer")

    def test_bare_parenthesized_and_configured_forms(self):
        @tool
        def bare():
            """A bare tool."""

        @tool()
        def parenthesized():
            """An empty-configuration tool."""

        @tool(name="custom", description="Explicit description.", strict=False)
        def configured():
            pass

        self.assertEqual(bare.tool_spec.name, "bare")
        self.assertEqual(parenthesized.tool_spec.name, "parenthesized")
        self.assertEqual(configured.tool_spec.name, "custom")
        self.assertEqual(configured.tool_spec.description, "Explicit description.")
        self.assertIs(configured.tool_spec.strict, False)

    def test_omitted_and_explicit_policies_survive_collection_and_provider_promotion(
        self,
    ):
        for options in ({}, {"strict": None}, {"strict": False}, {"strict": True}):
            with self.subTest(options=options):
                function = tool(**options)(example_function(str))
                collected = Tools(tool_list=[function]).tool_list[0]
                self.assertEqual(
                    "strict" in collected.model_fields_set, "strict" in options
                )
                wire = OPENAI.translator.tool_definitions(
                    Tools(tool_list=[function]), OpenAIToolSpec
                )[0]
                self.assertIs(wire["strict"], options.get("strict", True))

    def test_output_schema_is_explicit_and_copied_for_collection(self):
        schema = {"type": "string"}
        function = tool(output_schema=schema)(example_function(str))
        tools = Tools(tool_list=[function])
        function.tool_spec.output_schema["type"] = "integer"
        self.assertEqual(tools.tool_list[0].output_schema, {"type": "string"})
        self.assertIn("output_schema", function.tool_spec.model_fields_set)

    def test_unresolved_return_annotation_is_not_evaluated(self):
        function = example_function(str)
        function.__annotations__["return"] = "UnknownReturnType"
        spec = Tools(tool_list=[function]).tool_list[0]
        self.assertIsNone(spec.output_schema)
        self.assertNotIn("output_schema", spec.model_fields_set)

    def test_duplicate_decoration_and_conflicting_attributes_are_rejected(self):
        function = tool(example_function(str))
        with self.assertRaisesRegex(ValueError, "already has tool_spec"):
            tool(function)
        for metadata in (None, "bad metadata"):
            function = example_function(str)
            function.tool_spec = metadata
            with self.assertRaisesRegex(ValueError, "already has tool_spec"):
                tool(function)
            with self.assertRaisesRegex(ValueError, "must be a ToolSpec"):
                Tools(tool_list=[function])

    def test_errors_happen_at_definition_or_collection_time_as_requested(self):
        def invalid(value):
            """Missing input annotation."""

        with self.assertRaisesRegex(ValueError, "annotation is required"):
            tool(invalid)
        with self.assertRaisesRegex(ValueError, "annotation is required"):
            Tools(tool_list=[invalid])
        self.assertFalse(hasattr(invalid, "tool_spec"))


class ToolIntrospectionTests(unittest.TestCase):
    def test_primitives_typed_arrays_literals_enums_and_nullable_variants(self):
        cases = [
            (str, {"type": "string"}),
            (int, {"type": "integer"}),
            (float, {"type": "number"}),
            (bool, {"type": "boolean"}),
            (str | None, {"type": ["string", "null"]}),
            (Optional[int], {"type": ["integer", "null"]}),
            (Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}),
            (Literal[1, 2], {"type": "integer", "enum": [1, 2]}),
            (Literal[True, False], {"type": "boolean", "enum": [True, False]}),
            (Literal["a", None], {"type": ["string", "null"], "enum": ["a", None]}),
            (
                Literal["a", "b"] | None,
                {"type": ["string", "null"], "enum": ["a", "b", None]},
            ),
            (Unit, {"type": "string", "enum": ["c", "f"]}),
            (Level, {"type": "integer", "enum": [1, 2]}),
            (Unit | None, {"type": ["string", "null"], "enum": ["c", "f", None]}),
            (list[str], {"type": "array", "items": {"type": "string"}}),
            (
                list[list[int | None]],
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": ["integer", "null"]}},
                },
            ),
            (
                list[str] | None,
                {"type": ["array", "null"], "items": {"type": "string"}},
            ),
        ]
        for annotation, expected in cases:
            with self.subTest(annotation=annotation):
                schema = (
                    Tools(tool_list=[example_function(annotation)])
                    .tool_list[0]
                    .parameters.to_json_schema()
                )
                self.assertEqual(schema["required"], ["value"])
                actual = schema["properties"]["value"]
                self.assertEqual(actual.pop("description"), "Value from the docstring.")
                self.assertEqual(actual, expected)

    def test_defaults_do_not_change_nullability_or_leak_into_schema(self):
        cases = [
            (str, "default", False, "string"),
            (str | None, None, False, ["string", "null"]),
            (str | None, inspect.Parameter.empty, True, ["string", "null"]),
        ]
        for annotation, default, required, expected_type in cases:
            result = Tools(
                tool_list=[example_function(annotation, default=default)]
            ).tool_list[0]
            argument = result.parameters.properties[0]
            self.assertEqual(argument.required, required)
            self.assertEqual(argument.type_, expected_type)
            self.assertNotIn("default", argument.to_json_schema())

    def test_invalid_defaults_are_rejected_without_coercion(self):
        for annotation, default in [
            (str, None),
            (int, "1"),
            (int, True),
            (bool, 1),
            (list[int], ["1"]),
            (Literal["a"], "b"),
            (Literal[1], True),
            (Literal[True], 1),
            (Unit, "c"),
            (float, float("inf")),
            (list[float], [float("nan")]),
        ]:
            with self.subTest(annotation=annotation, default=default):
                with self.assertRaisesRegex(ValueError, "default"):
                    Tools(tool_list=[example_function(annotation, default=default)])

    def test_docstring_summary_long_description_and_multiline_argument(self):
        def function(city: str):
            """Get weather.

            Use this for current conditions.

            Args:
                city: The city to look up.
                    Include its country when ambiguous.

            Returns:
                str: A forecast.
            """

        spec = Tools(tool_list=[function]).tool_list[0]
        self.assertEqual(
            spec.description, "Get weather.\n\nUse this for current conditions."
        )
        self.assertEqual(
            spec.parameters.properties[0].description,
            "The city to look up.\nInclude its country when ambiguous.",
        )
        self.assertIsNone(spec.output_schema)

    def test_annotated_description_overrides_docstring(self):
        function = example_function(
            Annotated[str, Field(description="Metadata description.")]
        )
        self.assertEqual(
            Tools(tool_list=[function])
            .tool_list[0]
            .parameters.properties[0]
            .description,
            "Metadata description.",
        )

    def test_all_explicit_descriptions_work_without_docstring(self):
        function = example_function(Annotated[str, Field(description="Value.")])
        function.__doc__ = None
        function = tool(description="Explicit tool description.")(function)
        self.assertEqual(function.tool_spec.description, "Explicit tool description.")

    def test_constraints_and_unrecognized_metadata_are_not_silently_dropped(self):
        annotations = [
            Annotated[int, Field(ge=0)],
            Annotated[str, Field(alias="other")],
            Annotated[str, Field(default="x")],
            Annotated[str, Field(default_factory=str)],
            Annotated[str, "unrecognized"],
            Annotated[str, Field(description="")],
            Annotated[str, Field(description="A"), Field(description="B")],
            list[Annotated[int, Field(ge=0)]],
        ]
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaises(ValueError):
                    Tools(tool_list=[example_function(annotation)])

    def test_nested_array_descriptions_are_retained(self):
        function = example_function(list[Annotated[str, Field(description="An item.")]])
        result = Tools(tool_list=[function]).tool_list[0]
        self.assertEqual(
            result.parameters.properties[0].items,
            {"type": "string", "description": "An item."},
        )

    def test_unsupported_annotations_fail_instead_of_becoming_any(self):
        class Custom:
            pass

        annotations = [
            Any,
            list,
            dict,
            dict[str, int],
            tuple[int, str],
            set[str],
            int | str,
            Custom,
            Mixed,
            Literal[1, "a"],
            list[Any],
            [],
        ]
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaises(ValueError):
                    Tools(tool_list=[example_function(annotation)])

    def test_bad_names_and_missing_descriptions_are_actionable(self):
        for name in ("has space", "dotted.name", "1name", "a" * 65, ""):
            with self.assertRaisesRegex(ValueError, "tool name"):
                tool(name=name)(example_function(str))
        function = example_function(str)
        function.__doc__ = None
        with self.assertRaisesRegex(ValueError, "tool description"):
            tool(function)
        function.__doc__ = "Tool description, but no argument descriptions."
        with self.assertRaisesRegex(ValueError, "parameter 'value'.*description"):
            tool(function)

    def test_stale_or_duplicate_docstring_parameters_are_rejected(self):
        for doc in [
            "Tool.\n\nArgs:\n    other: Unknown parameter.",
            "Tool.\n\nArgs:\n    value: First.\n    value: Second.",
        ]:
            function = example_function(str)
            function.__doc__ = doc
            with self.assertRaisesRegex(ValueError, "docstring"):
                tool(function)

    def test_malformed_docstring_errors_are_local_definition_errors(self):
        function = example_function(str)
        function.__doc__ = "A tool.\n\nArgs:\n    malformed without colon"
        with self.assertRaisesRegex(
            ValueError, "Tool 'sample'.*invalid Google-style docstring"
        ):
            tool(function)
        with self.assertRaisesRegex(ValidationError, "invalid Google-style docstring"):
            Tools(tool_list=[function])

    def test_valid_defaults_remain_unchanged_on_the_function(self):
        for annotation, default in [
            (float, 1),
            (float, 1.5),
            (list[int], [1, 2]),
            (Unit, Unit.C),
            (Literal[1], 1),
            (Literal[True], True),
        ]:
            with self.subTest(annotation=annotation):
                function = example_function(annotation, default=default)
                spec = Tools(tool_list=[function]).tool_list[0]
                self.assertFalse(spec.parameters.properties[0].required)
                self.assertIs(function.__defaults__[0], default)

    def test_typed_list_of_nullable_literals_includes_null_in_items_enum(self):
        function = example_function(list[Literal["a"] | None])
        schema = Tools(tool_list=[function]).tool_list[0].parameters.to_json_schema()
        self.assertEqual(
            schema["properties"]["value"]["items"],
            {"type": ["string", "null"], "enum": ["a", None]},
        )

    def test_forward_references_resolve_but_unknown_inputs_are_rejected(self):
        result = Tools(tool_list=[example_function("str")]).tool_list[0]
        self.assertEqual(result.parameters.properties[0].type_, "string")
        with self.assertRaisesRegex(ValueError, "parameter 'value'.*resolve"):
            Tools(tool_list=[example_function("UnresolvedType")])

    def test_keyword_only_parameters_and_declared_order_are_preserved(self):
        def function(city: str, *, enabled: bool, count: int = 1):
            """Inspect ordered arguments.

            Args:
                city: City.
                enabled: Enabled.
                count: Count.
            """

        result = Tools(tool_list=[function]).tool_list[0]
        self.assertEqual(
            [p.name for p in result.parameters.properties], ["city", "enabled", "count"]
        )
        self.assertEqual(
            result.parameters.to_json_schema()["required"], ["city", "enabled"]
        )

    def test_disallowed_parameter_kinds_and_receivers(self):
        def positional(value: str, /):
            """Positional-only."""

        def variadic(*values: str):
            """Variadic."""

        def keywords(**values: str):
            """Variadic keywords."""

        def receiver(self: str):
            """Receiver."""

        for function in (positional, variadic, keywords, receiver):
            with self.assertRaises(ValueError):
                Tools(tool_list=[function])

    def test_generators_methods_partials_and_callable_instances_are_rejected(self):
        def generator():
            """Generator."""
            yield 1

        async def async_generator():
            """Async generator."""
            yield 1

        class Application:
            def method(self):
                """A bound method."""

            def __call__(self):
                pass

        for function in (
            generator,
            async_generator,
            Application().method,
            partial(weather, "Istanbul"),
            Application(),
        ):
            with self.subTest(function=function):
                with self.assertRaises(ValueError):
                    Tools(tool_list=[function])

    def test_wrapped_functions_use_the_advertised_signature(self):
        @wraps(weather)
        def wrapper(*args, **kwargs):
            raise AssertionError("must not execute wrapper")

        result = Tools(tool_list=[wrapper]).tool_list[0]
        self.assertEqual(result.name, "weather")
        self.assertEqual(result.parameters.properties[0].name, "city")


class ToolsCollectionTests(unittest.TestCase):
    def test_mixed_inputs_are_independent_specs_and_provider_options_survive(self):
        function = tool(example_function(str))
        provider = manual_spec(
            AnthropicToolSpec, input_examples=[{}], eager_input_streaming=True
        )
        tools = Tools(tool_list=[weather, function, provider])
        self.assertTrue(all(isinstance(item, ToolSpec) for item in tools.tool_list))
        self.assertIsInstance(tools.tool_list[2], AnthropicToolSpec)
        self.assertTrue(tools.tool_list[2].eager_input_streaming)
        tools.tool_list[1].parameters.properties[0].description = "Changed locally."
        tools.tool_list[2].input_examples[0]["new"] = True
        self.assertEqual(
            function.tool_spec.parameters.properties[0].description,
            "Value from the docstring.",
        )
        self.assertEqual(provider.input_examples, [{}])

    def test_plain_decorated_and_manual_duplicates_are_rejected(self):
        for items in (
            [weather, weather],
            [weather, manual_spec(name="weather")],
            [tool(example_function(str)), manual_spec(name="sample")],
        ):
            with self.assertRaisesRegex(ValueError, "names must be unique"):
                Tools(tool_list=items)

    def test_dict_validation_sequences_and_schema_serialization(self):
        for value in ([weather], (weather,), UserList([weather])):
            result = Tools.model_validate({"tool_list": value})
            self.assertIsInstance(result.tool_list[0], ToolSpec)
            self.assertEqual(
                result.model_dump_json(), Tools(tool_list=[weather]).model_dump_json()
            )
        result = Tools(tool_list=[manual_spec().model_dump()])
        self.assertEqual(result.tool_list[0].name, "manual")
        Tools.model_json_schema()

    def test_json_round_trip_contains_declarations_not_function_metadata(self):
        function = tool(example_function(str))
        result = Tools(tool_list=[function])
        self.assertEqual(Tools.model_validate_json(result.model_dump_json()), result)
        self.assertNotIn("tool_spec", result.model_dump_json())

    def test_all_provider_requests_accept_both_paths_without_callable_payloads(self):
        for eager in (False, True):
            function = example_function(str)
            if eager:
                function = tool(function)
            for kind in (OpenAILLM, AnthropicLLM, GeminiLLM, OllamaLLM):
                with self.subTest(eager=eager, provider=kind.provider):
                    cfg = LLMConfig(
                        model="model",
                        system_prompt=SystemPrompt(text="Be helpful."),
                        user_prompt=UserPrompt(text="Use the tool."),
                        tools=Tools(tool_list=[function]),
                    )
                    llm = kind(model="model", api_key="test-key")
                    local = llm.resolver.call_config(llm.config, cfg, {})
                    params = llm.translator.build_request(local)
                    declarations = params.get("config", params)["tools"]
                    self.assertTrue(declarations)
                    self.assertNotIn("tool_spec", repr(declarations))
                    self.assertIsNot(local.tools.tool_list[0], cfg.tools.tool_list[0])

    def test_optional_parameters_keep_existing_openai_strict_policy(self):
        function = example_function(str, default="default")
        tools = Tools(tool_list=[function])
        with self.assertRaisesRegex(ValueError, "every property in required"):
            OPENAI.translator.tool_definitions(tools, OpenAIToolSpec)
        function = tool(strict=False)(function)
        wire = OPENAI.translator.tool_definitions(Tools(tool_list=[function]), OpenAIToolSpec)[
            0
        ]
        self.assertFalse(wire["strict"])
        self.assertEqual(wire["parameters"]["required"], [])

    def test_generation_config_can_normalize_functions_in_nested_tools_input(self):
        cfg = LLMConfig(
            model="test",
            system_prompt=SystemPrompt(text="System."),
            user_prompt=UserPrompt(text="User."),
            tools={"tool_list": [weather]},
        )
        self.assertIsInstance(cfg.tools.tool_list[0], ToolSpec)


class AsyncToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_decorated_async_functions_stay_coroutine_functions(self):
        calls = []

        async def lookup(city: str) -> str:
            """Look up a city.

            Args:
                city: City to look up.
            """
            calls.append(city)
            await asyncio.sleep(0)
            return city

        undecorated_spec = Tools(tool_list=[lookup]).tool_list[0]
        decorated = tool(lookup)
        self.assertIs(decorated, lookup)
        self.assertTrue(inspect.iscoroutinefunction(decorated))
        self.assertEqual(Tools(tool_list=[decorated]).tool_list[0], undecorated_spec)
        self.assertEqual(calls, [])
        self.assertEqual(await decorated("Istanbul"), "Istanbul")
        self.assertEqual(calls, ["Istanbul"])
        self.assertIsNone(decorated.tool_spec.result_behavior)


if __name__ == "__main__":
    unittest.main()
