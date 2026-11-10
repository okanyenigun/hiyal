"""Build declarative tools from trusted application functions, without calling them."""

from __future__ import annotations

import inspect
import math
import re
from enum import Enum
from types import SimpleNamespace, UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from docstring_parser import DocstringStyle, ParseError, parse
from pydantic import Field, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from .calls.spec import ToolArgument, ToolParameters, ToolSpec


class _Unset:
    """Distinguish an omitted policy from an explicitly supplied None."""


UNSET = _Unset()
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_FIELD_DEFAULTS = Field().asdict()["attributes"]


def _annotated(annotation: Any) -> tuple[Any, str | None]:
    description = None
    if get_origin(annotation) is Annotated:
        annotation, *metadata = get_args(annotation)
        for item in metadata:
            if not isinstance(item, FieldInfo):
                raise ValueError(
                    "only Annotated[..., Field(description=...)] metadata is supported"
                )
            fields = item.asdict()
            unsupported = [
                key
                for key, value in fields["attributes"].items()
                if key != "description" and value != _FIELD_DEFAULTS[key]
            ]
            if fields["metadata"] or unsupported:
                raise ValueError(
                    "Field constraints, aliases, defaults, and other metadata are not supported"
                )
            if item.description is not None:
                if description is not None:
                    raise ValueError("supply only one Field description per annotation")
                description = item.description
    return annotation, description


def _enum_schema(values: list[Any], *, enum_class: bool = False) -> dict[str, Any]:
    nonnull = [value for value in values if value is not None]
    kinds = {type(value) for value in nonnull}
    allowed = {str, int} if enum_class else set(_TYPES)
    if len(kinds) != 1 or not kinds <= allowed:
        raise ValueError("literal/enum values must have one supported scalar type")
    if any(isinstance(value, float) and not math.isfinite(value) for value in values):
        raise ValueError("literal values must be finite JSON values")
    kind = _TYPES[next(iter(kinds))]
    return {
        "type": [kind, "null"] if None in values else kind,
        "enum": values,
    }


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    annotation, description = _annotated(annotation)
    origin = get_origin(annotation)
    if isinstance(annotation, type) and annotation in _TYPES:
        schema: dict[str, Any] = {"type": _TYPES[annotation]}
    elif origin in (Union, UnionType):
        arguments = get_args(annotation)
        if len(arguments) != 2 or type(None) not in arguments:
            raise ValueError("only a supported type unioned with None is supported")
        inner = next(value for value in arguments if value is not type(None))
        schema = _annotation_schema(inner)
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        schema["type"] = list(dict.fromkeys([*types, "null"]))
        if "enum" in schema and None not in schema["enum"]:
            schema["enum"] = [*schema["enum"], None]
    elif origin is Literal:
        schema = _enum_schema(list(get_args(annotation)))
    elif inspect.isclass(annotation) and issubclass(annotation, Enum):
        schema = _enum_schema([member.value for member in annotation], enum_class=True)
    elif origin is list and len(get_args(annotation)) == 1:
        schema = {"type": "array", "items": _annotation_schema(get_args(annotation)[0])}
    else:
        raise ValueError(
            "unsupported input annotation; use str, int, float, bool, typed lists, "
            "scalar Literal/Enum, or a supported nullable type"
        )
    if description is not None:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Field description must not be blank")
        schema["description"] = description
    return schema


def _input_annotations(
    function: Callable[..., Any], signature: inspect.Signature
) -> dict[str, Any]:
    """Resolve inputs only: return annotations do not define output schemas yet."""
    namespace = inspect.unwrap(function).__globals__
    resolved = {}
    for parameter in signature.parameters.values():
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError(
                f"parameter '{parameter.name}': an input annotation is required"
            )
        holder = SimpleNamespace(__annotations__={parameter.name: parameter.annotation})
        try:
            resolved[parameter.name] = get_type_hints(
                holder,
                globalns=namespace,
                include_extras=True,
            )[parameter.name]
        except (NameError, TypeError, SyntaxError) as error:
            raise ValueError(
                f"parameter '{parameter.name}': could not resolve its annotation"
            ) from error
    return resolved


def parse_function(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None | _Unset = UNSET,
    output_schema: dict[str, Any] | None | _Unset = UNSET,
) -> ToolSpec:
    """Inspect a plain sync/async function using Google-style argument docs.

    Annotation evaluation assumes trusted application code. This builds input
    declarations only; it does not execute a body or infer its output schema.
    """
    if not inspect.isfunction(function):
        raise ValueError(
            "tools must be plain functions; methods, partials, and callable objects are not supported"
        )
    try:
        unwrapped = inspect.unwrap(function)
        if not inspect.isfunction(unwrapped):
            raise ValueError("wrapped targets must also be plain functions")
        if inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(
            unwrapped
        ):
            raise ValueError("generator and async-generator tools are not supported")
        signature = inspect.signature(function)
        for parameter in signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise ValueError(
                    f"parameter '{parameter.name}': positional-only and variadic parameters are not supported"
                )
            if parameter.name in {"self", "cls"}:
                raise ValueError(
                    f"parameter '{parameter.name}': method binding is not supported"
                )

        tool_name = function.__name__ if name is None else name
        if not isinstance(tool_name, str) or _NAME.fullmatch(tool_name) is None:
            raise ValueError("tool name must match [A-Za-z_][A-Za-z0-9_]{0,63}")
        try:
            doc = parse(inspect.getdoc(function) or "", style=DocstringStyle.GOOGLE)
        except ParseError as error:
            raise ValueError(f"invalid Google-style docstring: {error}") from error
        tool_description = (
            description
            if description is not None
            else "\n\n".join(
                text for text in (doc.short_description, doc.long_description) if text
            )
        )
        if not isinstance(tool_description, str) or not tool_description.strip():
            raise ValueError(
                "a tool description is required; add a docstring or description=..."
            )
        docs = {}
        for parameter in doc.params:
            if parameter.arg_name not in signature.parameters:
                raise ValueError(
                    f"docstring names unknown parameter '{parameter.arg_name}'"
                )
            if parameter.arg_name in docs:
                raise ValueError(f"docstring repeats parameter '{parameter.arg_name}'")
            docs[parameter.arg_name] = parameter.description

        hints = _input_annotations(function, signature)
        properties = []
        for parameter in signature.parameters.values():
            try:
                schema = _annotation_schema(hints[parameter.name])
                argument_description = schema.pop("description", None)
                if argument_description is None:
                    argument_description = docs.get(parameter.name)
                if (
                    not isinstance(argument_description, str)
                    or not argument_description.strip()
                ):
                    raise ValueError(
                        "a description is required in Args: or Annotated[..., Field(description=...)]"
                    )
                required = parameter.default is inspect.Parameter.empty
                if not required:
                    try:
                        TypeAdapter(hints[parameter.name]).validate_python(
                            parameter.default, strict=True
                        )
                    except ValidationError as error:
                        raise ValueError(
                            "the Python default does not match its annotation"
                        ) from error
                    # Pydantic Literal accepts e.g. True for Literal[1], even in
                    # strict mode. Check JSON scalar kinds without coercion too.
                    _check_default(parameter.default, schema)
                properties.append(
                    ToolArgument(
                        name=parameter.name,
                        description=argument_description,
                        required=required,
                        **schema,
                    )
                )
            except ValueError as error:
                raise ValueError(f"parameter '{parameter.name}': {error}") from error

        options: dict[str, Any] = {}
        if not isinstance(strict, _Unset):
            options["strict"] = strict
        if not isinstance(output_schema, _Unset):
            options["output_schema"] = output_schema
        return ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters=ToolParameters(properties=properties),
            **options,
        )
    except ValueError as error:
        raise ValueError(f"Tool '{function.__name__}': {error}") from error


def _check_default(value: Any, schema: dict[str, Any]) -> None:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Python defaults must contain only finite numbers")
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    kind = "null" if value is None else _TYPES.get(type(value))
    if isinstance(value, list):
        kind = "array"
    if kind not in types and not (kind == "integer" and "number" in types):
        raise ValueError("the Python default does not match its annotation")
    if "enum" in schema and not any(
        type(value) is type(item) and value == item for item in schema["enum"]
    ):
        raise ValueError(
            "the Python default does not match its literal/enum annotation"
        )
    if isinstance(value, list):
        for item in value:
            _check_default(item, schema["items"])
