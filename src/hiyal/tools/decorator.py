"""Optional eager tool inference; decorated names remain ordinary functions."""

from __future__ import annotations

from typing import Any, Callable, ParamSpec, Protocol, TypeVar, cast, overload
from .calls.spec import ToolSpec
from .introspection import UNSET, _Unset, parse_function


P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)


class ToolFunction(Protocol[P, R_co]):
    """Typing interface only; no runtime wrapper is created."""

    @property
    def tool_spec(self) -> ToolSpec: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


@overload
def tool(
    function: Callable[P, R],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None | _Unset = UNSET,
    output_schema: dict[str, Any] | None | _Unset = UNSET,
) -> ToolFunction[P, R]: ...


@overload
def tool(
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None | _Unset = UNSET,
    output_schema: dict[str, Any] | None | _Unset = UNSET,
) -> Callable[[Callable[P, R]], ToolFunction[P, R]]: ...


def tool(
    function: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None | _Unset = UNSET,
    output_schema: dict[str, Any] | None | _Unset = UNSET,
) -> ToolFunction[P, R] | Callable[[Callable[P, R]], ToolFunction[P, R]]:
    """Attach an inferred ``.tool_spec`` once, without wrapping or running a function.

    Supports ``@tool``, ``@tool()``, and explicit metadata overrides. The same
    parser is used for undecorated functions passed directly to ``Tools``.
    """

    def decorate(candidate: Callable[P, R]) -> ToolFunction[P, R]:
        if hasattr(candidate, "tool_spec"):
            raise ValueError(
                "function already has tool_spec metadata; do not decorate it twice"
            )
        spec = parse_function(
            candidate,
            name=name,
            description=description,
            strict=strict,
            output_schema=output_schema,
        )
        setattr(candidate, "tool_spec", spec)
        return cast(ToolFunction[P, R], candidate)

    return decorate(function) if function is not None else decorate
