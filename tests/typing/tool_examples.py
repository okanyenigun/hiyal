"""Static typing smoke checks: pyright tests/typing/tool_examples.py."""

from typing import Annotated, assert_type

from pydantic import Field

from hiyal.tools import ToolSpec, Tools, tool


@tool
def decorated(city: Annotated[str, Field(description="City to look up.")]) -> str:
    """Look up a city."""
    return city


@tool(name="configured_lookup", strict=False)
def configured(city: Annotated[str, Field(description="City to look up.")]) -> int:
    """Count city-name characters."""
    return len(city)


def plain(city: str) -> str:
    """Look up a city.

    Args:
        city: City to look up.
    """
    return city


@tool()
async def asynchronous(city: Annotated[str, Field(description="City.")]) -> str:
    """Look up a city asynchronously."""
    return city


async def check_types() -> None:
    assert_type(decorated("Istanbul"), str)
    assert_type(configured("Istanbul"), int)
    assert_type(await asynchronous("Istanbul"), str)
    assert_type(decorated.tool_spec, ToolSpec)
    assert_type(Tools(tool_list=[decorated, configured, plain]), Tools)
    assert_type(Tools(tool_list=[decorated.tool_spec]).tool_list[0], ToolSpec)
    plain_functions = [plain]
    assert_type(Tools(tool_list=plain_functions), Tools)
