from collections.abc import Callable, Iterable
from typing import Any
from ..schemas import LLMStreamEvent

StreamEventAdapter = Callable[[Any], Iterable[LLMStreamEvent]]
StreamFinalizer = Callable[[], Iterable[LLMStreamEvent]]
