"""Policies: steps that choose the window, linked into a chain, each choosing from what the previous one kept."""

from .base import FilterPolicy, Policy, Selection, WindowPolicy
from .head_tail import HeadTailPolicy
from .kind_filter import KindFilterPolicy
from .last_n import LastNPolicy
from .predicate_filter import PredicateFilterPolicy
from .sliding_window import SlidingWindowPolicy
from .token_window import TokenWindowPolicy, text_tokens

__all__ = [
    "FilterPolicy",
    "HeadTailPolicy",
    "KindFilterPolicy",
    "LastNPolicy",
    "Policy",
    "PredicateFilterPolicy",
    "Selection",
    "SlidingWindowPolicy",
    "TokenWindowPolicy",
    "WindowPolicy",
    "text_tokens",
]
