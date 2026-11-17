"""The policy: one step in choosing the window, linked to the next one in a chain."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
from ..schemas import Entry


@dataclass(frozen=True)
class Selection:
    """What a chain of policies chose: the window, and what each kind of step dropped.

    Each list keeps the order of the entries the chain was given.
    """

    kept: list[Entry]
    """The window: what every step kept."""
    too_old: list[Entry]
    """Dropped by a ``WindowPolicy``: too old, so a summary should cover them."""
    unwanted: list[Entry]
    """Dropped by a ``FilterPolicy``: not wanted, so they are only hidden."""


class Policy(ABC):
    """One step in choosing the window; policies link into a chain, each choosing from what the previous one kept.

    A policy only selects: it gets entries in order and returns the ones it
    keeps, in the same order, never new or changed ones. It decides from the
    entries alone, so the same entries always give the same window, and a
    resumed session gets the window the live one had.

    Every policy is one of two kinds, which says what its drops mean:

    - a ``FilterPolicy`` drops what is **unwanted**, entry by entry (tool
      results, failed responses); those are only hidden;
    - a ``WindowPolicy`` drops what is **too old**, by position or size
      (all but the last 50); a summary should cover those.

    ``then`` links policies in the order written, and ``choose`` on the
    first one runs them all (chain of responsibility). The order matters::

        KindFilterPolicy(exclude=[EntryKind.TOOL_RESULT]).then(LastNPolicy(50))
        # the last 50 entries that are not tool results

        LastNPolicy(50).then(KindFilterPolicy(exclude=[EntryKind.TOOL_RESULT]))
        # the last 50 entries, without their tool results (50 or fewer)

    In async code, ``choose_async`` runs the chain the same way, calling
    each policy's ``select_async``. A policy only computes, with nothing to
    wait for, so by default that runs ``select`` directly; a policy that
    has to wait on something (e.g. an async API) overrides it.
    """

    _next: "Policy | None" = None

    def then(self, policy: "Policy") -> Self:
        """Link ``policy``, and the policies after it, at the end of this chain; returns this policy.

        It returns the chain's first policy, so calls can be stacked and the
        result is the chain to call. Raises ``TypeError`` for something that
        is not a policy, and ``ValueError`` for a policy already in the
        chain, which would make the chain loop.
        """
        if not isinstance(policy, Policy):
            raise TypeError(f"a chain links policies, not a {type(policy).__name__}")
        chain = self._chain()
        linked = {id(member) for member in chain}
        if any(id(member) in linked for member in policy._chain()):
            raise ValueError("a policy can be in a chain only once")
        chain[-1]._next = policy
        return self

    def choose(self, entries: list[Entry]) -> Selection:
        """Run this policy on ``entries``, then hand what it kept to the next one; returns the whole chain's choice.

        What this policy drops is recorded as ``too_old`` or ``unwanted`` by
        its kind. Raises ``ValueError`` when a policy returns an entry it was
        not given, or changes their order, and ``TypeError`` for a policy that
        is neither a ``FilterPolicy`` nor a ``WindowPolicy``.
        """
        entries = list(entries)
        kept = self.select(list(entries))
        dropped = self._dropped(entries, kept)
        rest = (
            self._next.choose(kept)
            if self._next is not None
            else Selection(kept=kept, too_old=[], unwanted=[])
        )
        return self._merge(entries, dropped, rest)

    async def choose_async(self, entries: list[Entry]) -> Selection:
        """``choose``, for async code: each policy runs ``select_async``."""
        entries = list(entries)
        kept = await self.select_async(list(entries))
        dropped = self._dropped(entries, kept)
        rest = (
            await self._next.choose_async(kept)
            if self._next is not None
            else Selection(kept=kept, too_old=[], unwanted=[])
        )
        return self._merge(entries, dropped, rest)

    @abstractmethod
    def select(self, entries: list[Entry]) -> list[Entry]:
        """This policy's one job: the entries it keeps of ``entries``, in their order."""

    async def select_async(self, entries: list[Entry]) -> list[Entry]:
        """``select``, run directly by default: a policy only computes, with nothing to wait for."""
        return self.select(entries)

    def _merge(
        self, given: list[Entry], dropped: list[Entry], rest: Selection
    ) -> Selection:
        """The whole choice: what this policy ``dropped`` of ``given``, added by its kind to the ``rest`` of the chain's choice."""
        position = {id(entry): index for index, entry in enumerate(given)}

        def in_order(*groups: list[Entry]) -> list[Entry]:
            return sorted(
                (entry for group in groups for entry in group),
                key=lambda e: position[id(e)],
            )

        if isinstance(self, FilterPolicy):
            return Selection(rest.kept, rest.too_old, in_order(dropped, rest.unwanted))
        if isinstance(self, WindowPolicy):
            return Selection(rest.kept, in_order(dropped, rest.too_old), rest.unwanted)
        raise TypeError(
            f"{type(self).__name__} must be a FilterPolicy or a WindowPolicy, "
            "so its drops are known to be unwanted or too old"
        )

    def _dropped(self, given: list[Entry], kept: list[Entry]) -> list[Entry]:
        """The entries of ``given`` that are not in ``kept``; raises ``ValueError`` unless ``kept`` is a part of ``given``, in order."""
        dropped: list[Entry] = []
        index = 0
        for entry in kept:
            while index < len(given) and given[index] is not entry:
                dropped.append(given[index])
                index += 1
            if index == len(given):
                raise ValueError(
                    f"{type(self).__name__} returned an entry it was not given, or changed their order; "
                    "a policy only selects"
                )
            index += 1
        return dropped + given[index:]

    def _chain(self) -> list["Policy"]:
        """This policy and every policy after it, in order."""
        chain: list[Policy] = []
        policy: Policy | None = self
        while policy is not None:
            chain.append(policy)
            policy = policy._next
        return chain


class FilterPolicy(Policy):
    """A policy that keeps or drops each entry on its own, whatever its position: what it drops is unwanted."""


class WindowPolicy(Policy):
    """A policy that keeps entries by their position or size: what it drops is too old."""
