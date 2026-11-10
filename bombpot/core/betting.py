"""The no-limit / pot-limit betting engine: pot-fraction sizes, stacks and all-in (PotPoker).

Same event-list design as core/limit.py: a game lists its events (deals, betting streets,
showdown) and supplies the deals and hand strengths; PotPoker reuses LimitPoker's dealing,
payoffs and keys (LimitPoker itself is unchanged) and replaces the betting:

- Each player has `ante + stack` chips; `PokerState.contrib` holds what each has put in,
  so the chips behind are ante + stack - contrib. Chips are floats.
- Actions: "f" (fold, only when facing a bet), "c" (check or call), "r<x>" (bet or raise
  x pot, written with the shortest repr: "r0.5", "r1") and "a" (all-in).
  A bet of x pot puts x times the current pot in. A raise of x pot calls, then raises by
  x times the pot after the call. Both are "raise to contrib[o] x (1 + 2x)" for the player
  facing contrib[o], since the street starts with equal contributions.
- A size that reaches or exceeds the actor's chips becomes "a"; if several do, "a" appears
  once. There is no separate shove: "a" exists only when a listed size collapses to it.
- Minimum raise: when facing, a raise must add at least the previous bet or raise on the
  street (heads-up with equal stacks that is exactly the amount faced), otherwise the size
  is dropped; only "a" may be smaller.
- `max_aggressive[t]` bets and raises per street ("a" counts).
- pot-limit caps every size at the pot (sizes above 1 become 1).
- Once both players are all-in (an all-in and its call) no more betting happens: the
  remaining betting streets get an empty betting string, their boards are dealt, then the
  showdown.
"""
from __future__ import annotations

from bombpot.core.limit import CALL, FOLD, LimitPoker, PokerState

ALLIN = "a"
LIMITS = ("no-limit", "pot-limit")


def size_label(x: float) -> str:
    """The action string of a bet or raise of x pot: r0.5, r1, r0.75, r2."""
    return "r" + format(float(x), "g")


class PotPoker(LimitPoker):
    """Two-player poker with pot-fraction bets and raises, stacks and all-in, driven by an
    event list exactly like LimitPoker (whose dealing, payoffs and keys it inherits)."""

    def __init__(self, events, num_boards, ante, stack, sizes=(0.5, 1.0), max_aggressive=2, limit="no-limit"):
        self.events = tuple(events)
        self.kinds = tuple(e[0] for e in self.events)
        self.num_boards = num_boards
        self.num_streets = self.kinds.count("bet")
        if limit not in LIMITS:
            raise ValueError(f"limit must be one of {LIMITS}")
        if float(stack) <= 0:
            raise ValueError("stack must be positive")
        self.ante, self.stack, self.limit = float(ante), float(stack), limit
        self.total = self.ante + self.stack  # chips each player starts with
        sizes = [float(x) for x in sizes]
        if not sizes or any(x <= 0 for x in sizes):
            raise ValueError("sizes must be positive pot fractions")
        if limit == "pot-limit":
            sizes = [min(x, 1.0) for x in sizes]
        self.sizes = tuple(sorted(set(sizes)))
        self.size_labels = tuple(size_label(x) for x in self.sizes)
        if len(set(self.size_labels)) != len(self.sizes):
            raise ValueError("two sizes share an action string")
        self._size_of = dict(zip(self.size_labels, self.sizes))
        if isinstance(max_aggressive, int):
            max_aggressive = [max_aggressive] * self.num_streets
        self.max_aggressive = tuple(int(m) for m in max_aggressive)
        if len(self.max_aggressive) < self.num_streets:
            raise ValueError("need a cap on aggressive actions for every street")

    # Betting --------------------------------------------------------------------------

    def aggressive_options(self, s: PokerState) -> list[tuple[str, float]]:
        """(action, chips the actor will have in) for each legal bet or raise, smallest first."""
        p, o = s.to_act, 1 - s.to_act
        c = s.contrib
        if s.raises >= self.max_aggressive[len(s.hist) - 1] or c[o] >= self.total:
            return []
        facing = c[o] - c[p]
        out, allin = [], False
        for x, label in zip(self.sizes, self.size_labels):
            target = c[o] * (1.0 + 2.0 * x)
            if target >= self.total:
                allin = True
            elif facing > 0 and target - c[o] < facing:  # below the minimum raise
                continue
            else:
                out.append((label, target))
        if allin:
            out.append((ALLIN, self.total))
        return out

    def legal_actions(self, s: PokerState) -> tuple[str, ...]:
        p = s.to_act
        facing = s.contrib[1 - p] > s.contrib[p]
        raises = tuple(a for a, _ in self.aggressive_options(s))
        return ((FOLD, CALL) if facing else (CALL,)) + raises

    def all_in(self, s: PokerState) -> bool:
        return s.contrib[0] >= self.total and s.contrib[1] >= self.total

    def _enter(self, s: PokerState, step: int) -> PokerState:
        """Move to event `step`; a betting street opens a fresh betting string, and is
        passed over (with an empty string) once both players are all-in."""
        if self.kinds[step] == "bet":
            if self.all_in(s):
                s = PokerState(step, s.hands, s.boards, s.hist + ("",), s.contrib, 0, 0, -1)
                return self._enter(s, step + 1)
            return PokerState(step, s.hands, s.boards, s.hist + ("",), s.contrib, 0, 0, -1)
        return PokerState(step, s.hands, s.boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)

    def _bet(self, s: PokerState, a: str) -> PokerState:
        p, o = s.to_act, 1 - s.to_act
        t = len(s.hist) - 1
        hist = s.hist[:t] + (s.hist[t] + a,)
        c = s.contrib
        if a == FOLD:
            if not c[o] > c[p]:
                raise ValueError("fold is legal only when facing a bet")
            return PokerState(s.step, s.hands, s.boards, hist, c, s.raises, p, p)
        if a == CALL:
            if c[o] > c[p]:  # a call closes the street
                s = PokerState(s.step, s.hands, s.boards, hist, (c[o], c[o]), s.raises, o, -1)
                return self._enter(s, s.step + 1)
            s = PokerState(s.step, s.hands, s.boards, hist, c, s.raises, o, -1)
            if p == 1:  # player 1 not facing a bet: a check behind closes the street
                return self._enter(s, s.step + 1)
            return s
        target = dict(self.aggressive_options(s)).get(a)
        if target is None:
            raise ValueError(f"illegal action {a!r} at {s.hist}")
        c = (target, c[1]) if p == 0 else (c[0], target)
        return PokerState(s.step, s.hands, s.boards, hist, c, s.raises + 1, o, -1)
