"""Kuhn poker: deck J, Q, K, ante 1, one card each, one betting round, bet 1, at most one bet."""
from __future__ import annotations

from bombpot.core.limit import LimitPoker, PokerState


class Kuhn(LimitPoker):
    name = "kuhn"
    rank_chars = "JQK"

    def __init__(self):
        events = [("hand", 0), ("hand", 1), ("bet", 0), ("showdown",)]
        super().__init__(events, num_boards=1, ante=1.0, bet_sizes=[1.0], max_raises=[1])
        self.params = {}

    def _counts(self, s: PokerState, event) -> list[int]:
        return [0 if r in s.hands else 1 for r in range(3)]

    def _strength(self, hand: int, cards: tuple) -> int:
        return hand
