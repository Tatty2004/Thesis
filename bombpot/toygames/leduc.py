"""Leduc poker, matching OpenSpiel's leduc_poker.

Ranks J, Q, K with two suits each. Ante 1, one card each, a betting round (bet 2,
at most 2 raises), one board card, a betting round (bet 4, at most 2 raises).
Pairing the board wins, otherwise the higher card wins, and equal ranks split.
Keys hold ranks only, so the tree matches leduc_poker(suit_isomorphism=True)
exactly and leduc_poker() up to suits.
"""
from __future__ import annotations

from bombpot.core.limit import LimitPoker, PokerState


class Leduc(LimitPoker):
    name = "leduc"
    rank_chars = "JQK"

    def __init__(self):
        events = [("hand", 0), ("hand", 1), ("bet", 0), ("board", 0), ("bet", 1), ("showdown",)]
        super().__init__(events, num_boards=1, ante=1.0, bet_sizes=[2.0, 4.0], max_raises=[2, 2])
        self.params = {}

    def _counts(self, s: PokerState, event) -> list[int]:
        used = s.hands + s.boards[0]
        return [2 - used.count(r) for r in range(3)]

    def _strength(self, hand: int, cards: tuple) -> int:
        return hand + 3 * (hand in cards)
