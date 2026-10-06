"""Poker hand values: the best five-card hand from any number of cards.

hand_value(ranks, suits) returns an int, and a bigger int is a better hand. With five
or more cards it is the best five-card hand among them; with fewer, all the cards play
(so no straight or flush). Standard ranking, ace high except in the wheel A-2-3-4-5,
which only exists when the deck has an ace (rank 12). Comparing values is only
meaningful between hands with the same number of cards, as at a showdown.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ("high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads",
                  "straight flush")
ACE = 12


def _value(category: int, ranks: Sequence[int]) -> int:
    """Category, then up to five ranks in order of importance, 4 bits each."""
    v = category
    for i in range(5):
        v = v * 16 + (ranks[i] if i < len(ranks) else 0)
    return v


def category(value: int) -> int:
    return value // 16**5


def _straight_high(ranks: set) -> int | None:
    """Top rank of the best straight among `ranks`, or None. The wheel counts as 5-high."""
    for top in range(ACE, 3, -1):
        if all(r in ranks for r in range(top - 4, top + 1)):
            return top
    if {ACE, 0, 1, 2, 3} <= ranks:
        return 3
    return None


def hand_value(ranks: Sequence[int], suits: Sequence[int]) -> int:
    n = len(ranks)
    counts = Counter(ranks)
    desc = sorted(ranks, reverse=True)

    flush = None
    if n >= 5:
        suit, k = Counter(suits).most_common(1)[0]
        if k >= 5:  # with at most 7 cards, only one suit can have five
            flush = sorted((r for r, s in zip(ranks, suits) if s == suit), reverse=True)
            top = _straight_high(set(flush))
            if top is not None:
                return _value(STRAIGHT_FLUSH, [top])

    quads = [r for r, c in counts.items() if c == 4]
    if quads:
        q = max(quads)
        return _value(QUADS, [q] + [r for r in desc if r != q][:min(1, n - 4)])

    trips = sorted((r for r, c in counts.items() if c == 3), reverse=True)
    if trips and n >= 5:
        others = sorted((r for r, c in counts.items() if c >= 2 and r != trips[0]), reverse=True)
        if others:
            return _value(FULL_HOUSE, [trips[0], others[0]])

    if flush is not None:
        return _value(FLUSH, flush[:5])

    if n >= 5:
        top = _straight_high(set(ranks))
        if top is not None:
            return _value(STRAIGHT, [top])

    if trips:
        t = trips[0]
        return _value(TRIPS, [t] + [r for r in desc if r != t][:min(2, n - 3)])

    pairs = sorted((r for r, c in counts.items() if c == 2), reverse=True)
    if len(pairs) >= 2:
        a, b = pairs[:2]
        return _value(TWO_PAIR, [a, b] + [r for r in desc if r not in (a, b)][:min(1, n - 4)])
    if pairs:
        p = pairs[0]
        return _value(PAIR, [p] + [r for r in desc if r != p][:min(3, n - 2)])
    return _value(HIGH_CARD, desc[:5])
