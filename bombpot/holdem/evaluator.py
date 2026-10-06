"""Poker hand values: the best five-card hand from any number of cards.

hand_value(ranks, suits) returns an int, and a bigger int is a better hand. With five
or more cards it is the best five-card hand among them; with fewer, all the cards play
(so no straight or flush). Standard ranking, ace high except in the wheel A-2-3-4-5,
which only exists when the deck has an ace (rank 12). Comparing values is only
meaningful between hands with the same number of cards, as at a showdown.

hand_values does the same for many hands at once, compiled with numba; it must equal
hand_value exactly (tests/holdem/test_holdem_evaluator.py).
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from numba import njit, prange

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


# Compiled version for bulk evaluation (must equal hand_value exactly) ------------------------


@njit(cache=True)
def _nb_straight_top(mask: int) -> int:
    """Top rank of the best straight in a 13-bit rank mask (3 for the wheel), or -1."""
    for top in range(ACE, 3, -1):
        if (mask >> (top - 4)) & 0x1F == 0x1F:
            return top
    if mask & (1 << ACE) and mask & 0xF == 0xF:
        return 3
    return -1


@njit(cache=True)
def _nb_pack(category: int, r: np.ndarray, n: int) -> int:
    v = category
    for i in range(5):
        v = v * 16 + (r[i] if i < n else 0)
    return v


@njit(cache=True)
def _nb_value(ranks: np.ndarray, suits: np.ndarray) -> int:
    n = ranks.shape[0]
    counts = np.zeros(13, np.int64)
    suit_count = np.zeros(4, np.int64)
    suit_mask = np.zeros(4, np.int64)
    mask = 0
    for c in range(n):
        counts[ranks[c]] += 1
        suit_count[suits[c]] += 1
        suit_mask[suits[c]] |= 1 << ranks[c]
        mask |= 1 << ranks[c]
    out = np.zeros(5, np.int64)
    flush = -1
    if n >= 5:
        for s in range(4):
            if suit_count[s] >= 5:
                flush = s
        if flush >= 0:
            top = _nb_straight_top(suit_mask[flush])
            if top >= 0:
                out[0] = top
                return _nb_pack(STRAIGHT_FLUSH, out, 1)
    quad = trip = -1
    for r in range(12, -1, -1):
        if counts[r] == 4 and quad < 0:
            quad = r
        if counts[r] == 3 and trip < 0:
            trip = r
    if quad >= 0:
        out[0] = quad
        m = 1
        for r in range(12, -1, -1):
            if r != quad and counts[r] > 0 and m < 1 + min(1, n - 4):
                out[m] = r
                m += 1
        return _nb_pack(QUADS, out, m)
    if trip >= 0 and n >= 5:
        for r in range(12, -1, -1):
            if r != trip and counts[r] >= 2:
                out[0] = trip
                out[1] = r
                return _nb_pack(FULL_HOUSE, out, 2)
    if flush >= 0:
        m = 0
        for r in range(12, -1, -1):
            if suit_mask[flush] >> r & 1 and m < 5:
                out[m] = r
                m += 1
        return _nb_pack(FLUSH, out, 5)
    if n >= 5:
        top = _nb_straight_top(mask)
        if top >= 0:
            out[0] = top
            return _nb_pack(STRAIGHT, out, 1)
    if trip >= 0:
        out[0] = trip
        m = 1
        for r in range(12, -1, -1):  # kickers, highest first, counted with multiplicity
            for _ in range(counts[r] if r != trip else 0):
                if m < 1 + min(2, n - 3):
                    out[m] = r
                    m += 1
        return _nb_pack(TRIPS, out, m)
    pa = pb = -1
    for r in range(12, -1, -1):
        if counts[r] == 2:
            if pa < 0:
                pa = r
            elif pb < 0:
                pb = r
    if pb >= 0:
        out[0] = pa
        out[1] = pb
        m = 2
        for r in range(12, -1, -1):
            for _ in range(counts[r] if r != pa and r != pb else 0):
                if m < 2 + min(1, n - 4):
                    out[m] = r
                    m += 1
        return _nb_pack(TWO_PAIR, out, m)
    if pa >= 0:
        out[0] = pa
        m = 1
        for r in range(12, -1, -1):
            for _ in range(counts[r] if r != pa else 0):
                if m < 1 + min(3, n - 2):
                    out[m] = r
                    m += 1
        return _nb_pack(PAIR, out, m)
    m = 0
    for r in range(12, -1, -1):
        for _ in range(counts[r]):
            if m < 5:
                out[m] = r
                m += 1
    return _nb_pack(HIGH_CARD, out, m)


@njit(parallel=True, cache=True)
def hand_values(cards: np.ndarray, num_suits: int) -> np.ndarray:
    """hand_value of every row of `cards` (an (N, n) array of card ints), in parallel."""
    N, n = cards.shape
    out = np.empty(N, np.int64)
    for i in prange(N):
        ranks = np.empty(n, np.int64)
        suits = np.empty(n, np.int64)
        for c in range(n):
            ranks[c] = cards[i, c] // num_suits
            suits[c] = cards[i, c] % num_suits
        out[i] = _nb_value(ranks, suits)
    return out
