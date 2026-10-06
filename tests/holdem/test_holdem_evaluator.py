"""R1: the hand evaluator.

Three independent checks: hand-worked examples, the direct algorithm against the best
of every five-card subset, and ACPC's evaluator (through universal_poker) on random
deals from full and short decks.
"""
import itertools
import random

import pytest

from bombpot.holdem.cards import Deck
import numpy as np

from bombpot.holdem.evaluator import (CATEGORY_NAMES, FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT,
                                      STRAIGHT_FLUSH, TRIPS, TWO_PAIR, category, hand_value, hand_values)

pytest.importorskip("pyspiel")
from acpc import acpc_winner  # noqa: E402

FULL = Deck(13, 4)


def value(text: str, deck: Deck = FULL) -> int:
    cards = deck.parse(text.replace(" ", ""))
    return hand_value([deck.rank(c) for c in cards], [deck.suit(c) for c in cards])


@pytest.mark.parametrize("text, expected", [
    ("AhKh QhJhTh", STRAIGHT_FLUSH),
    ("Ah2h 3h4h5h", STRAIGHT_FLUSH),     # steel wheel
    ("As2d 3c4h5s", STRAIGHT),          # wheel
    ("AsKd QcJhTs", STRAIGHT),
    ("7c7d 7h7s2c", QUADS),
    ("AsAd AcKdKs", FULL_HOUSE),
    ("2c3c 4c5c9c", FLUSH),
    ("7c7d 7h2s3c", TRIPS),
    ("KsKd QsQd2c", TWO_PAIR),
    ("KsKd Qs3d2c", PAIR),
    ("Ks9d Qs3d2c", HIGH_CARD),
    ("Ah9c 9d9s4c4d", FULL_HOUSE),      # six cards
    ("2c3d 4h5s6c7d8h", STRAIGHT),      # seven cards
    ("Kc9c 9d9s 9h4c4d", QUADS),        # quads beat the full house also present
    ("AcKc QcJc 9c8c7c", FLUSH),        # no straight flush: the Ten is missing
])
def test_categories(text, expected):
    assert category(value(text)) == expected, CATEGORY_NAMES[category(value(text))]


@pytest.mark.parametrize("better, worse", [
    ("As2d 3c4h5s", "AsKdQcJh9s"),      # the wheel beats ace high
    ("2c3d 4h5s6c", "As2d3c4h5s"),      # a six-high straight beats the wheel
    ("AsAd KcQh2s", "AsAd KcJh9s"),     # pair of aces, second kicker decides
    ("KsKd QsQd 3c", "KsKd QsQd 2c"),   # two pair, kicker decides
    ("KsKd 2s2d Ac", "QsQd JsJd Ac"),   # higher top pair beats a stronger second pair
    ("3c3d 3h 2s2c", "2s2c 2d AhAc"),   # trips in the full house decide
    ("AcKc 9c4c2c", "AdQd JdTd8d"),     # flush: top card decides
    ("9h9d 9s 8c8d 2h", "9h9d 9s 7c7d Ah"),  # six cards: the pair in the full house decides
])
def test_order(better, worse):
    assert value(better) > value(worse)


@pytest.mark.parametrize("a, b", [
    ("AsKd 7c7h2s", "AcKh 7d7s2c"),     # same ranks, different suits
    ("2c3d 9h9s KhKs Qc", "4c5d 9h9s KhKs Qc"),  # the board plays: K K 9 9 Q for both
    ("2c3d 9h9s AhAs 8c", "4c5d 9h9s AhAs 8c"),  # board's two pair with an 8 kicker plays for both
])
def test_ties(a, b):
    assert value(a) == value(b)


def test_few_cards():
    # Fewer than five cards: everything plays, and nothing makes a straight or flush.
    assert category(value("2c3c 4c5c")) == HIGH_CARD
    assert category(value("7c7d 7h")) == TRIPS
    assert value("7c7d 7hAs") > value("7c7d 7hKs")    # trips with a kicker
    assert value("KcKd 4h2s") > value("KcKd 3h2s")    # pair, second kicker
    assert value("KcKd 2h2s") > value("KcKd AhQs")    # two pair beats a pair
    assert category(value("7c7d 7h7s")) == QUADS


def test_direct_algorithm_equals_best_five_card_subset():
    rng = random.Random(0)
    for n in (6, 7):
        for _ in range(3000):
            cards = rng.sample(FULL.cards, n)
            best = max(hand_value([FULL.rank(c) for c in five], [FULL.suit(c) for c in five])
                       for five in itertools.combinations(cards, 5))
            assert hand_value([FULL.rank(c) for c in cards], [FULL.suit(c) for c in cards]) == best, \
                FULL.cards_str(cards)


def test_every_five_card_category_occurs_in_proportion():
    # The known counts of five-card hands out of C(52, 5): high card 1,302,540 down to 40
    # straight flushes. Check on every hand (2.6M), as a whole-ranking sanity check.
    values = hand_values(np.array(list(itertools.combinations(FULL.cards, 5))), 4)
    assert np.bincount(values // 16**5).tolist() == [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]


@pytest.mark.slow
def test_reference_evaluator_equals_compiled_on_every_five_card_hand():
    # With the count above, this ties the reference evaluator to the whole ranking.
    five = np.array(list(itertools.combinations(FULL.cards, 5)))
    want = hand_values(five, 4)
    for row, v in zip(five.tolist(), want.tolist()):
        assert hand_value([c // 4 for c in row], [c % 4 for c in row]) == v


@pytest.mark.parametrize("num_ranks, num_suits", [(13, 4), (6, 3), (5, 2), (4, 4)])
def test_compiled_evaluator_equals_reference(num_ranks, num_suits):
    deck = Deck(num_ranks, num_suits)
    rng = random.Random(num_ranks * num_suits)
    for n in range(1, min(7, deck.size) + 1):
        rows = np.array([rng.sample(deck.cards, n) for _ in range(5000)])
        want = [hand_value([deck.rank(c) for c in r], [deck.suit(c) for c in r]) for r in rows.tolist()]
        assert hand_values(rows, num_suits).tolist() == want


@pytest.mark.parametrize("num_ranks, num_suits, board_len, deals", [
    (13, 4, 3, 4000), (13, 4, 4, 4000), (13, 4, 5, 6000),   # flop, turn, river
    (6, 4, 5, 3000), (5, 3, 3, 3000), (7, 2, 4, 3000),      # short decks: no ace, so no wheel
    (4, 2, 2, 2000), (3, 3, 1, 1000),                       # fewer than five cards
])
def test_agrees_with_acpc(num_ranks, num_suits, board_len, deals):
    deck = Deck(num_ranks, num_suits)
    rng = random.Random(num_ranks * 100 + num_suits * 10 + board_len)
    seen = set()
    for _ in range(deals):
        cards = rng.sample(deck.cards, 4 + board_len)
        h0, h1, board = cards[:2], cards[2:4], cards[4:]
        v0 = hand_value([deck.rank(c) for c in h0 + board], [deck.suit(c) for c in h0 + board])
        v1 = hand_value([deck.rank(c) for c in h1 + board], [deck.suit(c) for c in h1 + board])
        ours = (v0 > v1) - (v0 < v1)
        assert acpc_winner(num_ranks, num_suits, h0, h1, board) == ours, \
            f"{deck.cards_str(h0)} vs {deck.cards_str(h1)} on {deck.cards_str(board)}"
        seen.add(category(max(v0, v1)))
    assert len(seen) >= 3  # the deals exercised several categories


def test_agrees_with_acpc_on_close_calls():
    # Hand-picked spots where a sloppy evaluator goes wrong.
    spots = [
        ("As2d", "KcKh", "3c4h5s9d9c"),   # wheel vs a pair of kings with the board pair
        ("Ah2h", "6c6d", "3h4h5hKsKd"),   # steel wheel vs two pair, kings and sixes
        ("AcKc", "QdQh", "2c7c9cQsQc"),   # flush vs quads
        ("2c2d", "AhKh", "QsQdQhJs3c"),   # full house (Q Q Q 2 2) vs Q Q Q A K
        ("9s8s", "AdAh", "7s6s5d4c2h"),   # nine-high straight vs aces
        ("KdQd", "KhJh", "KsKc7d7h2s"),   # board full house K K K 7 7: tie
        ("Th9h", "AsKs", "QhJh8h2c3d"),   # queen-high straight flush vs ace high
        ("3c3d", "AhAd", "3h3sAsAcKd"),   # quads vs quads
    ]
    for h0, h1, board in spots:
        cards = FULL.parse(h0 + h1 + board)
        v0 = value(h0 + board)
        v1 = value(h1 + board)
        assert acpc_winner(13, 4, cards[:2], cards[2:4], cards[4:]) == (v0 > v1) - (v0 < v1), (h0, h1, board)
