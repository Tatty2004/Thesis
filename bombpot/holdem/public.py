"""The public tree of a Hold'em game with card-removal streets, for any deck up to 52 cards.

Unlike core/public.build_public_tree, which walks every deal of the game tree, this lists
the public deals directly and gives each street a CardStreet (core/removal.py), so its
cost grows with deals x hands. Fix the flops (`Holdem(flops=...)`) for the real deck:
a bomb pot is one independent game per pair of flops.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from bombpot.core.public import PublicTree, betting
from bombpot.core.removal import CardStreet
from bombpot.holdem.game import HOLE_CARDS, Holdem


def holdem_public_tree(game: Holdem) -> PublicTree:
    deck = game.deck
    fixed = [] if game.flops is None else sorted({c for f in game.flops for c in f})
    pool = [c for c in deck.cards if c not in fixed]
    hands = list(combinations(pool, HOLE_CARDS))
    hand_cards = np.array(hands, dtype=np.int64)
    # Chance weight of one (hand0, hand1) pair before any board card is dealt.
    weight = 1.0 / (comb(len(pool), HOLE_CARDS) * comb(len(pool) - HOLE_CARDS, HOLE_CARDS))

    # Deals, street by street, as (per-board card tuples, chance weight of one valid deal).
    start = tuple(game.flops) if game.flops is not None else ((),) * game.num_boards
    level = [(start, weight)]
    streets = []
    prev_index = None
    for t, k in enumerate(game.board_cards):
        if not (t == 0 and game.flops is not None):
            boards_dealt = [0] if game.deck_mode == "identical" else range(game.num_boards)
            for j in boards_dealt:
                nxt = []
                for boards, w in level:
                    used = {c for b in (boards[:1] if game.deck_mode == "identical" else boards) for c in b}
                    avail = [c for c in pool if c not in used]
                    q = w / comb(len(avail) - 2 * HOLE_CARDS, k)
                    for cards in combinations(avail, k):
                        if game.deck_mode == "identical":
                            nb = tuple(b + cards for b in boards)
                        else:
                            nb = tuple(b + cards if i == j else b for i, b in enumerate(boards))
                        nxt.append((nb, q))
                level = nxt
        keyed = sorted(((_board_key(b), b, w) for b, w in level), key=lambda x: x[0])
        deals = [key for key, _, _ in keyed]
        if t == 0:
            parent = np.full(len(deals), -1, dtype=np.int64)
        else:
            n = len(streets[-1].deals[0])
            parent = np.array([prev_index[key[:n]] for key in deals], dtype=np.int64)
        public = np.array([sorted({c for b in boards for c in b}) for _, boards, _ in keyed], dtype=np.int64)
        kappa = np.array([w for _, _, w in keyed])
        strength = None
        if t == game.num_streets - 1:
            strength = _strengths(game, hand_cards, [boards for _, boards, _ in keyed])
        street = CardStreet(deals, parent, hand_cards, public, kappa, strength, num_cards=deck.size)
        streets.append(street)
        prev_index = street.index
        level = [(boards, w) for _, boards, w in keyed]
    return PublicTree(game.name, hands, streets, *betting(game))


def _board_key(boards: tuple) -> tuple:
    """The game's board tuple (LimitPoker.board): card positions interleaved across boards."""
    return tuple(c for cards in zip(*boards) for c in cards)


def _strengths(game: Holdem, hand_cards: np.ndarray, deal_boards: list) -> np.ndarray:
    """(D, B, H) hand values, computed once per distinct board."""
    cache: dict = {}
    out = np.zeros((len(deal_boards), game.num_boards, len(hand_cards)), dtype=np.int64)
    for d, boards in enumerate(deal_boards):
        for b, cards in enumerate(boards):
            key = tuple(sorted(cards))
            if key not in cache:
                cache[key] = np.array([game.hand_value(tuple(h) + key) if not set(h) & set(key) else 0
                                       for h in hand_cards.tolist()], dtype=np.int64)
            out[d, b] = cache[key]
    return out
