"""The public tree of a Hold'em game with card-removal streets, for any deck up to 52 cards.

Unlike core/public.build_public_tree, which walks every deal of the game tree, this lists
the public deals directly and gives each street a CardStreet (core/removal.py), so its
cost grows with deals x hands. Fix the flops (`Holdem(flops=...)`) for the real deck:
a bomb pot is one independent game per pair of flops.

With flop, turn and river the river has millions of deals (3.9M per flop pair on 52
cards), so `holdem_public_tree(game, upto=2)` builds only the flop and turn, and
`HoldemRiver` produces any batch of river deals on demand.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from bombpot.core.public import PublicTree, betting
from bombpot.core.removal import CardStreet, HandSet
from bombpot.holdem.evaluator import hand_values
from bombpot.holdem.game import HOLE_CARDS, Holdem


def _pool(game: Holdem) -> list:
    """Cards that can still be dealt after any fixed flops."""
    fixed = set() if game.flops is None else {c for f in game.flops for c in f}
    return [c for c in game.deck.cards if c not in fixed]


def hand_set(game: Holdem) -> tuple[list, HandSet]:
    hands = list(combinations(_pool(game), HOLE_CARDS))
    return hands, HandSet(np.array(hands, dtype=np.int64), game.deck.size)


def holdem_public_tree(game: Holdem, upto: int | None = None) -> PublicTree:
    """The public tree, with streets 0 .. upto - 1 (all of them by default)."""
    T = game.num_streets if upto is None else upto
    pool = _pool(game)
    hands, hs = hand_set(game)
    # Chance weight of one (hand0, hand1) pair before any board card is dealt.
    weight = 1.0 / (comb(len(pool), HOLE_CARDS) * comb(len(pool) - HOLE_CARDS, HOLE_CARDS))

    # Deals, street by street, as (per-board card tuples, chance weight of one valid deal).
    start = tuple(game.flops) if game.flops is not None else ((),) * game.num_boards
    level = [(start, weight)]
    streets = []
    prev_index = None
    for t in range(T):
        k = game.board_cards[t]
        if not (t == 0 and game.flops is not None):
            for j in ([0] if game.deck_mode == "identical" else range(game.num_boards)):
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
            strength = _strengths(game, hs.cards, [boards for _, boards, _ in keyed])
        street = CardStreet(hs, parent, public, kappa, strength, deals=deals)
        streets.append(street)
        prev_index = street.index
        level = [(boards, w) for _, boards, w in keyed]
    return PublicTree(game.name, hands, streets, *betting(game))


def _board_key(boards: tuple) -> tuple:
    """The game's board tuple (LimitPoker.board): card positions interleaved across boards."""
    return tuple(c for cards in zip(*boards) for c in cards)


def _strengths(game: Holdem, hand_cards: np.ndarray, deal_boards: list) -> np.ndarray:
    """(D, B, H) hand values, computed once per distinct board."""
    distinct = sorted({tuple(sorted(cards)) for boards in deal_boards for cards in boards})
    where = {b: i for i, b in enumerate(distinct)}
    values = np.zeros((len(distinct), len(hand_cards)), dtype=np.int64)
    for i, board in enumerate(distinct):
        fixed = np.broadcast_to(np.array(board, dtype=np.int64), (len(hand_cards), len(board)))
        values[i] = hand_values(np.ascontiguousarray(np.hstack([hand_cards, fixed])), game.num_suits)
    out = np.empty((len(deal_boards), game.num_boards, len(hand_cards)), dtype=np.int64)
    for d, boards in enumerate(deal_boards):
        for b, cards in enumerate(boards):
            out[d, b] = values[where[tuple(sorted(cards))]]
    return out


class HoldemRiver:
    """The river of a fixed-flop, shared-deck Hold'em game with flop, turn and river,
    produced on demand.

    The children of turn deal d are the ordered (river A, river B) pairs of cards not yet
    public, numbered lexicographically. A hand's strength on a board depends only on that
    board's turn and river cards, so one table per board, indexed by the unordered
    (turn, river) pair, covers every river deal.
    """

    def __init__(self, game: Holdem, turn: PublicTree):
        if game.flops is None or game.deck_mode != "shared" or tuple(game.board_cards[1:]) != (1, 1):
            raise ValueError("HoldemRiver needs fixed flops, a shared deck and one turn and one river card")
        self.game, self.turn = game, turn.streets[-1]
        self.hands = self.turn.hands
        deals = np.array(self.turn.deals, dtype=np.int64)  # (D1, 2 * flop + 2): boards interleaved
        self.turn_cards = deals[:, -2:]
        self.public = np.sort(deals, axis=1)
        pool = _pool(game)
        self.left = len(pool) - 2  # cards not public once the turns are out
        self.num_children = self.left * (self.left - 1)
        rest = self.left - 2 * HOLE_CARDS  # the private cards come out of the same deck
        self.scale = 1.0 / (rest * (rest - 1))  # river A, then river B
        is_public = np.zeros((len(deals), game.deck.size), dtype=bool)
        np.put_along_axis(is_public, deals, True, axis=1)
        self.remaining = np.array([np.flatnonzero(~row) for row in is_public], dtype=np.int64)  # (D1, left)
        # Strength tables: table[b][pair[c1, c2], hand] for the unordered (turn, river) pair.
        self.pair = np.full((game.deck.size, game.deck.size), -1, dtype=np.int64)
        pairs = list(combinations(pool, 2))
        for i, (a, b) in enumerate(pairs):
            self.pair[a, b] = self.pair[b, a] = i
        cards = self.hands.cards
        self.table = []
        for flop in game.flops:
            rows = np.empty((len(pairs) * len(cards), 2 + len(flop) + 2), dtype=np.int64)
            rows[:, :2] = np.tile(cards, (len(pairs), 1))
            rows[:, 2:2 + len(flop)] = flop
            rows[:, 2 + len(flop):] = np.repeat(np.array(pairs, dtype=np.int64), len(cards), axis=0)
            self.table.append(hand_values(rows, game.num_suits).reshape(len(pairs), len(cards)))

    def river_cards(self, parents: np.ndarray, children: np.ndarray) -> np.ndarray:
        """(n, 2) river A and river B of child `children[n]` of turn deal `parents[n]`."""
        i, j = np.divmod(children, self.left - 1)
        rem = self.remaining[parents]
        a = np.take_along_axis(rem, i[:, None], axis=1)[:, 0]
        j = j + (j >= i)  # skip river A's position
        b = np.take_along_axis(rem, j[:, None], axis=1)[:, 0]
        return np.stack([a, b], axis=1)

    def street(self, parents, children, scale: float = 1.0) -> CardStreet:
        """A CardStreet of the given river deals: child `children[n]` of turn deal `parents[n]`.
        Its `parent` field holds `parents`; chance weights are multiplied by `scale`."""
        parents, children = np.asarray(parents), np.asarray(children)
        river = self.river_cards(parents, children)
        public = np.hstack([self.public[parents], river])
        tc = self.turn_cards[parents]
        strength = np.stack([self.table[b][self.pair[tc[:, b], river[:, b]]] for b in range(2)], axis=1)
        kappa = self.turn.kappa[parents] * self.scale * scale
        return CardStreet(self.hands, parents, public, kappa, strength)

    def all_children(self, parents) -> tuple[np.ndarray, np.ndarray]:
        """Every child of each turn deal in `parents`, as (parents, children) arrays."""
        parents = np.asarray(parents)
        return np.repeat(parents, self.num_children), np.tile(np.arange(self.num_children), len(parents))
