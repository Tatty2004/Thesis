"""Card removal: the range solver's fold and showdown sums in O(hands) per deal.

Private hands are sets of h cards. At deal d, W[d, i, j] = kappa[d] when hands i and j
are disjoint and both avoid the deal's public cards, and 0 otherwise. By
inclusion-exclusion over the subsets S of hand i (the empty set and i itself included),

    sum over j disjoint from i of r[j]  =  sum_S (-1)^|S| R_S,    R_S = sum of r[j] over hands j that contain S,

so a fold sum needs only the per-subset totals R_S. A showdown sum restricts j to hands
weaker than i on a board, plus half of the equally strong ones: sweeping the hands in
order of strength, R_S over the hands seen so far gives the weaker part, and R_S over
the hand's own tie group the equal part. (Hand i is in its own tie group, and the
S = i term removes it again.) Johanson et al. (2011), "Accelerating best response
calculation in large extensive games", use this sweep for two-card hands.

The sweeps are compiled with numba and run in parallel over deals.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numba import prange

from bombpot.core.jit import kernel
from bombpot.core.public import Street


@kernel
def _fold_kernel(reach, valid, kappa, subs, signs, n_subsets, out):
    D, L, H = reach.shape
    K = subs.shape[1]
    for d in prange(D):
        total = np.zeros(n_subsets)
        for l in range(L):
            total[:] = 0.0
            for i in range(H):
                r = reach[d, l, i]
                if valid[d, i] and r != 0.0:
                    for k in range(K):
                        total[subs[i, k]] += r
            for i in range(H):
                v = 0.0
                if valid[d, i]:
                    for k in range(K):
                        v += signs[k] * total[subs[i, k]]
                out[d, l, i] = v * kappa[d]


@kernel
def _beats_kernel(reach, valid, kappa, order, group_end, subs, signs, n_subsets, out):
    D, L, H = reach.shape
    B = order.shape[1]
    K = subs.shape[1]
    for d in prange(D):
        weaker = np.zeros(n_subsets)
        tied = np.zeros(n_subsets)
        for l in range(L):
            for i in range(H):
                out[d, l, i] = 0.0
            for b in range(B):
                weaker[:] = 0.0
                q = 0
                while q < H:
                    e = group_end[d, b, q]
                    for u in range(q, e):
                        i = order[d, b, u]
                        r = reach[d, l, i]
                        if valid[d, i] and r != 0.0:
                            for k in range(K):
                                tied[subs[i, k]] += r
                    for u in range(q, e):
                        i = order[d, b, u]
                        w = 0.0
                        t = 0.0
                        for k in range(K):
                            w += signs[k] * weaker[subs[i, k]]
                            t += signs[k] * tied[subs[i, k]]
                        out[d, l, i] += w + 0.5 * t
                    for u in range(q, e):  # the group joins the weaker hands for what follows
                        i = order[d, b, u]
                        for k in range(K):
                            s = subs[i, k]
                            weaker[s] += tied[s]
                            tied[s] = 0.0
                    q = e
            for i in range(H):
                out[d, l, i] = out[d, l, i] / B * kappa[d] if valid[d, i] else 0.0


class HandSet:
    """Private hands as sets of cards, with every subset of every hand numbered. One per
    game, shared by its streets."""

    def __init__(self, hand_cards, num_cards: int):
        self.cards = np.asarray(hand_cards, dtype=np.int64)  # (H, h)
        self.num_cards = int(num_cards)
        h = self.cards.shape[1]
        ids: dict = {}
        cols = [c for s in range(h + 1) for c in combinations(range(h), s)]
        self.subs = np.array([[ids.setdefault(tuple(row[j] for j in c), len(ids)) for c in cols]
                              for row in self.cards.tolist()], dtype=np.int64)  # (H, 2^h) subset ids
        self.signs = np.array([(-1.0) ** len(c) for c in cols])
        self.n_subsets = len(ids)

    def __len__(self) -> int:
        return len(self.cards)

    def valid(self, public_cards) -> np.ndarray:
        """(D, H): which hands avoid each deal's public cards ((D, n) array)."""
        pc = np.asarray(public_cards, dtype=np.int64)
        public = np.zeros((pc.shape[0], self.num_cards), dtype=bool)
        if pc.size:
            np.put_along_axis(public, pc.reshape(pc.shape[0], -1), True, axis=1)
        return np.ascontiguousarray(~public[:, self.cards].any(axis=2))


class CardStreet(Street):

    def __init__(self, hands: HandSet, parent, public_cards, kappa, strength=None, deals=None, valid=None):
        """
        hands         the game's HandSet
        parent        (D,) parent deal on the previous street (-1 on street 0)
        public_cards  (D, n) the public cards dealt so far at each deal
        kappa         (D,) chance weight of each valid (hand0, hand1, deal)
        strength      (D, B, H) each hand's strength on each board (last street only);
                      bigger wins, equal ties
        deals         [d] keys in the game's board format, for `index` (optional)
        """
        self.hands = hands
        self.parent = np.asarray(parent)
        self.kappa = np.asarray(kappa, dtype=float)
        self.deals = deals
        self.index = {} if deals is None else {b: i for i, b in enumerate(deals)}
        self.valid = hands.valid(public_cards) if valid is None else valid
        self.order = self.group_end = None
        if strength is not None:
            strength = np.asarray(strength, dtype=np.int64)
            self.order = np.argsort(strength, axis=2, kind="stable")
            s = np.take_along_axis(strength, self.order, axis=2)
            pos = np.broadcast_to(np.arange(s.shape[2]), s.shape)
            last = np.ones(s.shape, dtype=bool)
            last[..., :-1] = s[..., 1:] != s[..., :-1]
            self.group_end = np.minimum.accumulate(np.where(last, pos + 1, s.shape[2])[..., ::-1], axis=2)[..., ::-1].copy()

    @property
    def num_deals(self) -> int:
        return len(self.kappa)

    def subset(self, idx, scale: float = 1.0) -> "CardStreet":
        """The street restricted to deals `idx`, with chance weights times `scale` (an
        importance weight when the deals are a sample)."""
        out = CardStreet.__new__(CardStreet)
        out.hands, out.parent = self.hands, self.parent[idx]
        out.kappa = self.kappa[idx] * scale
        out.deals = None if self.deals is None else [self.deals[i] for i in idx]
        out.index = {} if out.deals is None else {b: i for i, b in enumerate(out.deals)}
        out.valid = self.valid[idx]
        out.order = None if self.order is None else self.order[idx]
        out.group_end = None if self.group_end is None else self.group_end[idx]
        return out

    def fold(self, reach, player):
        out = np.empty(reach.shape)
        h = self.hands
        _fold_kernel(reach.size * h.subs.shape[1], np.ascontiguousarray(reach, dtype=np.float64), self.valid,
                     self.kappa, h.subs, h.signs, h.n_subsets, out)
        return out

    def _beats(self, reach, order, group_end):
        r = np.ascontiguousarray(reach, dtype=np.float64)
        beats = np.empty(r.shape)
        h = self.hands
        _beats_kernel(r.size * order.shape[1] * h.subs.shape[1], r, self.valid, self.kappa, order, group_end,
                      h.subs, h.signs, h.n_subsets, beats)
        return beats

    def share(self, reach, player):
        """Player 0's share-weighted sum (see core/public.Street); player 1's is the fold
        sum minus player 1's own sum against player 0, since the two shares add up to 1."""
        if self.order is None:
            raise ValueError("this street has no showdown")
        beats = self._beats(reach, self.order, self.group_end)
        return beats if player == 0 else self.fold(reach, player) - beats

    def equity(self) -> np.ndarray:
        """(D, H, B): each hand's share of each board against a uniform range of opponent
        hands (those that avoid its cards and the deal's), 0 where it can't be held."""
        ones = np.ones((self.num_deals, 1, len(self.hands)))
        total = self.fold(ones, 0)[:, 0]
        out = np.zeros((self.num_deals, len(self.hands), self.order.shape[1]))
        for b in range(self.order.shape[1]):
            beats = self._beats(ones, np.ascontiguousarray(self.order[:, b:b + 1]),
                                np.ascontiguousarray(self.group_end[:, b:b + 1]))[:, 0]
            out[:, :, b] = np.divide(beats, total, out=np.zeros_like(beats), where=total > 0)
        return out

    def possible(self, player):
        return self.valid
