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
from numba import njit, prange

from bombpot.core.public import Street


@njit(parallel=True, cache=True)
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


@njit(parallel=True, cache=True)
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


class CardStreet(Street):

    def __init__(self, deals, parent, hand_cards, public_cards, kappa, strength=None, num_cards=None):
        """
        deals         [d] public cards in the game's board format (keys for `index`)
        parent        [d] parent deal on the previous street (-1 on street 0)
        hand_cards    (H, h) the cards of each private hand
        public_cards  (D, n) the public cards dealt so far at each deal
        kappa         (D,) chance weight of each valid (hand0, hand1, deal)
        strength      (D, B, H) each hand's strength on each board (last street only);
                      bigger wins, equal ties
        """
        self.deals, self.parent = deals, np.asarray(parent)
        self.index = {b: i for i, b in enumerate(deals)}
        hand_cards = np.asarray(hand_cards, dtype=np.int64)
        H, h = hand_cards.shape
        C = int(num_cards if num_cards is not None else hand_cards.max() + 1)
        public = np.zeros((len(deals), C), dtype=bool)
        pc = np.asarray(public_cards, dtype=np.int64).reshape(len(deals), -1)
        if pc.size:
            np.put_along_axis(public, pc, True, axis=1)
        self.valid = ~public[:, hand_cards].any(axis=2)  # (D, H)
        self.kappa = np.asarray(kappa, dtype=float)

        # Every subset of every hand, numbered: subs[i, k] is the id of hand i's k-th subset.
        ids: dict = {}
        cols = [c for s in range(h + 1) for c in combinations(range(h), s)]
        self.subs = np.array([[ids.setdefault(tuple(row[j] for j in c), len(ids)) for c in cols]
                              for row in hand_cards.tolist()], dtype=np.int64)
        self.signs = np.array([(-1.0) ** len(c) for c in cols])
        self.n_subsets = len(ids)

        self.order = self.group_end = None
        if strength is not None:
            strength = np.asarray(strength, dtype=np.int64)
            self.order = np.argsort(strength, axis=2, kind="stable")
            s = np.take_along_axis(strength, self.order, axis=2)
            pos = np.broadcast_to(np.arange(H), s.shape)
            last = np.ones(s.shape, dtype=bool)
            last[..., :-1] = s[..., 1:] != s[..., :-1]
            self.group_end = np.minimum.accumulate(np.where(last, pos + 1, H)[..., ::-1], axis=2)[..., ::-1].copy()

    def fold(self, reach, player):
        out = np.empty(reach.shape)
        _fold_kernel(np.ascontiguousarray(reach, dtype=np.float64), self.valid, self.kappa, self.subs, self.signs,
                     self.n_subsets, out)
        return out

    def share(self, reach, player):
        """Player 0's share-weighted sum (see core/public.Street); player 1's is the fold
        sum minus player 1's own sum against player 0, since the two shares add up to 1."""
        if self.order is None:
            raise ValueError("this street has no showdown")
        r = np.ascontiguousarray(reach, dtype=np.float64)
        beats = np.empty(r.shape)
        _beats_kernel(r, self.valid, self.kappa, self.order, self.group_end, self.subs, self.signs, self.n_subsets,
                      beats)
        return beats if player == 0 else self.fold(r, player) - beats

    def possible(self, player):
        return self.valid
