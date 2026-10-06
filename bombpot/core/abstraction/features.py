"""Equity features for bucketing, computed only through the CardGame protocol.

Each hand is measured against a uniform opponent range, meaning the opponent's
hand is drawn from the deal model given our hand and the public board (card
removal only, no betting). With a shared deck that is uniform over the remaining
cards.

Last street: the hand's point, its equity on each board.
Earlier streets: a histogram over next-street points, one atom per possible next
deal weighted by its probability given our hand. Its mean is the expected point.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class HandFeatures:
    hands: list                # hands the player can hold at this public state
    mass: np.ndarray           # P(player holds hand, board)
    points: np.ndarray         # (n, num_boards): equity point, or histogram mean on earlier streets
    atoms: list | None = None  # earlier streets: per hand, (m, num_boards) next-street points
    atom_weights: list | None = None  # per hand, (m,) probabilities of those next deals

    @property
    def weights(self) -> np.ndarray:
        return self.mass / self.mass.sum()


@dataclass
class Features:
    num_streets: int
    num_boards: int
    board_len: list   # length of the board tuple on each street
    streets: list     # [t] -> {(player, board): HandFeatures}


def _merge_atoms(points: np.ndarray, weights: np.ndarray):
    """Combine atoms at the same point (the histogram is unchanged)."""
    uniq, inv = np.unique(np.round(points, 12), axis=0, return_inverse=True)
    return uniq, np.bincount(inv.ravel(), weights=weights, minlength=len(uniq))


def compute_features(game) -> Features:
    T, B = game.num_streets, game.num_boards
    joint = []  # [t] -> {board: {(hand0, hand1): prob}}
    for t in range(T):
        d = defaultdict(lambda: defaultdict(float))
        for board, h0, h1, q in game.deals(t):
            d[board][(h0, h1)] += q
        joint.append(d)
    board_len = [len(next(iter(joint[t]))) for t in range(T)]
    streets = [dict() for _ in range(T)]

    for board, pairs in joint[T - 1].items():
        for p in (0, 1):
            mass, eq = defaultdict(float), defaultdict(lambda: np.zeros(B))
            for (h0, h1), q in pairs.items():
                share0 = game.showdown(board, h0, h1)
                own, share = (h0, share0) if p == 0 else (h1, 1.0 - share0)
                mass[own] += q
                eq[own] += q * share
            hands = sorted(mass)
            m = np.array([mass[h] for h in hands])
            streets[T - 1][(p, board)] = HandFeatures(hands, m, np.array([eq[h] / mass[h] for h in hands]))

    for t in range(T - 2, -1, -1):
        children = defaultdict(list)
        for nb in joint[t + 1]:
            children[nb[:board_len[t]]].append(nb)
        for board, pairs in joint[t].items():
            for p in (0, 1):
                mass = defaultdict(float)
                for (h0, h1), q in pairs.items():
                    mass[h0 if p == 0 else h1] += q
                hands = sorted(mass)
                atoms, atom_w, means = [], [], []
                for x in hands:
                    pts, ws = [], []
                    for nb in children[board]:
                        nf = streets[t + 1][(p, nb)]
                        if x in nf.hands:
                            i = nf.hands.index(x)
                            pts.append(nf.points[i])
                            ws.append(nf.mass[i] / mass[x])  # P(next deal | hand, board)
                    ws = np.array(ws)
                    if abs(ws.sum() - 1.0) > 1e-9:
                        raise ValueError(f"next-deal probabilities sum to {ws.sum()} at {board}, hand {x}")
                    pts = np.array(pts)
                    means.append(ws @ pts)
                    a, w = _merge_atoms(pts, ws)
                    atoms.append(a)
                    atom_w.append(w)
                m = np.array([mass[h] for h in hands])
                streets[t][(p, board)] = HandFeatures(hands, m, np.array(means), atoms, atom_w)
    return Features(T, B, board_len, streets)
