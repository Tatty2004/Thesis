"""A last-street abstraction that forgets the last cards, as poker solvers do for the river.

Every child of a parent deal (every river card after a turn deal) shares that parent's
stored strategy, so the row of a deal is its parent. A hand's id within the row is its
equity bucket: the nearest of k centers, fitted for that parent, to the hand's equity on
each board against a uniform opponent range. Centers are numbered weakest first, so an
id means about the same thing whichever river card came. Hands that can't be held at a
deal get a spare id. This is imperfect recall: the river infoset forgets both the river
cards and the exact hand.

Methods:
  avg_1d     buckets on the mean equity over boards (the 1D baseline)
  kmeans_2d  buckets on the point (equity A, equity B)
"""
from __future__ import annotations

import numpy as np

from bombpot.core.abstraction.bucketing import kmeans

METHODS = ("avg_1d", "kmeans_2d")


def features(street, method: str) -> np.ndarray:
    """(D, H, dim) equity features of every hand at every deal of a last street."""
    eq = street.equity()
    return eq.mean(axis=2, keepdims=True) if method == "avg_1d" else eq


class EquityBuckets:

    def __init__(self, centers: np.ndarray, method: str):
        """centers: (num_parents, k, dim), weakest first."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.centers, self.method = centers, method
        self.num_rows, self.k = centers.shape[:2]
        self.num_ids = self.k + 1

    def rows(self, parents, children) -> np.ndarray:
        return np.asarray(parents)

    def ids(self, street, parents, children, player: int) -> np.ndarray:
        x = features(street, self.method)
        c = self.centers[np.asarray(parents)]
        d2 = ((x[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1)
        return np.where(street.possible(player), d2.argmin(-1), self.k)


def fit_equity_buckets(last, num_parents: int, method: str, k: int, samples: int, seed: int = 0,
                       restarts: int = 3) -> EquityBuckets:
    """Fit each parent's centers by weighted k-means on the features of `samples` of its
    children (every child when there are fewer)."""
    rng = np.random.default_rng(seed)
    dim = 1 if method == "avg_1d" else 2
    centers = np.zeros((num_parents, k, dim))
    for parent in range(num_parents):
        m = min(samples, last.num_children)
        children = rng.choice(last.num_children, size=m, replace=False)
        street = last.street(np.full(m, parent), children)
        x = features(street, method)[street.possible(0)]  # (points, dim)
        labels = kmeans(x, np.ones(len(x)), k, rng, restarts)
        found = np.array([x[labels == c].mean(axis=0) for c in range(labels.max() + 1)])
        found = found[np.argsort(found.mean(axis=1), kind="stable")]
        # Fewer distinct points than k: repeat the strongest center (ids past it stay unused).
        centers[parent] = np.vstack([found, np.repeat(found[-1:], k - len(found), axis=0)])
    return EquityBuckets(centers, method)
