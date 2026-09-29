"""Bucket hands per public state (street, player, board) from their equity features.

Methods, compared at the same bucket count k:
  avg_1d     k-means on (equity A + equity B) / 2, using the mean point on earlier
             streets. The baseline.
  product    1D k-means on each board's equity with sqrt(k) clusters; a bucket is
             the pair. Needs k to be a perfect square.
  kmeans_2d  k-means on the equity point (its mean on earlier streets).
  emd_2d     earlier streets: k-means under earth mover's distance on the
             histograms, with each center the weighted mixture of its members'
             histograms. Last street: kmeans_2d.
  lossless   one bucket per hand.

Every k-means weights hands by probability and keeps the best of `restarts` runs,
each seeded with k-means++. "Best" means the lowest weighted sum of squared
distances to the centers, with EMD as the distance for emd_2d. A single start often
stalls in a poor local optimum on these few points, so restarts matter. Each public
state gets its own random stream derived from `seed`, shared by both players, so a
public state has one bucketing. A public state with at most k distinct features
gives each its own bucket. Buckets are numbered weakest first (by the center's mean
equity).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import ot

from toygames.abstraction.features import Features, HandFeatures, _merge_atoms

METHODS = ("avg_1d", "product", "kmeans_2d", "emd_2d", "lossless")


@dataclass
class Bucketing:
    method: str
    k: int | None
    seed: int
    board_len: list
    table: list  # [t] -> {(player, board): {hand: bucket}}
    restarts: int = 1

    def bucket(self, street: int, player: int, board: tuple, hand) -> int:
        return self.table[street][(player, board[: self.board_len[street]])][hand]

    def bucket_counts(self, street: int) -> np.ndarray:
        return np.array([len(set(m.values())) for m in self.table[street].values()])


def public_state_rng(seed: int, street: int, board: tuple) -> np.random.Generator:
    """A random stream for one public state. Both players share it, so identical
    features give identical buckets, and no state's buckets depend on the others."""
    digest = hashlib.sha256(repr((street, board)).encode()).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "little")])


def make_bucketing(features: Features, method: str, k: int | None, seed: int, restarts: int = 20) -> Bucketing:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "product" and math.isqrt(k) ** 2 != k:
        raise ValueError("product bucketing needs k to be a perfect square")
    table = []
    for t in range(features.num_streets):
        last = t == features.num_streets - 1
        tab = {}
        for (player, board), hf in features.streets[t].items():
            rng = public_state_rng(seed, t, board)
            labels = _labels(hf, method, k, rng, last, restarts)
            tab[(player, board)] = dict(zip(hf.hands, (int(b) for b in labels)))
        table.append(tab)
    return Bucketing(method, k, seed, features.board_len, table, restarts)


def _labels(hf: HandFeatures, method: str, k: int, rng, last: bool, restarts: int) -> np.ndarray:
    w = hf.weights
    if method == "lossless":
        return np.arange(len(hf.hands))
    if method == "avg_1d":
        return kmeans(hf.points.mean(axis=1, keepdims=True), w, k, rng, restarts)
    if method == "kmeans_2d" or (method == "emd_2d" and last):
        return kmeans(hf.points, w, k, rng, restarts)
    if method == "emd_2d":
        return emd_kmeans(hf.atoms, hf.atom_weights, w, k, rng, restarts)
    m = math.isqrt(k)
    labels = np.zeros(len(hf.hands), dtype=np.int64)
    for j in range(hf.points.shape[1]):
        labels = labels * m + kmeans(hf.points[:, j:j + 1], w, m, rng, restarts)
    return labels


# Euclidean k-means ------------------------------------------------------------------


def _canonical(labels: np.ndarray, center_score: np.ndarray) -> np.ndarray:
    """Relabel clusters 0, 1, ... in increasing order of center_score."""
    used = np.unique(labels)
    order = used[np.lexsort((used, center_score[used]))]
    remap = np.empty(labels.max() + 1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    return remap[labels]


def _fill_empty(labels: np.ndarray, cost: np.ndarray, k: int) -> np.ndarray:
    """Give each empty cluster the costliest point from a cluster with at least two points."""
    counts = np.bincount(labels, minlength=k)
    for c in np.flatnonzero(counts == 0):
        movable = np.where(counts[labels] > 1, cost, -1.0)
        i = int(np.argmax(movable))
        counts[labels[i]] -= 1
        labels[i] = c
        counts[c] = 1
    return labels


def kmeans(X: np.ndarray, w: np.ndarray, k: int, rng, restarts: int = 1, max_iter: int = 300) -> np.ndarray:
    """Weighted k-means: the best of `restarts` runs of k-means++ seeding then Lloyd's
    iterations. Returns a label per row of X."""
    uniq, first, inv = np.unique(np.round(X, 12), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    P, W = X[first], np.bincount(inv, weights=w, minlength=len(uniq))
    if len(P) <= k:
        return _canonical(inv, P.mean(axis=1))
    best = None
    for _ in range(restarts):
        labels, C = _lloyd(P, W, k, rng, max_iter)
        cost = float(W @ ((P - C[labels]) ** 2).sum(1))
        if best is None or cost < best[0] - 1e-15:
            best = (cost, labels, C)
    _, labels, C = best
    return _canonical(labels, C.mean(axis=1))[inv]


def _lloyd(P: np.ndarray, W: np.ndarray, k: int, rng, max_iter: int):
    """One k-means++ seeding (weighted by probability) and Lloyd's iterations."""
    chosen = [rng.choice(len(P), p=W / W.sum())]
    d2 = ((P - P[chosen[0]]) ** 2).sum(1)
    for _ in range(1, k):
        prob = W * d2
        i = rng.choice(len(P), p=prob / prob.sum())
        chosen.append(i)
        d2 = np.minimum(d2, ((P - P[i]) ** 2).sum(1))
    C = P[chosen].copy()
    labels = None
    for _ in range(max_iter):
        dist = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        new = _fill_empty(new, W * dist[np.arange(len(P)), new], k)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        C = np.array([np.average(P[labels == c], axis=0, weights=W[labels == c]) for c in range(k)])
    return labels, C


# Earth mover's distance k-means ------------------------------------------------------


def emd(a_pts, a_w, b_pts, b_w) -> float:
    """Earth mover's distance between two weighted point sets, Euclidean ground distance."""
    M = ot.dist(a_pts, b_pts, metric="euclidean")
    return float(ot.emd2(a_w / a_w.sum(), b_w / b_w.sum(), M))


def _mixture(members, weights):
    pts = np.concatenate([a for a, _ in members])
    ws = np.concatenate([aw * w for (_, aw), w in zip(members, weights)])
    return _merge_atoms(pts, ws / ws.sum())


def emd_kmeans(atoms: list, atom_w: list, w: np.ndarray, k: int, rng, restarts: int = 1,
               max_iter: int = 100) -> np.ndarray:
    """k-means over histograms: assign by EMD, recenter as the weighted mixture of the
    members. Keeps the best of `restarts` runs by weighted sum of squared EMD to the centers."""
    n = len(atoms)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = emd(atoms[i], atom_w[i], atoms[j], atom_w[j])
    # Group identical histograms, then cluster the distinct ones with summed weights.
    group = np.full(n, -1)
    reps = []
    for i in range(n):
        if group[i] < 0:
            group[(D[i] < 1e-12) & (group < 0)] = len(reps)
            reps.append(i)
    W = np.bincount(group, weights=w, minlength=len(reps))
    means = np.array([aw @ a for a, aw in zip(atoms, atom_w)]).mean(axis=1)
    if len(reps) <= k:
        return _canonical(group, means[reps])
    Dr = D[np.ix_(reps, reps)]
    hists = [(atoms[r], atom_w[r]) for r in reps]
    best = None
    for _ in range(restarts):
        labels, centers = _emd_lloyd(hists, W, Dr, k, rng, max_iter)
        cost = sum(W[i] * emd(*hists[i], *centers[labels[i]]) ** 2 for i in range(len(reps)))
        if best is None or cost < best[0] - 1e-15:
            best = (cost, labels, centers)
    _, labels, centers = best
    score = np.array([cw @ ca for ca, cw in centers]).mean(axis=1)
    return _canonical(labels, score)[group]


def _emd_lloyd(hists: list, W: np.ndarray, Dr: np.ndarray, k: int, rng, max_iter: int):
    """One k-means++ seeding under EMD and the assign / re-mix iterations."""
    chosen = [rng.choice(len(hists), p=W / W.sum())]
    d = Dr[chosen[0]].copy()
    for _ in range(1, k):
        prob = W * d ** 2
        i = rng.choice(len(hists), p=prob / prob.sum())
        chosen.append(i)
        d = np.minimum(d, Dr[i])
    centers = [hists[c] for c in chosen]
    labels, seen = None, set()
    for _ in range(max_iter):
        dist = np.array([[emd(a, aw, ca, cw) for ca, cw in centers] for a, aw in hists])
        new = dist.argmin(1)
        new = _fill_empty(new, W * dist[np.arange(len(hists)), new] ** 2, k)
        if labels is not None and np.array_equal(new, labels):
            break
        if tuple(new) in seen:  # EMD k-means is not guaranteed to settle; stop at a repeat
            break
        seen.add(tuple(new))
        labels = new
        centers = [_mixture([hists[i] for i in np.flatnonzero(labels == c)], W[labels == c]) for c in range(k)]
    return labels, centers
