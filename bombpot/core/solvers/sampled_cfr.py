"""Public chance sampling with range vectors, for games whose last street is too big to
enumerate (the river of double-board Hold'em: 3.9 million deals per pair of flops).

Each iteration samples `batch` deals of the second-to-last street (without replacement)
and `samples` children of each on the last street, and runs the range solver's passes
(range_cfr.forward and backward) on that sample. Chance weights are divided by the
probability of being sampled, so sampled counterfactual values are unbiased (Johanson
et al. 2012, "Efficient Nash equilibrium approximation through Monte Carlo
counterfactual regret minimization"). The average strategy is accumulated with the
player's own reach at the sampled infosets only, as in that paper.

The last street's infosets come from an abstraction: `rows(parents, children)` picks the
stored strategy a deal uses, and `ids(street, ...)` each hand's infoset within that row.
`Lossless` gives every deal its own row (an exact game, for checks on small decks);
`abstraction/river.py` shares a row across all river cards of a turn deal and buckets
hands by equity, as poker solvers do for the river.

Regrets and averages use linear weighting (iteration t counts t), as in Pluribus (Brown
and Sandholm 2019), which needs no discounting of the infosets a sample skips.
"""
from __future__ import annotations

import time

import numpy as np
from numba import prange

from bombpot.core.jit import kernel
from bombpot.core.public import PublicTree
from bombpot.core.solvers.cfr import IterativeSolver
from bombpot.core.solvers.range_cfr import (backward, close_values, forward, next_root, street_reach,
                                            street_values)


@kernel
def _update(regret, instant, sigma, cum, own, w):
    """regret += w x instant, cum += w x own x sigma (the strategy just played), then regret
    matching. Rows are infosets."""
    N, A = regret.shape
    for n in prange(N):
        total = 0.0
        for a in range(A):
            cum[n, a] += w * own[n] * sigma[n, a]
            regret[n, a] += w * instant[n, a]
            if regret[n, a] > 0.0:
                total += regret[n, a]
        for a in range(A):
            sigma[n, a] = (max(regret[n, a], 0.0) / total) if total > 0.0 else 1.0 / A


def _matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    total = pos.sum(-1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / regret.shape[-1])


class Lossless:
    """Every last-street deal is its own row and every hand its own infoset: the exact game."""

    def __init__(self, num_parents: int, num_children: int, num_hands: int):
        self.num_children = num_children
        self.num_rows, self.num_ids = num_parents * num_children, num_hands

    def rows(self, parents, children) -> np.ndarray:
        return np.asarray(parents) * self.num_children + np.asarray(children)

    def ids(self, street, parents, children, player: int) -> np.ndarray:
        return np.broadcast_to(np.arange(self.num_ids), (len(parents), self.num_ids))


class SampledRangeCFR(IterativeSolver):

    def __init__(self, pt: PublicTree, last, abstraction, batch: int, samples: int = 1, seed: int = 0):
        """pt holds every street but the last (pt.templates and pt.contrib cover all of them);
        last.street(parents, children, scale) builds any set of last-street deals, and
        last.num_children is how many each deal of the street before has."""
        self.pt, self.last, self.abs = pt, last, abstraction
        self.T = len(pt.templates)
        if pt.num_streets != self.T - 1:
            raise ValueError("pt must hold every street except the last")
        self.batch, self.samples = batch, samples
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.solve_seconds = 0.0
        self.value = [0.0, 0.0]
        H = len(pt.hands)
        self.regret, self.cum = [], []
        for t, tp in enumerate(pt.templates):
            if t < self.T - 1:
                shape = pt.shape(t)
            else:
                shape = (abstraction.num_rows, len(pt.contrib[t]), abstraction.num_ids)
            self.regret.append([np.zeros(shape + (len(a),)) for a in tp.actions])
            self.cum.append([np.zeros(shape + (len(a),)) for a in tp.actions])
        self.sigma = [[_matching(r) for r in row] for row in self.regret]
        self.H = H

    # Sampling -------------------------------------------------------------------------

    def sample(self):
        """Deals for one iteration: `batch` deals of the second-to-last street, and for
        each, `samples` of its children. Returns (positions, parents, children, q1, q2)."""
        D = self.pt.streets[-1].num_deals
        S = np.sort(self.rng.choice(D, size=self.batch, replace=False))
        n = self.last.num_children
        children = np.stack([self.rng.choice(n, size=self.samples, replace=False) for _ in S]).ravel()
        positions = np.repeat(np.arange(len(S)), self.samples)
        return S, positions, children, self.batch / D, self.samples / n

    def view(self, S, positions, children, q1: float, q2: float):
        """The public tree of one sample (chance weights over sampling probabilities), the
        last street's rows, and each player's infoset ids there."""
        pt = self.pt
        streets = list(pt.streets[:-1]) + [pt.streets[-1].subset(S, 1.0 / q1)]
        last = self.last.street(S[positions], children, 1.0 / (q1 * q2))
        last.parent = positions  # index into the sampled deals of the street before
        view = PublicTree(pt.name, pt.hands, streets + [last], pt.templates, pt.contrib)
        rows = self.abs.rows(S[positions], children)
        ids = [np.asarray(self.abs.ids(last, S[positions], children, p)) for p in (0, 1)]
        return view, rows, ids

    def view_sigma(self, sigma, S, rows, ids) -> list:
        """Per-hand strategies on a sample's view."""
        T = self.T
        out = [list(row) for row in sigma[:T - 2]]
        out.append([s[S] for s in sigma[T - 2]])
        tp = self.pt.templates[T - 1]
        out.append([s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[T - 1], tp.players)])
        return out

    # Iterations -------------------------------------------------------------------------

    def iteration(self) -> None:
        start = time.perf_counter()
        self.t += 1
        S, positions, children, q1, q2 = self.sample()
        view, rows, ids = self.view(S, positions, children, q1, q2)
        for p in (0, 1):  # alternating, both on the same sample
            hsig = self.view_sigma(self.sigma, S, rows, ids)
            reach = forward(view, hsig)
            V, R = backward(view, hsig, reach, p)
            self.value[p] = float(V[0][0].sum())
            self._apply(p, float(self.t), view, S, rows, ids[p], reach, R)
        self.solve_seconds += time.perf_counter() - start

    def instant_regrets(self, p: int, S, positions, children, q1, q2) -> list:
        """Player p's sampled instant regrets in storage layout (zero where not sampled), under
        the current strategies: for testing that their expectation is the full-width regret."""
        view, rows, ids = self.view(S, positions, children, q1, q2)
        hsig = self.view_sigma(self.sigma, S, rows, ids)
        reach = forward(view, hsig)
        _, R = backward(view, hsig, reach, p)
        out = [[np.zeros_like(r) for r in row] for row in self.regret]
        T = self.T
        for t, tp in enumerate(self.pt.templates):
            for k, q in enumerate(tp.players):
                if q != p:
                    continue
                if t < T - 2:
                    out[t][k] += R[t][k]
                elif t == T - 2:
                    out[t][k][S] += R[t][k]
                else:
                    touched, sums = self._scatter(R[t][k], rows, ids[p], view.streets[-1].possible(p))
                    out[t][k][touched] += sums
        return out

    def _scatter(self, x: np.ndarray, rows, ids, possible):
        """Per-hand (n, L, H, ...) values summed into their infosets: returns the touched
        rows and their sums, (rows, L, num_ids, ...). Hands that can't be held are left out."""
        n, L, H = x.shape[:3]
        touched, inv = np.unique(rows, return_inverse=True)
        keep = possible.ravel()
        out = np.zeros((len(touched), self.abs.num_ids, L) + x.shape[3:])
        vals = np.moveaxis(x, 2, 1).reshape((n * H, L) + x.shape[3:])[keep]
        np.add.at(out, (np.repeat(inv.ravel(), H)[keep], ids.ravel()[keep]), vals)
        return touched, np.moveaxis(out, 1, 2)

    def _apply(self, p, w, view, S, rows, ids, reach, R) -> None:
        T = self.T
        for t, tp in enumerate(self.pt.templates):
            for k, q in enumerate(tp.players):
                if q != p:
                    continue
                A = len(tp.actions[k])
                own = reach[t][k][p]
                if t < T - 2:
                    reg, sig, cum, inst = self.regret[t][k], self.sigma[t][k], self.cum[t][k], R[t][k]
                elif t == T - 2:
                    reg, sig, cum, inst = (self.regret[t][k][S], self.sigma[t][k][S], self.cum[t][k][S], R[t][k])
                else:  # sum hands into their infosets; own reach sums too (the average weights histories)
                    possible = view.streets[-1].possible(p)
                    touched, inst = self._scatter(R[t][k], rows, ids, possible)
                    _, own = self._scatter(own, rows, ids, possible)
                    reg, sig, cum = self.regret[t][k][touched], self.sigma[t][k][touched], self.cum[t][k][touched]
                reg, sig, cum = np.ascontiguousarray(reg), np.ascontiguousarray(sig), np.ascontiguousarray(cum)
                _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(inst).reshape(-1, A), sig.reshape(-1, A),
                        cum.reshape(-1, A), np.ascontiguousarray(own).reshape(-1), w)
                if t < T - 2:
                    self.regret[t][k], self.sigma[t][k], self.cum[t][k] = reg, sig, cum
                elif t == T - 2:
                    self.regret[t][k][S], self.sigma[t][k][S], self.cum[t][k][S] = reg, sig, cum
                else:
                    self.regret[t][k][touched], self.sigma[t][k][touched], self.cum[t][k][touched] = reg, sig, cum

    def average_strategy(self) -> list:
        out = []
        for row in self.cum:
            out.append([])
            for c in row:
                total = c.sum(-1, keepdims=True)
                out[-1].append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
        return out

    def log_row(self) -> dict:
        """Exact exploitability over every last-street deal: about 2 s on the small test game,
        but about 14 hours (8 cores) for the real deck's 3.9 million river deals."""
        return {"iteration": self.t, "wall_time": self.solve_seconds,
                **exact_exploitability(self.pt, self.last, self.abs, self.average_strategy())}


# Exact evaluation, streaming the last street ------------------------------------------------


def exact_exploitability(pt: PublicTree, last, abstraction, sigma, chunk: int = 256) -> dict:
    """Best responses and value of a stored strategy in the full game, every last-street
    deal included, streamed `chunk` deals at a time so memory stays bounded."""
    T = len(pt.templates)
    D1 = pt.streets[-1].num_deals
    hsig_upper = [list(row) for row in sigma[:T - 1]]
    reach = forward(pt, hsig_upper)
    _, closing = street_reach(pt.templates[T - 2], hsig_upper[T - 2], reach[T - 2][0])
    tp_last = pt.templates[T - 1]
    jobs = [(0, "br"), (1, "br"), (0, "cfr")]
    agg = [np.zeros(pt.shape(T - 2) + (pt.templates[T - 2].num_closes,)) for _ in jobs]
    agg = [np.moveaxis(a, 3, 2) for a in agg]  # (D1, L, closes, H)
    total = D1 * last.num_children
    for lo in range(0, total, chunk):
        flat = np.arange(lo, min(total, lo + chunk))
        parents, children = np.divmod(flat, last.num_children)
        st = last.street(parents, children)
        root = next_root(closing, st.parent)
        rows = abstraction.rows(parents, children)
        ids = [np.asarray(abstraction.ids(st, parents, children, p)) for p in (0, 1)]
        hsig = [s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[T - 1], tp_last.players)]
        nodes, _ = street_reach(tp_last, hsig, root)
        for j, (p, mode) in enumerate(jobs):
            V, _ = street_values(tp_last, st, pt.contrib[T - 1], hsig, nodes, p, mode, True)
            agg[j] += close_values(V[0], st.parent, D1, pt.templates[T - 2].num_closes)
    out = []
    for j, (p, mode) in enumerate(jobs):
        a = agg[j]
        for t in range(T - 2, -1, -1):
            V, _ = street_values(pt.templates[t], pt.streets[t], pt.contrib[t], hsig_upper[t], reach[t], p, mode,
                                 False, a)
            if t > 0:
                a = close_values(V[0], pt.streets[t].parent, pt.streets[t - 1].num_deals,
                                 pt.templates[t - 1].num_closes)
        out.append(float(V[0].sum()))
    br0, br1, value = out
    return {"exploitability": (br0 + br1) / 2, "br_value_p0": br0, "br_value_p1": br1, "game_value": value}
