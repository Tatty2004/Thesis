"""CFR, CFR+ and DCFR on a public tree, with each player's range as a vector over hands.

The same algorithm as solvers/cfr.py, organized by public node instead of by terminal:
a forward pass carries both players' reach vectors down the public tree, and a backward
pass returns each hand's counterfactual value, where chance and card removal enter only
through each street's `fold` and `share` products (see core/public.py). Every (hand, public node)
pair is one infoset, so with the same updates the iterates equal the tree solver's.

Arrays on street t have shape (D_t, L_t, H, ...): deals, entering lines, hands. Player
p's strategy at template node k is sigma[t][k], shaped (D_t, L_t, H, actions).

Bucketing: `buckets[t][p]` (D_t, H) gives each hand's infoset id at each deal; hands
with the same id share regrets and strategy. Ids must already include the buckets of
earlier streets (perfect recall). Without it every hand is its own infoset.
"""
from __future__ import annotations

import time

import numpy as np
from numba import prange

from bombpot.core.jit import kernel
from bombpot.core.public import PublicTree
from bombpot.core.solvers.cfr import IterativeSolver

VARIANTS = {"cfr": 0, "cfr+": 1, "dcfr": 2}


@kernel
def _expectation(sig, cv, value, regret):
    """value[n] = sum_a sig[n, a] cv[a, n] and regret[n, a] = cv[a, n] - value[n]."""
    N, A = sig.shape
    for n in prange(N):
        v = 0.0
        for a in range(A):
            v += sig[n, a] * cv[a, n]
        value[n] = v
        for a in range(A):
            regret[n, a] = cv[a, n] - v


@kernel
def _update(regret, instant, sigma, cum, own, weight, pos, neg, variant):
    """One node's update for its player, rows = infosets: add the strategy just played to
    the average (weighted by own reach), add the instant regret, apply CFR+'s floor or
    DCFR's discount (positive regrets x pos, negative x neg), then regret matching."""
    N, A = regret.shape
    for n in prange(N):
        w = weight * own[n]
        total = 0.0
        for a in range(A):
            cum[n, a] += w * sigma[n, a]
            r = regret[n, a] + instant[n, a]
            if variant == 1 and r < 0.0:
                r = 0.0
            elif variant == 2:
                r = r * (pos if r > 0.0 else neg)
            regret[n, a] = r
            if r > 0.0:
                total += r
        for a in range(A):
            sigma[n, a] = (max(regret[n, a], 0.0) / total) if total > 0.0 else 1.0 / A


def _segment_sum(values: np.ndarray, parent: np.ndarray, n: int) -> np.ndarray:
    """Sum the rows of `values` (first axis = child deals) into their parent deals."""
    out = np.zeros((n,) + values.shape[1:])
    np.add.at(out, parent, values)
    return out


# Passes over a public tree ----------------------------------------------------------------
# hsig[t][k] is the per-hand strategy at street t's template node k, shaped (D, L, H, A).


def street_reach(tp, hsig_t, root):
    """Forward through one street from its root reach (reach0, reach1): the reach at each
    decision node, and after each line that closes the street."""
    nodes = [None] * tp.num_nodes
    nodes[0] = root
    closing = [None] * tp.num_closes
    for k in range(tp.num_nodes):
        r = nodes[k]
        q = tp.players[k]
        for a, child in enumerate(tp.children[k]):
            c = list(r)
            c[q] = r[q] * hsig_t[k][..., a]
            if child[0] == "node":
                nodes[child[1]] = tuple(c)
            elif child[0] == "close":
                closing[child[1]] = tuple(c)
    return nodes, closing


def next_root(closing, parent: np.ndarray) -> tuple:
    """Reach at the next street's root deals: each deal takes its parent's reach after every
    closing line, so its lines are (previous lines x closing lines)."""
    D, H = len(parent), closing[0][0].shape[-1]
    return tuple(np.stack([c[p][parent] for c in closing], axis=2).reshape(D, -1, H) for p in (0, 1))


def forward(pt: PublicTree, hsig) -> list:
    """Both players' reach at every decision node: reach[t][k] = (reach0, reach1)."""
    D0, _, H = pt.shape(0)
    root = (np.ones((D0, 1, H)), np.ones((D0, 1, H)))
    out = []
    for t, tp in enumerate(pt.templates[:pt.num_streets]):
        nodes, closing = street_reach(tp, hsig[t], root)
        out.append(nodes)
        if t + 1 < pt.num_streets:
            root = next_root(closing, pt.streets[t + 1].parent)
    return out


def close_values(root_values: np.ndarray, parent: np.ndarray, n_parents: int, n_closes: int) -> np.ndarray:
    """The next street's root values summed into their parent deals, per closing line:
    (D_parent, L, closes, H)."""
    D, _, H = root_values.shape
    return _segment_sum(root_values.reshape(D, -1, n_closes, H), parent, n_parents)


def street_values(tp, st, contrib, hsig_t, reach_t, p: int, mode: str, last: bool, agg=None):
    """Player p's values at each node of one street (bottom-up), and instant regrets at p's
    nodes in mode "cfr". `agg` holds the closing lines' values when the street isn't last."""
    o = 1 - p
    sign = 1.0 if p == 0 else -1.0
    contrib = contrib[None, :, None]
    V = [None] * tp.num_nodes
    R = [None] * tp.num_nodes
    for k in range(tp.num_nodes - 1, -1, -1):
        q = tp.players[k]
        rk = reach_t[k]
        cv = np.empty((len(tp.actions[k]),) + rk[o].shape)
        for a, child in enumerate(tp.children[k]):
            ro = rk[o] * hsig_t[k][..., a] if q == o else rk[o]
            if child[0] == "node":
                v = V[child[1]]
            elif child[0] == "fold":
                _, folder, extra = child
                lost = contrib + extra
                u0 = -lost if folder == 0 else lost
                v = sign * u0 * st.fold(ro, p)
            elif last:  # showdown: player 0 wins pot x share - chips; player 1 the negative
                chips = contrib + tp.closes[child[1]][1]
                v = sign * chips * (2 * st.share(ro, p) - st.fold(ro, p))
            else:
                v = agg[:, :, child[1], :]
            cv[a] = v
        if q != p:
            V[k] = cv.sum(0)
        elif mode == "br":
            V[k] = cv.max(0)
        else:
            A = cv.shape[0]
            V[k] = np.empty(cv.shape[1:])
            R[k] = np.empty(cv.shape[1:] + (A,))
            _expectation(cv.size, np.ascontiguousarray(hsig_t[k]).reshape(-1, A), cv.reshape(A, -1),
                         V[k].reshape(-1), R[k].reshape(-1, A))
    return V, R


def backward(pt: PublicTree, hsig, reach, p: int, mode: str = "cfr"):
    """Player p's counterfactual value of each hand at every decision node, bottom-up.

    mode "cfr": p plays hsig; returns (values, regrets at p's nodes).
    mode "br":  p best-responds; returns (values, None for every node).
    """
    T = pt.num_streets
    all_values, all_regrets = [None] * T, [None] * T
    agg = None
    for t in range(T - 1, -1, -1):
        tp, st = pt.templates[t], pt.streets[t]
        V, R = street_values(tp, st, pt.contrib[t], hsig[t], reach[t], p, mode, t == len(pt.templates) - 1, agg)
        all_values[t], all_regrets[t] = V, R
        if t > 0:
            agg = close_values(V[0], st.parent, pt.streets[t - 1].num_deals, pt.templates[t - 1].num_closes)
    return all_values, all_regrets


class RangeCFR(IterativeSolver):

    def __init__(self, pt: PublicTree, variant: str = "dcfr", alpha: float = 1.5, beta: float = 0.0,
                 gamma: float = 2.0, buckets=None):
        if variant not in ("cfr", "cfr+", "dcfr"):
            raise ValueError(f"unknown variant {variant!r}")
        self.pt, self.variant = pt, variant
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.t = 0
        self.solve_seconds = 0.0
        self.value = [0.0, 0.0]
        self.buckets = buckets
        self.regret, self.cum = [], []
        for t, tp in enumerate(pt.templates):
            D, L, H = pt.shape(t)
            self.regret.append([np.zeros((D, L, self._ids(t, p), len(a))) for p, a in zip(tp.players, tp.actions)])
            self.cum.append([np.zeros_like(r) for r in self.regret[t]])
        self.sigma = [[self._strategy(r) for r in rs] for rs in self.regret]

    # Infosets -----------------------------------------------------------------------

    def _ids(self, t: int, p: int) -> int:
        """Infoset slots per (deal, line) on street t for player p."""
        if self.buckets is None:
            return len(self.pt.hands)
        return int(self.buckets[t][p].max()) + 1

    def _to_hands(self, t: int, p: int, x: np.ndarray) -> np.ndarray:
        """Per-infoset array (D, L, ids, ...) -> per-hand array (D, L, H, ...)."""
        if self.buckets is None:
            return x
        b = self.buckets[t][p]
        D = b.shape[0]
        return x[np.arange(D)[:, None], :, b].transpose(0, 2, 1, *range(3, x.ndim))

    def _to_ids(self, t: int, p: int, x: np.ndarray, n_ids: int) -> np.ndarray:
        """Per-hand array (D, L, H, ...) -> per-infoset sums (D, L, ids, ...)."""
        if self.buckets is None:
            return x
        b = self.buckets[t][p]
        D, L, H = x.shape[:3]
        out = np.zeros((D, n_ids, L) + x.shape[3:])
        np.add.at(out, (np.repeat(np.arange(D), H), b.ravel()), x.transpose(0, 2, 1, *range(3, x.ndim)).reshape(
            (D * H, L) + x.shape[3:]))
        return out.transpose(0, 2, 1, *range(3, x.ndim))

    @staticmethod
    def _strategy(regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0)
        total = pos.sum(-1, keepdims=True)
        n = regret.shape[-1]
        return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / n)

    def hand_sigma(self, sigma=None) -> list:
        """The strategy per hand: sigma[t][k] shaped (D, L, H, actions)."""
        sigma = self.sigma if sigma is None else sigma
        return [[self._to_hands(t, p, s) for p, s in zip(self.pt.templates[t].players, sigma[t])]
                for t in range(self.pt.num_streets)]

    def average_strategy(self) -> list:
        out = []
        for t, cums in enumerate(self.cum):
            row = []
            for c in cums:
                total = c.sum(-1, keepdims=True)
                row.append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
            out.append(row)
        return out

    # Passes -------------------------------------------------------------------------

    def reach(self, hsig) -> list:
        return forward(self.pt, hsig)

    def values(self, p: int, hsig, reach, mode: str = "cfr"):
        return backward(self.pt, hsig, reach, p, mode)

    # Iterations ---------------------------------------------------------------------

    def iteration(self) -> None:
        start = time.perf_counter()
        self.t += 1
        t_ = self.t
        weight = {"cfr": 1.0, "cfr+": float(t_), "dcfr": float(t_) ** self.gamma}[self.variant]
        pos = t_ ** self.alpha / (t_ ** self.alpha + 1.0)  # DCFR discounts (solvers/cfr.dcfr_discount)
        neg = t_ ** self.beta / (t_ ** self.beta + 1.0)
        for p in (0, 1):  # alternating: player 1's update sees player 0's new strategy
            reach, regrets = self.instant_regrets(p)
            for t, k in self._nodes(p):
                A = self.regret[t][k].shape[-1]
                self.sigma[t][k] = np.ascontiguousarray(self.sigma[t][k])  # updated in place below
                own = np.ascontiguousarray(self.own_reach(t, k, reach)).reshape(-1)
                _update(self.regret[t][k].size, self.regret[t][k].reshape(-1, A),
                        np.ascontiguousarray(regrets[t][k]).reshape(-1, A),
                        self.sigma[t][k].reshape(-1, A), self.cum[t][k].reshape(-1, A), own, weight, pos, neg,
                        VARIANTS[self.variant])
        self.solve_seconds += time.perf_counter() - start

    def _nodes(self, p: int):
        return [(t, k) for t, tp in enumerate(self.pt.templates) for k, q in enumerate(tp.players) if q == p]

    def instant_regrets(self, p: int):
        """Both players' reach and player p's instant regret per infoset, under the current strategies."""
        hsig = self.hand_sigma()
        reach = self.reach(hsig)
        V, R = self.values(p, hsig, reach)
        self.value[p] = float(V[0][0].sum())
        regrets = [[None] * tp.num_nodes for tp in self.pt.templates]
        for t, k in self._nodes(p):
            regrets[t][k] = self._to_ids(t, p, R[t][k], self.regret[t][k].shape[2])
        return reach, regrets

    def own_reach(self, t: int, k: int, reach) -> np.ndarray:
        """The acting player's own reach of each infoset at node k, shaped (D, L, ids, 1). Every
        hand in an infoset has the same own reach (perfect recall), so it's their mean."""
        p = self.pt.templates[t].players[k]
        r = reach[t][k][p]
        if self.buckets is None:
            return r[..., None]
        n_ids = self.regret[t][k].shape[2]
        return (self._to_ids(t, p, r, n_ids) / np.maximum(self._counts(t, p, n_ids), 1)[:, None, :])[..., None]

    def _counts(self, t: int, p: int, n_ids: int) -> np.ndarray:
        """Hands in each infoset at each deal, (D, ids)."""
        b = self.buckets[t][p]
        out = np.zeros((b.shape[0], n_ids))
        np.add.at(out, (np.repeat(np.arange(b.shape[0]), b.shape[1]), b.ravel()), 1.0)
        return out

    # Evaluation ---------------------------------------------------------------------

    def exploitability(self, sigma=None) -> dict:
        """Best responses and game value of a profile (per-infoset sigma; the average by default)."""
        hsig = self.hand_sigma(self.average_strategy() if sigma is None else sigma)
        reach = self.reach(hsig)
        br = [float(self.values(p, hsig, reach, "br")[0][0][0].sum()) for p in (0, 1)]
        value = float(self.values(0, hsig, reach)[0][0][0].sum())
        return {"exploitability": (br[0] + br[1]) / 2, "br_value_p0": br[0], "br_value_p1": br[1],
                "game_value": value}

    def log_row(self) -> dict:
        return {"iteration": self.t, "wall_time": self.solve_seconds, **self.exploitability()}


def infoset_count(pt: PublicTree) -> int:
    """Infosets of both players: (hand, public node) pairs where the hand can be held."""
    return int(sum(st.possible(p).sum() * pt.num_lines(t) * sum(q == p for q in pt.templates[t].players)
                   for t, st in enumerate(pt.streets) for p in (0, 1)))


def save_range_strategy(path, sigma, **info) -> None:
    """Save a range strategy (sigma[t][k] arrays) and `info` as a compressed .npz."""
    import json
    from pathlib import Path

    arrays = {f"t{t}_k{k}": s for t, row in enumerate(sigma) for k, s in enumerate(row)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, info=np.array(json.dumps(info, default=str)), **arrays)


def load_range_strategy(path) -> tuple[list, dict]:
    import json

    with np.load(path) as z:
        info = json.loads(str(z["info"]))
        keys = sorted((k for k in z.files if k != "info"), key=lambda k: tuple(int(x[1:]) for x in k.split("_")))
        sigma: list = []
        for key in keys:
            t, k = (int(x[1:]) for x in key.split("_"))
            while len(sigma) <= t:
                sigma.append([])
            sigma[t].append(z[key])
    return sigma, info
