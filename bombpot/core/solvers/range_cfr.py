"""CFR, CFR+ and DCFR on a public tree, with each player's range as a vector over hands.

The same algorithm as solvers/cfr.py, organized by public node instead of by terminal:
a forward pass carries both players' reach vectors down the public tree, and a backward
pass returns each hand's counterfactual value, where chance and card removal enter only
through each street's `fold` and `share` products (see core/public.py). Every (hand, public node)
pair is one infoset, so with the same updates the iterates equal the tree solver's.

Street t's lines are grouped into classes of the same betting shape (core/public.py,
`pt.classes[t]`). Arrays of class c on street t have shape (D_t, L_c, H, ...): deals, the
class's entering lines, hands. Player p's strategy at node k of class c's template is
sigma[t][c][k], shaped (D_t, L_c, H, actions). Reach and values are lists the same way:
reach[t][c][k] = (reach0, reach1) and values[t][c][k], where entry 0 is the class's root,
also for a zero-node class (all-in lines: no betting), whose template has no node 0.

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


# The passes' elementwise work, compiled so that it runs in parallel on big streets. Each
# kernel does the same arithmetic in the same order as the numpy expression it replaced,
# so results are bit-identical. Arrays are (D, L, H) per hand, as in the passes below.


@kernel
def _after_actions(r, sig, out):
    """out[a, d, l, h] = r[d, l, h] * sig[d, l, h, a]: the reach after each action."""
    D, L, H, A = sig.shape
    for m in prange(D * L):
        d, l = m // L, m % L
        for h in range(H):
            x = r[d, l, h]
            for a in range(A):
                out[a, d, l, h] = x * sig[d, l, h, a]


@kernel
def _scaled(coef, x, out):
    """out[d, l, h] = coef[l] * x[d, l, h]."""
    D, L, H = out.shape
    for m in prange(D * L):
        d, l = m // L, m % L
        c = coef[l]
        for h in range(H):
            out[d, l, h] = c * x[d, l, h]


@kernel
def _showdown(coef, share, fold, out):
    """out[d, l, h] = coef[l] * (2 share[d, l, h] - fold[d, l, h])."""
    D, L, H = out.shape
    for m in prange(D * L):
        d, l = m // L, m % L
        c = coef[l]
        for h in range(H):
            out[d, l, h] = c * (2.0 * share[d, l, h] - fold[d, l, h])


@kernel
def _copy(src, out):
    D, L, H = out.shape
    for m in prange(D * L):
        d, l = m // L, m % L
        for h in range(H):
            out[d, l, h] = src[d, l, h]


@kernel
def _combine(cv, out, best):
    """out[n] = cv[0, n] + cv[1, n] + ... in action order, or their max when `best`."""
    A, N = cv.shape
    for n in prange(N):
        v = cv[0, n]
        for a in range(1, A):
            x = cv[a, n]
            if not best:
                v = v + x
            elif x > v:  # on a tie keep the earlier value, as numpy's maximum does
                v = x
        out[n] = v


@kernel
def _segment_sum_kernel(values, parent, out):
    """out[parent[d], j] += values[d, j], adding child deals in order (as np.add.at does)."""
    D, M = values.shape
    for j in prange(M):
        for d in range(D):
            out[parent[d], j] += values[d, j]


@kernel
def _gather_rows(src, parent, out):
    """out[d, j] = src[parent[d], j]."""
    D, M = out.shape
    for d in prange(D):
        for j in range(M):
            out[d, j] = src[parent[d], j]


def _children_reach(r: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """r * sig[..., a] for every action a, as one (A, D, L, H) array."""
    out = np.empty((sig.shape[-1],) + r.shape)
    _after_actions(out.size, r, sig, out)
    return out


def _segment_sum(values: np.ndarray, parent: np.ndarray, n: int) -> np.ndarray:
    """Sum the rows of `values` (first axis = child deals) into their parent deals."""
    out = np.zeros((n,) + values.shape[1:])
    flat = np.ascontiguousarray(values).reshape(len(values), -1)
    _segment_sum_kernel(flat.size, flat, np.ascontiguousarray(parent), out.reshape(n, -1))
    return out


# Passes over a public tree ----------------------------------------------------------------
# hsig[t][c][k] is the per-hand strategy at node k of class c's template on street t,
# shaped (D, L_c, H, A). The class_* functions do one class; the others loop over classes.


def class_reach(tp, hsig_c, root):
    """Forward through one class's betting from its root reach (reach0, reach1): the reach at
    each decision node, and after each line that closes the street. A zero-node class passes
    its root through: nodes = [root] and one close."""
    if tp.num_nodes == 0:
        return [root], [root]
    nodes = [None] * tp.num_nodes
    nodes[0] = root
    closing = [None] * tp.num_closes
    for k in range(tp.num_nodes):
        r = nodes[k]
        q = tp.players[k]
        after = _children_reach(r[q], hsig_c[k])
        for a, child in enumerate(tp.children[k]):
            c = list(r)
            c[q] = after[a]
            if child[0] == "node":
                nodes[child[1]] = tuple(c)
            elif child[0] == "close":
                closing[child[1]] = tuple(c)
    return nodes, closing


def street_reach(classes_t, hsig_t, roots):
    """class_reach for every class of a street: (nodes[c], closing[c])."""
    nodes, closing = [], []
    for lc, h, root in zip(classes_t, hsig_t, roots):
        n, c = class_reach(lc.template, h, root)
        nodes.append(n)
        closing.append(c)
    return nodes, closing


def _identity(classes_t, classes_next) -> bool:
    """One class on both streets, the next street's lines in (line, close) order."""
    if len(classes_t) != 1 or len(classes_next) != 1:
        return False
    lc, J = classes_next[0], classes_t[0].template.num_closes
    return bool((lc.parent_class == 0).all() and np.array_equal(lc.parent_line * J + lc.parent_close,
                                                                 np.arange(lc.num_lines)))


def next_root(closing, classes_t, classes_next, parent: np.ndarray) -> list:
    """Reach at the next street's roots, per class: each deal takes its parent deal's reach
    after the closing line each of its lines continues (parent_class, parent_line,
    parent_close). closing[c][j] = (reach0, reach1) after close j of class c."""
    D, H = len(parent), closing[0][0][0].shape[-1]
    parent = np.ascontiguousarray(parent)
    if _identity(classes_t, classes_next):  # one class: lines = (previous lines x closes), as one gather
        out = []
        for p in (0, 1):
            src = np.stack([c[p] for c in closing[0]], axis=2)  # (parent deals, L, closes, H)
            rows = np.empty((D,) + src.shape[1:])
            _gather_rows(rows.size, src.reshape(len(src), -1), parent, rows.reshape(D, -1))
            out.append(rows.reshape(D, -1, H))
        return [tuple(out)]
    roots = []
    for lc in classes_next:
        root = []
        for p in (0, 1):
            Dp = closing[0][0][p].shape[0]
            src = np.empty((Dp, lc.num_lines, H))
            for pc in np.unique(lc.parent_class):
                m = np.flatnonzero(lc.parent_class == pc)
                stacked = np.stack([c[p] for c in closing[pc]], axis=2)  # (Dp, L_pc, closes, H)
                src[:, m] = stacked[:, lc.parent_line[m], lc.parent_close[m]]
            rows = np.empty((D, lc.num_lines, H))
            _gather_rows(rows.size, src.reshape(Dp, -1), parent, rows.reshape(D, -1))
            root.append(rows)
        roots.append(tuple(root))
    return roots


def stack_closing(closing_t) -> list:
    """closing[c][j] = (reach0, reach1) as [c] (reach0, reach1), each (D, L_c, closes_c, H)."""
    return [tuple(np.stack([cl[p] for cl in row], axis=2) for p in (0, 1)) for row in closing_t]


def forward(pt: PublicTree, hsig, closing: bool = False):
    """Both players' reach at every decision node: reach[t][c][k] = (reach0, reach1).

    With closing=True, returns (reach, last) where last[c] = (reach0, reach1) is the closing
    reach of the last street's class c, each shaped (D, L_c, closes_c, H): the reach after
    every line that ends the last street of pt without a fold (what a leaf value provider
    at the end of that street takes)."""
    D0, _, H = pt.shape(0, 0)
    if len(pt.classes[0]) != 1 or pt.classes[0][0].num_lines != 1:
        raise ValueError("the first street must have one line")
    roots = [(np.ones((D0, 1, H)), np.ones((D0, 1, H)))]
    out = []
    for t in range(pt.num_streets):
        nodes, closing_t = street_reach(pt.classes[t], hsig[t], roots)
        out.append(nodes)
        if t + 1 < pt.num_streets:
            roots = next_root(closing_t, pt.classes[t], pt.classes[t + 1], pt.streets[t + 1].parent)
    if closing:
        return out, stack_closing(closing_t)
    return out


def close_values(root_values, classes_t, classes_next, parent: np.ndarray, n_parents: int) -> list:
    """The next street's root values (root_values[c'] shaped (D, L_c', H)) summed into their
    parent deals and scattered back to the closing lines they continue: [c] (D_parent, L_c,
    closes_c, H) for each class c of this street."""
    if _identity(classes_t, classes_next):
        D, _, H = root_values[0].shape
        J = classes_t[0].template.num_closes
        return [_segment_sum(root_values[0].reshape(D, -1, J, H), parent, n_parents)]
    H = root_values[0].shape[-1]
    agg = [np.zeros((n_parents, lc.num_lines, lc.template.num_closes, H)) for lc in classes_t]
    for lc, v in zip(classes_next, root_values):
        summed = _segment_sum(v, parent, n_parents)  # (D_parent, L_c', H)
        for pc in np.unique(lc.parent_class):
            m = np.flatnonzero(lc.parent_class == pc)
            agg[pc][:, lc.parent_line[m], lc.parent_close[m]] = summed[:, m]
    return agg


def class_values(tp, st, contrib, hsig_c, reach_c, p: int, mode: str, last: bool, agg=None):
    """Player p's values at each node of one class (bottom-up), and instant regrets at p's
    nodes in mode "cfr". `contrib` (L_c,) is what each player has in at the street's start;
    `agg` (D, L_c, closes, H) holds the closing lines' values when the street isn't last.
    A zero-node class returns its root value as V[0]."""
    o = 1 - p
    sign = 1.0 if p == 0 else -1.0
    if tp.num_nodes == 0:
        ro = reach_c[0][o]
        v = np.empty(ro.shape)
        if last:
            chips = contrib + tp.closes[0][1]
            _showdown(v.size, sign * chips, st.share(ro, p), st.fold(ro, p), v)
        else:
            _copy(v.size, agg[:, :, 0, :], v)
        return [v], []
    V = [None] * tp.num_nodes
    R = [None] * tp.num_nodes
    for k in range(tp.num_nodes - 1, -1, -1):
        q = tp.players[k]
        rk = reach_c[k]
        A = len(tp.actions[k])
        cv = np.empty((A,) + rk[o].shape)
        ros = _children_reach(rk[o], hsig_c[k]) if q == o else None
        for a, child in enumerate(tp.children[k]):
            ro = ros[a] if q == o else rk[o]
            n = cv[a].size
            if child[0] == "node":
                _copy(n, V[child[1]], cv[a])
            elif child[0] == "fold":
                _, folder, extra = child
                lost = contrib + extra
                u0 = -lost if folder == 0 else lost
                _scaled(n, sign * u0, st.fold(ro, p), cv[a])
            elif last:  # showdown: player 0 wins pot x share - chips; player 1 the negative
                chips = contrib + tp.closes[child[1]][1]
                _showdown(n, sign * chips, st.share(ro, p), st.fold(ro, p), cv[a])
            else:
                _copy(n, agg[:, :, child[1], :], cv[a])
        V[k] = np.empty(cv.shape[1:])
        if q != p or mode == "br":
            _combine(cv.size, cv.reshape(A, -1), V[k].reshape(-1), q == p)
        else:
            R[k] = np.empty(cv.shape[1:] + (A,))
            _expectation(cv.size, np.ascontiguousarray(hsig_c[k]).reshape(-1, A), cv.reshape(A, -1),
                         V[k].reshape(-1), R[k].reshape(-1, A))
    return V, R


def street_values(classes_t, st, hsig_t, reach_t, p: int, mode: str, last: bool, agg=None):
    """class_values for every class of a street: (V[c], R[c]). agg[c] as in class_values."""
    V, R = [], []
    for c, lc in enumerate(classes_t):
        v, r = class_values(lc.template, st, lc.contrib, hsig_t[c], reach_t[c], p, mode, last,
                            None if agg is None else agg[c])
        V.append(v)
        R.append(r)
    return V, R


def backward(pt: PublicTree, hsig, reach, p: int, mode: str = "cfr"):
    """Player p's counterfactual value of each hand at every decision node, bottom-up:
    values[t][c][k] (entry 0 is the class root's).

    mode "cfr": p plays hsig; returns (values, regrets at p's nodes).
    mode "br":  p best-responds; returns (values, None for every node).
    """
    T = pt.num_streets
    all_values, all_regrets = [None] * T, [None] * T
    agg = None
    for t in range(T - 1, -1, -1):
        st = pt.streets[t]
        V, R = street_values(pt.classes[t], st, hsig[t], reach[t], p, mode, t == len(pt.classes) - 1, agg)
        all_values[t], all_regrets[t] = V, R
        if t > 0:
            agg = close_values([v[0] for v in V], pt.classes[t - 1], pt.classes[t], st.parent,
                               pt.streets[t - 1].num_deals)
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
        self.regret, self.cum = [], []  # [t][c][k] (D, L_c, ids, A)
        for t in range(pt.num_streets):
            self.regret.append([])
            for c, lc in enumerate(pt.classes[t]):
                D, L, H = pt.shape(t, c)
                tp = lc.template
                self.regret[t].append([np.zeros((D, L, self._ids(t, p), len(a))) for p, a in zip(tp.players, tp.actions)])
            self.cum.append([[np.zeros_like(r) for r in row] for row in self.regret[t]])
        self.sigma = [[[self._strategy(r) for r in rs] for rs in row] for row in self.regret]

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
        """The strategy per hand: sigma[t][c][k] shaped (D, L_c, H, actions)."""
        sigma = self.sigma if sigma is None else sigma
        return [[[self._to_hands(t, p, s) for p, s in zip(lc.template.players, sigma[t][c])]
                 for c, lc in enumerate(self.pt.classes[t])] for t in range(self.pt.num_streets)]

    def average_strategy(self) -> list:
        out = []
        for row in self.cum:
            out.append([])
            for cums in row:
                cls = []
                for c in cums:
                    total = c.sum(-1, keepdims=True)
                    cls.append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
                out[-1].append(cls)
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
            for t, c, k in self._nodes(p):
                reg = self.regret[t][c][k]
                A = reg.shape[-1]
                self.sigma[t][c][k] = np.ascontiguousarray(self.sigma[t][c][k])  # updated in place below
                own = np.ascontiguousarray(self.own_reach(t, c, k, reach)).reshape(-1)
                _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(regrets[t][c][k]).reshape(-1, A),
                        self.sigma[t][c][k].reshape(-1, A), self.cum[t][c][k].reshape(-1, A), own, weight, pos, neg,
                        VARIANTS[self.variant])
        self.solve_seconds += time.perf_counter() - start

    def _nodes(self, p: int):
        """(street, class, node) of every node where player p acts."""
        return [(t, c, k) for t in range(self.pt.num_streets) for c, lc in enumerate(self.pt.classes[t])
                for k, q in enumerate(lc.template.players) if q == p]

    def instant_regrets(self, p: int):
        """Both players' reach and player p's instant regret per infoset, under the current strategies."""
        hsig = self.hand_sigma()
        reach = self.reach(hsig)
        V, R = self.values(p, hsig, reach)
        self.value[p] = float(V[0][0][0].sum())
        regrets = [[[None] * lc.template.num_nodes for lc in self.pt.classes[t]] for t in range(self.pt.num_streets)]
        for t, c, k in self._nodes(p):
            regrets[t][c][k] = self._to_ids(t, p, R[t][c][k], self.regret[t][c][k].shape[2])
        return reach, regrets

    def own_reach(self, t: int, c: int, k: int, reach) -> np.ndarray:
        """The acting player's own reach of each infoset at node k of class c, shaped (D, L_c,
        ids, 1). Every hand in an infoset has the same own reach (perfect recall), so it's their
        mean."""
        p = self.pt.classes[t][c].template.players[k]
        r = reach[t][c][k][p]
        if self.buckets is None:
            return r[..., None]
        n_ids = self.regret[t][c][k].shape[2]
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
        br = [float(self.values(p, hsig, reach, "br")[0][0][0][0].sum()) for p in (0, 1)]
        value = float(self.values(0, hsig, reach)[0][0][0][0].sum())
        return {"exploitability": (br[0] + br[1]) / 2, "br_value_p0": br[0], "br_value_p1": br[1],
                "game_value": value}

    def log_row(self) -> dict:
        return {"iteration": self.t, "wall_time": self.solve_seconds, **self.exploitability()}


def infoset_count(pt: PublicTree) -> int:
    """Infosets of both players: (hand, public node) pairs where the hand can be held."""
    return int(sum(st.possible(p).sum() * lc.num_lines * sum(q == p for q in lc.template.players)
                   for t, st in enumerate(pt.streets) for lc in pt.classes[t] for p in (0, 1)))


def save_range_strategy(path, sigma, **info) -> None:
    """Save a range strategy (sigma[t][c][k] arrays) and `info` as a compressed .npz, keyed
    t<t>_c<c>_k<k>."""
    import json
    from pathlib import Path

    arrays = {f"t{t}_c{c}_k{k}": s for t, row in enumerate(sigma) for c, cls in enumerate(row)
              for k, s in enumerate(cls)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, info=np.array(json.dumps(info, default=str)), **arrays)


def load_range_strategy(path) -> tuple[list, dict]:
    """(sigma[t][c][k], info). Files from before line classes (keys t<t>_k<k>) load as class 0."""
    import json

    with np.load(path) as z:
        info = json.loads(str(z["info"]))
        names = [k for k in z.files if k != "info"]
        index = {}
        for key in names:
            parts = [int(x[1:]) for x in key.split("_")]
            index[key] = tuple(parts) if len(parts) == 3 else (parts[0], 0, parts[1])
        sigma: list = []
        for key in sorted(names, key=index.get):
            t, c, k = index[key]
            while len(sigma) <= t:
                sigma.append([])
            while len(sigma[t]) <= c:
                sigma[t].append([])
            sigma[t][c].append(z[key])
    return sigma, info
