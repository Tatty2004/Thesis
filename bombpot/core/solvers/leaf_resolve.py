"""Safe re-solving of one street whose closing lines take their values from a leaf provider
(range_cfr.backward's `leaf`): a street above the last, e.g. the turn of a flop+turn+river
game whose river is valued by a leaf at the end of the turn.

The gadget is resolve.py's (Burch, Johanson and Bowling 2014; DeepStack's continual
re-solving), moved up one street. Player p re-solves; o = 1 - p. At each subgame root
(deal d, entering line l) o chooses per hand between F (enter: o's reach into the street is
its F probability) and T (take `cfv[d, l, h]`, o's counterfactual value there under the
blueprint, solver units: chance weight x p's reach x payoff, without o's reach). p's reach
is its blueprint reach. Folds pay as in range_cfr; a line that closes the street takes the
leaf's values for both players' closing reach there, exactly as range_cfr.backward feeds
the leaf (leaf.values(q, t, closing) with closing[c] = (reach0, reach1), each (D, L_c,
closes_c, H), returning (D, L_c, closes_c, H) per class). Without a leaf the street is the
last one and its closes are showdowns (this is then resolve._solve, which the tests use to
pin the gadget).

Everything is per (deal, line): many roots are solved at once, vectorized over the arrays
(D, L_c, H) of a street's line classes, and nothing couples them except what the leaf does
with its batch (a leaf solving exact subgames to a summed tolerance couples their accuracy).

Iterations are alternating (player 0 first), with range_cfr's update kernel and weights, so
CFR, CFR+ or DCFR as RangeCFR runs them; `simultaneous=True` updates both players from one
profile instead (one leaf pass per iteration, both players' values from it: half the network
calls with NetLeaf.values_both). Averaging weights: o's strategy by its reach within
the street times its current F probability (resolve._solve's own reach), p's by its reach
within the street (the same average as weighting by p's root reach too wherever that is
positive, and defined for hands p doesn't hold). The leaf's values for each player are kept
averaged over the iterations with the iteration weights (DeepStack's opponent values for the
next street's re-solve; gate_f.LeafCFR keeps the trunk's the same way).

`evaluate` gives the gadget game's NashConv for the average profile per root (`gap`, solver
units), with the leaf valued at the average profile's closing reach:
    gap = sum_h max(br_o(h), cfv(h)) + sum_h reach_p(h) br_p(h) - sum_h T_avg(h) cfv(h)
and `mass` = sum over o's hands of chance x p's reach, so gap / mass is in chips per unit of
reach (resolve's normalization). With cfv = o's best-response value against the blueprint
(and a leaf that is a fixed value function), sum_h max(0, br_o(h) - cfv(h)) <= gap per root:
replacing the blueprint below the roots by the re-solve raises o's best response by at most
the summed gaps (resolve.py's module doc).

Unsafe re-solving (`cfv=None`, `reach_o` given): the street alone from both blueprint reaches.
"""
from __future__ import annotations

import time

import numpy as np

from bombpot.core.solvers.range_cfr import VARIANTS, _update, class_reach, class_values, stack_closing

F, T = 0, 1


def _matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    total = pos.sum(-1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / regret.shape[-1])


class LeafGadget:
    """The re-solving gadget over the roots of one street.

    street    the street's Street (fold sums with its chance weights; share sums too when
              leaf is None), D deals
    classes   [c] objects with .template (a Template with betting) and .contrib (L_c,) chips
              each player has in at the root (LineClass, or one restricted to some lines)
    reach_p   [c] (D, L_c, H) p's root reach
    cfv       [c] (D, L_c, H) o's T values (solver units), or None for an unsafe re-solve
    reach_o   [c] (D, L_c, H) o's root reach, for an unsafe re-solve only
    leaf      a provider with values(q, t, closing) (above), called with t = `leaf_street`;
              None: the street is the last and closes are showdowns
    """

    def __init__(self, street, classes, reach_p, cfv, player: int, leaf=None, reach_o=None, variant: str = "dcfr",
                 alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0, leaf_street: int = 1,
                 simultaneous: bool = False):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if any(lc.template.num_nodes == 0 for lc in classes):
            raise ValueError("a class without betting has nothing to re-solve")
        self.safe = cfv is not None
        if not self.safe and reach_o is None:
            raise ValueError("an unsafe re-solve needs the opponent's root reach")
        self.street, self.classes, self.leaf, self.leaf_street = street, list(classes), leaf, leaf_street
        self.p, self.o = player, 1 - player
        self.simultaneous = simultaneous
        self.variant, self.alpha, self.beta, self.gamma = variant, alpha, beta, gamma
        f = lambda xs: [np.ascontiguousarray(x, dtype=np.float64) for x in xs]
        self.rp = f(reach_p)
        self.cfv = f(cfv) if self.safe else None
        self.ro = None if self.safe else f(reach_o)
        self.shape = [x.shape for x in self.rp]
        self.regret = [[np.zeros(s + (len(a),)) for a in lc.template.actions] for s, lc in zip(self.shape, self.classes)]
        self.cum = [[np.zeros_like(r) for r in row] for row in self.regret]
        self.sigma = [[_matching(r) for r in row] for row in self.regret]
        if self.safe:
            self.g_regret = [np.zeros(s + (2,)) for s in self.shape]
            self.g_cum = [np.zeros(s + (2,)) for s in self.shape]
            self.g_sig = [np.full(s + (2,), 0.5) for s in self.shape]
        self.leaf_cum = [None, None]  # [q][c] (D, L_c, closes_c, H): q's leaf values x iteration weight
        self.weight_sum = [0.0, 0.0]
        self.t = 0
        self.seconds = 0.0
        self.leaf_seconds = 0.0
        self.leaf_calls = 0

    # Pieces ---------------------------------------------------------------------------------

    def _roots(self, follow) -> list:
        """[c] (reach0, reach1) at the roots: p's blueprint reach and o's F probability (or
        o's blueprint reach when unsafe)."""
        out = []
        for c in range(len(self.classes)):
            ro = follow[c] if self.safe else self.ro[c]
            out.append((self.rp[c], ro) if self.p == 0 else (ro, self.rp[c]))
        return out

    def _follow(self, avg: bool = False) -> list | None:
        if not self.safe:
            return None
        g = self.gadget_average() if avg else self.g_sig
        return [np.ascontiguousarray(x[..., F]) for x in g]

    def _leaf_values(self, q: int, closing) -> list | None:
        """q's values at every close of every class (the leaf's), or None without a leaf."""
        if self.leaf is None:
            return None
        start = time.perf_counter()
        agg = self.leaf.values(q, self.leaf_street, stack_closing(closing))
        self.leaf_seconds += time.perf_counter() - start
        self.leaf_calls += 1
        return [np.asarray(a, dtype=np.float64) for a in agg]

    def _leaf_both(self, closing) -> tuple:
        """Both players' leaf values for one closing reach: one call when the leaf has
        values_both (NetLeaf: one network pass), else two."""
        if self.leaf is None:
            return None, None
        if not hasattr(self.leaf, "values_both"):
            return self._leaf_values(0, closing), self._leaf_values(1, closing)
        start = time.perf_counter()
        both = self.leaf.values_both(self.leaf_street, stack_closing(closing))
        self.leaf_seconds += time.perf_counter() - start
        self.leaf_calls += 1
        return tuple([np.asarray(a, dtype=np.float64) for a in x] for x in both)

    def _reach(self, sigma, roots):
        nodes, closing = [], []
        for lc, sg, r in zip(self.classes, sigma, roots):
            n, cl = class_reach(lc.template, sg, r)
            nodes.append(n)
            closing.append(cl)
        return nodes, closing

    def _pass(self, sigma, roots, q: int, mode: str, agg="call", reach=None):
        """Player q's values (V[c][k], R[c][k]) at every node, the nodes' reach, and q's leaf
        values (or None). agg: q's leaf values when already known; reach: (nodes, closing)."""
        nodes, closing = self._reach(sigma, roots) if reach is None else reach
        if isinstance(agg, str):
            agg = self._leaf_values(q, closing)
        V, R = [], []
        for c, lc in enumerate(self.classes):
            v, r = class_values(lc.template, self.street, lc.contrib, sigma[c], nodes[c], q, mode, agg is None,
                                None if agg is None else agg[c])
            V.append(v)
            R.append(r)
        return V, R, nodes, agg

    def _weights(self):
        t = self.t
        weight = {"cfr": 1.0, "cfr+": float(t), "dcfr": float(t) ** self.gamma}[self.variant]
        pos = t ** self.alpha / (t ** self.alpha + 1.0)
        neg = t ** self.beta / (t ** self.beta + 1.0)
        return weight, pos, neg

    # Iterations -----------------------------------------------------------------------------

    def iterate(self, n: int = 1) -> None:
        start = time.perf_counter()
        code = VARIANTS[self.variant]
        ones = [(np.ones(s), np.ones(s)) for s in self.shape]
        for _ in range(n):
            self.t += 1
            weight, pos, neg = self._weights()
            if self.simultaneous:  # both players' regrets from one profile (one leaf pass for both)
                follow = self._follow()
                roots = self._roots(follow)
                reach = self._reach(self.sigma, roots)
                aggs = self._leaf_both(reach[1])
                res = [self._pass(self.sigma, roots, q, "cfr", aggs[q], reach) for q in (0, 1)]
                for q in (0, 1):
                    self._accumulate(q, res[q][3], weight)
                for c, lc in enumerate(self.classes):
                    within, _ = class_reach(lc.template, self.sigma[c], ones[c])
                    for k, owner in enumerate(lc.template.players):
                        own = within[k][owner] * follow[c] if (self.safe and owner == self.o) else within[k][owner]
                        self._node_update(c, k, res[owner][1][c][k], own, weight, pos, neg, code)
                if self.safe:
                    self._gadget_update(res[self.o][0], weight, pos, neg, code)
                continue
            for q in (0, 1):
                follow = self._follow()
                V, R, _, agg = self._pass(self.sigma, self._roots(follow), q, "cfr")
                self._accumulate(q, agg, weight)
                for c, lc in enumerate(self.classes):
                    within, _ = class_reach(lc.template, self.sigma[c], ones[c])  # each player's own reach in the street
                    for k, owner in enumerate(lc.template.players):
                        if owner == q:
                            own = within[k][q] * follow[c] if (self.safe and q == self.o) else within[k][q]
                            self._node_update(c, k, R[c][k], own, weight, pos, neg, code)
                if self.safe and q == self.o:
                    self._gadget_update(V, weight, pos, neg, code)
        self.seconds += time.perf_counter() - start

    def _accumulate(self, q: int, agg, weight: float) -> None:
        if agg is not None:
            self.leaf_cum[q] = [weight * a for a in agg] if self.leaf_cum[q] is None else \
                [x + weight * a for x, a in zip(self.leaf_cum[q], agg)]
            self.weight_sum[q] += weight

    def _node_update(self, c: int, k: int, instant, own, weight, pos, neg, code) -> None:
        reg = self.regret[c][k]
        A = reg.shape[-1]
        self.sigma[c][k] = np.ascontiguousarray(self.sigma[c][k])
        _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(instant).reshape(-1, A),
                self.sigma[c][k].reshape(-1, A), self.cum[c][k].reshape(-1, A), np.ascontiguousarray(own).reshape(-1),
                weight, pos, neg, code)

    def _gadget_update(self, V, weight, pos, neg, code) -> None:
        """The gadget node: F pays o its root value V[c][0], T pays cfv."""
        for c in range(len(self.classes)):
            g, w, x = self.g_sig[c], V[c][0], self.cfv[c]
            value = g[..., F] * w + g[..., T] * x
            inst = np.stack([w - value, x - value], axis=-1)
            _update(g.size, self.g_regret[c].reshape(-1, 2), np.ascontiguousarray(inst).reshape(-1, 2),
                    g.reshape(-1, 2), self.g_cum[c].reshape(-1, 2), np.ones(g.size // 2), weight, pos, neg, code)

    # Results --------------------------------------------------------------------------------

    def average_strategy(self) -> list:
        """[c][k] (D, L_c, H, A): the average strategy at every node (p's nodes are the
        re-solve; o's are o's play in the gadget game)."""
        return [[_matching_avg(x) for x in row] for row in self.cum]

    def gadget_average(self) -> list:
        """[c] (D, L_c, H, 2): o's average (F, T) probabilities."""
        return [_matching_avg(x, 0.5) for x in self.g_cum]

    def leaf_average(self, q: int) -> list | None:
        """[c] (D, L_c, closes_c, H): q's leaf values averaged over the iterations (solver
        units): the opponent's T values for re-solving the next street."""
        if self.leaf_cum[q] is None:
            return None
        return [x / self.weight_sum[q] for x in self.leaf_cum[q]]

    def mass(self) -> list:
        """[c] (D, L_c): the chance x p-reach weight of each root over o's hands (x o's reach
        when unsafe), which normalizes the gap."""
        out = []
        for c in range(len(self.classes)):
            m = self.street.fold(self.rp[c], self.o)
            out.append((m if self.safe else m * self.ro[c]).sum(-1))
        return out

    def evaluate(self) -> dict:
        """The average profile's values and the gadget NashConv per root (module doc):
        br[q], values[q]  [c] (D, L_c, H) best-response and on-policy root values
        gap, mass          [c] (D, L_c); exploitability = max gap / mass (chips)
        leaf               [q][c] the leaf's values at the average's closing reach
        """
        start = time.perf_counter()
        avg = self.average_strategy()
        roots = self._roots(self._follow(avg=True))
        br, on, leaf = [None, None], [None, None], [None, None]
        reach = self._reach(avg, roots)
        aggs = self._leaf_both(reach[1])  # the closing reach is the same for both players
        for q in (0, 1):
            Vb, _, _, agg = self._pass(avg, roots, q, "br", aggs[q], reach)
            br[q] = [v[0] for v in Vb]
            leaf[q] = agg
            Vc, _, _, _ = self._pass(avg, roots, q, "cfr", agg, reach)
            on[q] = [v[0] for v in Vc]
        p, o = self.p, self.o
        gap, ga = [], self.gadget_average() if self.safe else None
        for c in range(len(self.classes)):
            best_p = (self.rp[c] * br[p][c]).sum(-1)
            if self.safe:
                best_o = np.maximum(br[o][c], self.cfv[c]).sum(-1)
                best_p = best_p - (ga[c][..., T] * self.cfv[c]).sum(-1)
            else:
                best_o = (self.ro[c] * br[o][c]).sum(-1)
            gap.append(best_o + best_p)
        mass = self.mass()
        norm = max(float(np.divide(g, m, out=np.zeros_like(g), where=m > 0).max()) for g, m in zip(gap, mass))
        self.seconds += time.perf_counter() - start
        return {"br": br, "values": on, "gap": gap, "mass": mass, "exploitability": norm, "leaf": leaf,
                "gadget": ga, "iterations": self.t}

    def solve(self, iters: int, tol: float | None = None, check_every: int = 10) -> dict:
        """Iterate to `iters` iterations, or until the largest gap / mass is <= tol (checked
        every `check_every` iterations). Returns evaluate()'s dict with `history`
        [(iteration, exploitability)] of the checks."""
        ev, history = None, []
        while self.t < iters:
            n = iters - self.t if tol is None else min(check_every - self.t % check_every, iters - self.t)
            self.iterate(n)
            if tol is not None or self.t >= iters:
                ev = self.evaluate()
                history.append((self.t, ev["exploitability"]))
                if tol is not None and ev["exploitability"] <= tol:
                    break
        if ev is None:
            ev = self.evaluate()
            history.append((self.t, ev["exploitability"]))
        ev["history"] = history
        return ev


def _matching_avg(c: np.ndarray, empty: float | None = None) -> np.ndarray:
    total = c.sum(-1, keepdims=True)
    fill = 1.0 / c.shape[-1] if empty is None else empty
    return np.where(total > 0, c / np.where(total > 0, total, 1.0), fill)
