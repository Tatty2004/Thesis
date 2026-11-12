"""Last-street subgames solved from given root ranges, and a leaf provider built on them.

A subgame is one last street entered with fixed ranges: the deals of that street below one
deal of the street before (e.g. every river below one turn), one betting template, the chips
each player already has in, and both players' reach at the root. Its counterfactual values
are in the range solver's units (chance weight x opponent reach x payoff), so the root
values summed over the subgame's deals are exactly what range_cfr.close_values puts into the
street above.

Many subgames that share a template are solved at once: the street holds the deals of all of
them, `group[d]` says which subgame-root deal deal d belongs to, and the template's line axis
holds independent subgames with their own chips (`contrib[l]`). Nothing couples them (regrets
live per deal, line and hand), so this is the same as solving each alone, vectorized as the
range solver vectorizes a street.

The solver is RangeCFR on a one-street public tree whose root reach is the given ranges
instead of ones: CFR+ by default (regret floor, alternating updates, linear averaging), or
DCFR or vanilla CFR.

Exploitability of a subgame is (best response of player 0 + best response of player 1) / 2
at the root, each player's root values weighted by its own root range (counterfactual values
leave out the player's own reach), in chips per unit of root reach mass: divided by the sum
over deals and disjoint hand pairs of kappa x reach0 x reach1. With root reach 1 and chance
weights that sum to 1 the mass is 1, as in a whole game.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bombpot.core.public import LineClass, PublicTree, Template
from bombpot.core.solvers.range_cfr import RangeCFR, _segment_sum, backward, class_reach


def line_template(tp: Template, lines) -> Template:
    """A class template restricted to some of its lines: the fold and close chips ((L_c,)
    arrays, or scalars) indexed by `lines` (an int or an index array)."""
    idx = np.atleast_1d(np.asarray(lines))

    def pick(x):
        x = np.asarray(x, dtype=float)
        return x[idx] if x.ndim else np.full(len(idx), float(x))

    children = [[("fold", ch[1], pick(ch[2])) if ch[0] == "fold" else ch for ch in row] for row in tp.children]
    closes = [(h, pick(x)) for h, x in tp.closes]
    return Template(tp.players, tp.actions, tp.hist, children, closes)


class SubgameCFR(RangeCFR):
    """RangeCFR on one street entered with given ranges.

    street    a Street (CardStreet or DenseStreet) over the subgames' deals
    template  that street's betting template
    contrib   (L,) chips each player has in at the root, per line (one line per subgame); the
              template's chips are (L,) arrays (or scalars) for the same lines
    root      (reach0, reach1), each (G, L, H): the root ranges of each subgame
    group     (D,) the subgame root of each deal (default: all deals in group 0)
    stack     (L,) chips behind (bookkeeping only: the template carries the betting)
    """

    def __init__(self, street, template, contrib, root, group=None, variant: str = "cfr+", alpha: float = 1.5,
                 beta: float = 0.0, gamma: float = 2.0, stack=None):
        contrib = np.atleast_1d(np.asarray(contrib, dtype=float))
        r0, r1 = (np.asarray(r, dtype=float) for r in root)
        if r0.ndim == 1:
            r0, r1 = r0[None, None], r1[None, None]
        G, L, H = r0.shape
        if r1.shape != r0.shape or len(contrib) != L:
            raise ValueError(f"root ranges {r0.shape}, {r1.shape} and contrib {contrib.shape} don't agree")
        D = street.num_deals
        self.group = np.zeros(D, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)
        self.G = G
        stack = np.full(L, np.inf) if stack is None else np.broadcast_to(np.asarray(stack, dtype=float), (L,))
        none = np.full(L, -1, dtype=np.int64)
        lc = LineClass(template, np.arange(L), contrib, np.array(stack), none, none, none)
        pt = PublicTree("subgame", list(range(H)), [street], [[lc]])
        super().__init__(pt, variant, alpha, beta, gamma)
        self.template = template
        self.root_ranges = (r0, r1)
        self.root = tuple(np.ascontiguousarray(r[self.group]) for r in (r0, r1))  # (D, L, H)
        self._ones = (np.ones((D, L, H)), np.ones((D, L, H)))
        self._within = None

    def reach(self, hsig) -> list:
        nodes, _ = class_reach(self.template, hsig[0][0], self.root)
        self._within, _ = class_reach(self.template, hsig[0][0], self._ones)
        return [[nodes]]

    def own_reach(self, t: int, c: int, k: int, reach) -> np.ndarray:
        """The acting player's reach of node k *within the subgame* (from a root reach of 1),
        the weight of its average strategy. A hand's root reach is a constant factor of its
        own reach at every node, so for a hand with positive root reach the average is the same
        as with the full reach. A hand with root reach 0 would otherwise have no average at
        all (uniform play), and its root value would be the value of uniform play; this way its
        average is its regret-minimizing play against the opponent, and its root value
        converges to its best-response value like every other hand's. Whoever takes root
        values for all hands (a leaf provider, a value network's targets) needs that: the
        street above decides from them whether such a hand should come here."""
        return self._within[k][self.template.players[k]][..., None]

    def per_group(self, x: np.ndarray) -> np.ndarray:
        """(D, L, H) per deal -> (G, L, H) summed over each subgame's deals."""
        return _segment_sum(x, self.group, self.G)

    def mass(self) -> np.ndarray:
        """(G, L): sum of kappa x reach0[i] x reach1[j] over each subgame's deals and disjoint pairs."""
        st = self.pt.streets[0]
        return (self.per_group(st.fold(self.root[1], 0)) * self.root_ranges[0]).sum(-1)

    def evaluate(self, sigma=None) -> dict:
        """Values of a profile (the average strategy by default), per subgame:
        values[p], br_values[p] (G, L, H); exploitability (G, L) in chips (see the module
        docstring) and exploitability_raw (G, L) in solver units; mass (G, L)."""
        hsig = self.hand_sigma(self.average_strategy() if sigma is None else sigma)
        reach = self.reach(hsig)
        values = [self.per_group(backward(self.pt, hsig, reach, p, "cfr")[0][0][0][0]) for p in (0, 1)]
        br = [self.per_group(backward(self.pt, hsig, reach, p, "br")[0][0][0][0]) for p in (0, 1)]
        r0, r1 = self.root_ranges  # counterfactual values leave out the player's own reach
        raw = ((br[0] * r0).sum(-1) + (br[1] * r1).sum(-1)) / 2
        mass = self.mass()
        expl = np.divide(raw, mass, out=np.zeros_like(raw), where=mass > 0)
        cfv_gap = ((br[0] - values[0]).sum(-1) + (br[1] - values[1]).sum(-1)) / 2
        return {"values": values, "br_values": br, "exploitability": expl, "exploitability_raw": raw,
                "mass": mass, "cfv_gap": cfv_gap}

    def solve(self, iters: int, tol: float | None = None, check_every: int = 10, criterion: str = "max") -> dict:
        """Iterate until the error is <= tol, checked every `check_every` iterations, or until
        `iters` iterations. criterion "max": every subgame's exploitability (chips per unit of
        its reach mass); "total": the sum of the subgames' exploitability_raw (solver units:
        with the game's own chance weights, the batch's share of the whole game's error in
        chips); "cfv": the sum of the subgames' cfv_gap, i.e. over both players and *every* hand,
        whatever its own root reach, of best-response root value - root value, halved (solver
        units). It bounds "total" when root reaches are at most 1, and it is the error of the
        root values as a leaf provider hands them out: a hand's root value doesn't contain its
        own reach, so a hand that is rare here weighs as much in the street above's regrets as
        any other. Returns evaluate() of the final average strategy."""
        if criterion not in ("max", "total", "cfv"):
            raise ValueError(f"unknown criterion {criterion!r}")
        if self.template.num_nodes == 0:  # no betting (all-in): a showdown, nothing to solve
            return self.evaluate()
        ev, at = None, -1
        while self.t < iters:
            self.iteration()
            if tol is not None and (self.t % check_every == 0 or self.t == iters):
                ev, at = self.evaluate(), self.t
                err = (ev["exploitability"].max() if criterion == "max" else
                       ev["exploitability_raw"].sum() if criterion == "total" else ev["cfv_gap"].sum())
                if err <= tol:
                    break
        return ev if at == self.t else self.evaluate()


@dataclass
class Subgame:
    """A solved subgame (or batch of them; see solve_subgames for the shapes)."""
    values: list            # [p] root counterfactual values, solver units
    br_values: list         # [p] root best-response values, solver units
    strategy: list          # [k] average strategy at template node k
    iterations: int
    exploitability: object  # chips per unit of root reach mass
    exploitability_raw: object  # (br0 + br1) / 2, solver units
    mass: object            # root reach mass
    seconds: float = 0.0
    solver: object = field(default=None, repr=False)
    cfv_gap: object = None      # sum over both players and every hand of br value - value, / 2


def solve_subgames(street, template, contrib, reach0, reach1, group=None, variant: str = "cfr+", iters: int = 1000,
                   tol: float | None = None, check_every: int = 10, stack=None, keep_solver: bool = False,
                   criterion: str = "max") -> Subgame:
    """Solve a batch of subgames sharing one template: reach0/reach1 (G, L, H), contrib (L,),
    group (D,) the subgame root of each deal. values (G, L, H); strategy[k] (D, L, H, A);
    exploitability (G, L). `criterion` as in SubgameCFR.solve. `stack` (chips behind) is
    accepted for the no-limit engine and not used yet: the template carries the betting."""
    import time

    start = time.perf_counter()
    s = SubgameCFR(street, template, contrib, (reach0, reach1), group, variant, stack=stack)
    ev = s.solve(iters, tol, check_every, criterion)
    strategy = s.hand_sigma(s.average_strategy())[0][0]
    return Subgame(ev["values"], ev["br_values"], strategy, s.t, ev["exploitability"], ev["exploitability_raw"],
                   ev["mass"], time.perf_counter() - start, s if keep_solver else None, ev["cfv_gap"])


def solve_subgame(street, template, contrib, reach0, reach1, variant: str = "cfr+", iters: int = 1000,
                  tol: float | None = None, stack=None, check_every: int = 10, keep_solver: bool = False) -> Subgame:
    """One subgame: `street` holds its deals (all below one deal of the street before),
    `template` its betting (a class template with one line, or line_template(class template,
    l)), `contrib` the chips each player has in at its root, reach0 and
    reach1 the root ranges (H,), broadcast to every deal. Stops when its exploitability (chips
    per unit of root reach mass) is <= tol, checked every `check_every` iterations, or after
    `iters` iterations.

    Returns a Subgame with values[p] (H,), the root counterfactual values of the average
    strategy in solver units (street_values's root values summed over the deals), br_values[p]
    (H,), strategy[k] (D, H, A), and exploitability (float, chips) of the average strategy."""
    stack = None if stack is None else [float(np.asarray(stack).reshape(-1)[0])]
    out = solve_subgames(street, template, [float(np.asarray(contrib).reshape(-1)[0])],
                         np.asarray(reach0, dtype=float)[None, None], np.asarray(reach1, dtype=float)[None, None],
                         None, variant, iters, tol, check_every, stack, keep_solver)
    out.values = [v[0, 0] for v in out.values]
    out.br_values = [v[0, 0] for v in out.br_values]
    out.strategy = [x[:, 0] for x in out.strategy]
    out.exploitability = float(out.exploitability[0, 0])
    out.exploitability_raw = float(out.exploitability_raw[0, 0])
    out.mass = float(out.mass[0, 0])
    return out


class ExactRiverLeaf:
    """Leaf values at the end of the second-to-last street, from exact last-street subgames:
    every (deal, line, close) at the leaf is solved from the closing ranges and its root values
    are returned (range_cfr.backward's `leaf` interface). Each last-street line class is one
    solve_subgames batch over every (deal, line); its roots are gathered from the closing reach
    of the parent classes (parent_class, parent_line, parent_close) and its values scattered
    back the same way. For small decks: it holds every last-street deal.

    pt             the public tree of the streets above; pt.classes[pt.num_streets] (or
                   `classes_last`) are the last street's line classes
    last           a last-street provider: last.street(parents, children) and
                   last.all_children(parents), as for sampled_cfr

    Tolerance: criterion "total" (default) stops each class's batch when its summed
    exploitability_raw is <= tol / (classes with betting), so the leaf's error in chips of the
    whole game (with the game's chance weights) is at most tol; "max" when every subgame is
    within tol chips per unit of its own reach mass. Nothing is reused across calls except the
    last street's deals (fixed) and the previous call's result when its ranges are identical
    (the three passes of one evaluation). `stats` collects each call's iterations and errors."""

    def __init__(self, pt, last, tol, classes_last=None, variant: str = "cfr+", iters: int = 10_000,
                 check_every: int = 10, criterion: str = "total"):
        self.pt = pt
        self.classes = pt.classes[pt.num_streets] if classes_last is None else classes_last
        self.tol, self.variant, self.iters, self.check_every = tol, variant, iters, check_every
        self.criterion = criterion
        D = pt.streets[-1].num_deals
        self.street = last.street(*last.all_children(np.arange(D)))
        self.group = np.asarray(self.street.parent)
        self.stats = []
        self._memo = None

    def solve(self, closing) -> list:
        """The batches' Subgames, one per last-street class, from closing[c] = (reach0, reach1)."""
        if self._memo is not None and all(np.array_equal(a, b) for x, y in zip(self._memo[0], closing)
                                          for a, b in zip(x, y)):
            return self._memo[1]
        betting = sum(lc.template.num_nodes > 0 for lc in self.classes)
        tol = None if self.tol is None else (self.tol if self.criterion == "max" else self.tol / max(betting, 1))
        subs = []
        for lc, roots in zip(self.classes, self.roots(closing)):
            subs.append(solve_subgames(self.street, lc.template, lc.contrib, roots[0], roots[1], self.group,
                                       self.variant, self.iters, tol, self.check_every, lc.stack,
                                       criterion=self.criterion))
        self.stats.append({"iterations": max(s.iterations for s in subs),
                           "max_exploitability": max(float(s.exploitability.max()) for s in subs),
                           "total_exploitability_raw": sum(float(s.exploitability_raw.sum()) for s in subs),
                           "total_cfv_gap": sum(float(s.cfv_gap.sum()) for s in subs),
                           "seconds": sum(s.seconds for s in subs)})
        self._memo = (tuple(tuple(np.array(r) for r in c) for c in closing), subs)
        return subs

    def roots(self, closing) -> list:
        """The last street's root reach per class, [c] (reach0, reach1) each (D, L_c, H): each
        line takes the closing reach of the line and close it continues."""
        out = []
        for lc in self.classes:
            roots = [np.empty((len(closing[0][q]), lc.num_lines, closing[0][q].shape[-1])) for q in (0, 1)]
            for q in (0, 1):
                for pc in np.unique(lc.parent_class):
                    m = np.flatnonzero(lc.parent_class == pc)
                    roots[q][:, m] = closing[pc][q][:, lc.parent_line[m], lc.parent_close[m]]
            out.append(tuple(roots))
        return out

    def strategy(self, closing) -> list:
        """The last street's average strategy per class, [c][k] (D, L_c, H, A): every subgame's,
        in the full tree's layout (the deals in last.all_children order)."""
        return [s.strategy for s in self.solve(closing)]

    def values(self, p: int, t: int, closing) -> list:
        subs = self.solve(closing)
        agg = [np.zeros(c[p].shape) for c in closing]
        for lc, sub in zip(self.classes, subs):
            for pc in np.unique(lc.parent_class):
                m = np.flatnonzero(lc.parent_class == pc)
                agg[pc][:, lc.parent_line[m], lc.parent_close[m]] = sub.values[p][:, m]
        return agg
