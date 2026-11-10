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

from bombpot.core.public import PublicTree
from bombpot.core.solvers.range_cfr import RangeCFR, _segment_sum, backward, street_reach


class SubgameCFR(RangeCFR):
    """RangeCFR on one street entered with given ranges.

    street    a Street (CardStreet or DenseStreet) over the subgames' deals
    template  that street's betting template
    contrib   (L,) chips each player has in at the root, per line (one line per subgame)
    root      (reach0, reach1), each (G, L, H): the root ranges of each subgame
    group     (D,) the subgame root of each deal (default: all deals in group 0)
    """

    def __init__(self, street, template, contrib, root, group=None, variant: str = "cfr+", alpha: float = 1.5,
                 beta: float = 0.0, gamma: float = 2.0):
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
        pt = PublicTree("subgame", list(range(H)), [street], [template], [contrib])
        super().__init__(pt, variant, alpha, beta, gamma)
        self.root_ranges = (r0, r1)
        self.root = tuple(np.ascontiguousarray(r[self.group]) for r in (r0, r1))  # (D, L, H)

    def reach(self, hsig) -> list:
        nodes, _ = street_reach(self.pt.templates[0], hsig[0], self.root)
        return [nodes]

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
        values = [self.per_group(backward(self.pt, hsig, reach, p, "cfr")[0][0][0]) for p in (0, 1)]
        br = [self.per_group(backward(self.pt, hsig, reach, p, "br")[0][0][0]) for p in (0, 1)]
        r0, r1 = self.root_ranges  # counterfactual values leave out the player's own reach
        raw = ((br[0] * r0).sum(-1) + (br[1] * r1).sum(-1)) / 2
        mass = self.mass()
        expl = np.divide(raw, mass, out=np.zeros_like(raw), where=mass > 0)
        return {"values": values, "br_values": br, "exploitability": expl, "exploitability_raw": raw,
                "mass": mass}

    def solve(self, iters: int, tol: float | None = None, check_every: int = 10) -> dict:
        """Iterate until every subgame's exploitability (chips) is <= tol, checked every
        `check_every` iterations, or until `iters` iterations. Returns evaluate() of the final
        average strategy."""
        ev, at = None, -1
        while self.t < iters:
            self.iteration()
            if tol is not None and (self.t % check_every == 0 or self.t == iters):
                ev, at = self.evaluate(), self.t
                if ev["exploitability"].max() <= tol:
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


def solve_subgames(street, template, contrib, reach0, reach1, group=None, variant: str = "cfr+", iters: int = 1000,
                   tol: float | None = None, check_every: int = 10, stack=None, keep_solver: bool = False) -> Subgame:
    """Solve a batch of subgames sharing one template: reach0/reach1 (G, L, H), contrib (L,),
    group (D,) the subgame root of each deal. values (G, L, H); strategy[k] (D, L, H, A);
    exploitability (G, L). `stack` (chips behind) is accepted for the no-limit engine and not
    used yet: the template carries the betting."""
    import time

    start = time.perf_counter()
    s = SubgameCFR(street, template, contrib, (reach0, reach1), group, variant)
    ev = s.solve(iters, tol, check_every)
    strategy = s.hand_sigma(s.average_strategy())[0]
    return Subgame(ev["values"], ev["br_values"], strategy, s.t, ev["exploitability"], ev["exploitability_raw"],
                   ev["mass"], time.perf_counter() - start, s if keep_solver else None)


def solve_subgame(street, template, contrib, reach0, reach1, variant: str = "cfr+", iters: int = 1000,
                  tol: float | None = None, stack=None, check_every: int = 10, keep_solver: bool = False) -> Subgame:
    """One subgame: `street` holds its deals (all below one deal of the street before),
    `template` its betting, `contrib` the chips each player has in at its root, reach0 and
    reach1 the root ranges (H,), broadcast to every deal. Stops when its exploitability (chips
    per unit of root reach mass) is <= tol, checked every `check_every` iterations, or after
    `iters` iterations.

    Returns a Subgame with values[p] (H,), the root counterfactual values of the average
    strategy in solver units (street_values's root values summed over the deals), br_values[p]
    (H,), strategy[k] (D, H, A), and exploitability (float, chips) of the average strategy."""
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
    every (deal, line, close) at the leaf is solved from the closing ranges (solve_subgames,
    all at once) to `tol` chips, and its root values are returned (range_cfr.backward's
    `leaf` interface). For small decks: it holds every last-street deal.

    pt             the public tree of the streets above (its last street is where the leaf sits)
    last           a last-street provider: last.street(parents, children) and
                   last.all_children(parents), as for sampled_cfr
    template_last  the last street's betting template
    contrib_last   (L * closes,) chips each player has in at the last street's start, per line
                   (line l x closes + close c, as core/public.betting orders them)

    Nothing is reused across calls except the last street's deals (fixed) and the result of
    the previous call when its ranges are identical (the three passes of one evaluation).
    `stats` collects each solve's iterations and worst exploitability."""

    def __init__(self, pt, last, template_last, contrib_last, tol: float, variant: str = "cfr+",
                 iters: int = 10_000, check_every: int = 10):
        self.pt, self.template, self.contrib = pt, template_last, np.asarray(contrib_last, dtype=float)
        self.tol, self.variant, self.iters, self.check_every = tol, variant, iters, check_every
        D = pt.streets[-1].num_deals
        self.street = last.street(*last.all_children(np.arange(D)))
        self.group = np.asarray(self.street.parent)
        self.stats = []
        self._memo = None

    def solve(self, closing) -> Subgame:
        r0, r1 = closing
        D, L, C, H = r0.shape
        if self._memo is not None and all(np.array_equal(a, b) for a, b in zip(self._memo[0], closing)):
            return self._memo[1]
        sub = solve_subgames(self.street, self.template, self.contrib, r0.reshape(D, L * C, H),
                             r1.reshape(D, L * C, H), self.group, self.variant, self.iters, self.tol,
                             self.check_every)
        self.stats.append({"iterations": sub.iterations, "max_exploitability": float(sub.exploitability.max()),
                           "total_exploitability_raw": float(sub.exploitability_raw.sum()),
                           "seconds": sub.seconds})
        self._memo = (tuple(np.array(r) for r in closing), sub)
        return sub

    def values(self, p: int, t: int, closing) -> np.ndarray:
        sub = self.solve(closing)
        return sub.values[p].reshape(closing[p].shape)
