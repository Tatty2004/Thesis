"""Re-solving one last-street subgame: safe (with the re-solving gadget) and unsafe.

A subgame here is everything below one deal of the street before the last (one turn deal):
the last street's chance (every river deal of that turn deal) and its betting, for every
entering line at once. Lines are independent subgames that share a betting template, so
they are solved together, vectorized like the range solver: arrays are (n, L, H, ...) over
the n last-street deals of the subgame, its L entering lines and the H hands.

Units (the solver's, as everywhere in range_cfr): `street_values` gives player q's
counterfactual value of each hand at each last-street deal as
    chance weight x the other player's reach x payoff,
without q's own reach. Summed over the subgame's deals (close_values does exactly that) it
is q's counterfactual value at the subgame root, the quantity the street above sees. Every
root value in this module, `opponent_cfv` included, is in these units, per (line, hand),
shaped (L, H). `blueprint_cfv` computes it from a stored river strategy with the same
passes, so the gadget compares like with like.

Safe re-solving (Burch, Johanson and Bowling 2014, the re-solving gadget; Brown and Sandholm
2017; DeepStack's continual re-solving). Player p re-solves; o = 1 - p. At the root, for each
line and hand, o chooses between F ("follow": enter the subgame) and T ("terminate": take
`opponent_cfv[l, h]`). In the gadget o's reach into the subgame is just its probability of
F (its blueprint reach plays no part), and p's reach is its blueprint reach. So p minimizes
    G = sum_h max(w_o(h), opponent_cfv(h)),
where w_o is o's best-response root value against p's subgame strategy. With
opponent_cfv = o's best-response value against the blueprint river (`mode="br"`, the
default of `blueprint_cfv`), the blueprint itself has G = sum_h opponent_cfv, so for any
re-solved strategy whose gadget NashConv (`gap`, below) is g,
    sum_h max(0, w_o(h) - opponent_cfv(h)) <= g    per line,
and replacing the blueprint river by re-solved ones raises o's best response in the full
game by at most the sum of the gaps over every re-solved root (the leaf values only enter
the street above through sums and maxima). The exploitability (NashConv / 2) of a profile
whose river is re-solved for both players rises by at most (sum of gaps for p = 0 +
sum of gaps for p = 1) / 2.

Unsafe re-solving: the subgame alone, from both blueprint reaches, with no gadget.

`tol` is a normalized gap in chips: gap(l) / mass(l), with mass(l) the total chance x p-reach
weight of the line (x o's reach when unsafe). Since the masses of all roots add up to at
most 1, a tolerance tol per root keeps the sum of gaps, and so the rise in exploitability,
at most tol. The exact sums are returned so callers can use the tighter bound.

`solve_subgame` has the subgame stream's signature (core/solvers/subgame.py), so the two
can be swapped; `resolve` adds the gadget around the same loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from bombpot.core.solvers.range_cfr import VARIANTS, _segment_sum, _update, street_reach, street_values

F, T = 0, 1  # the gadget's actions


@dataclass
class Subgame:
    """A solved subgame (or gadget game). Root values are (L, H) in solver units; a batch
    (`resolve_batch`) has a leading axis of G subgames on every per-subgame field, and
    strategy arrays keep the batch's deal axis."""
    strategy: list          # [k] average strategy at template node k, (n, L, H, A), both players
    root_cfv: tuple         # (player 0, player 1) on-policy root values of the average profile
    br_cfv: tuple           # (player 0, player 1) best-response root values against the average
    gap: np.ndarray         # (L,) NashConv of the average profile (gadget game when safe), solver units
    mass: np.ndarray        # (L,) the weight that normalizes gap into chips
    exploitability: float   # max over lines of gap / mass: chips
    iterations: int
    seconds: float
    gadget: np.ndarray | None = None        # (L, H, 2): o's average (F, T) probabilities
    opponent_cfv: np.ndarray | None = None  # (L, H): the T values the gadget used
    player: int | None = None               # the re-solving player (None for a plain solve)
    history: list = field(default_factory=list)  # (iteration, exploitability) at each check


def _as_groups(r, G: int, L: int | None = None) -> np.ndarray:
    """(H,), (L, H) or (G, L, H) root arrays -> (G, L, H) float64."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r[None]
    if r.ndim == 2:
        r = np.broadcast_to(r, (G,) + r.shape)
    return np.ascontiguousarray(r)


def _matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    total = pos.sum(-1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / regret.shape[-1])


def _average(c: np.ndarray) -> np.ndarray:
    total = c.sum(-1, keepdims=True)
    return np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1])


def _group_sum(x: np.ndarray, group: np.ndarray, G: int) -> np.ndarray:
    """(n, L, H) per deal -> (G, L, H) per subgame root, adding each group's deals in order
    (as close_values does)."""
    return _segment_sum(x, group, G)


def root_values(street, template, contrib, hsig, root, player: int, mode: str = "cfr", group=None,
                num_groups: int = 1) -> np.ndarray:
    """Player `player`'s root value per (line, hand): street_values at the root node, summed
    over each subgame's deals in deal order. (L, H) without `group`; with it, (G, L, H)."""
    nodes, _ = street_reach(template, hsig, root)
    V, _ = street_values(template, street, contrib, hsig, nodes, player, mode, True)
    if group is None:
        return _group_sum(V[0], np.zeros(len(V[0]), dtype=np.int64), 1)[0]
    return _group_sum(V[0], group, num_groups)


def blueprint_cfv(street, template, contrib, hsig, reach0, reach1, player: int, mode: str = "br", group=None,
                  num_groups: int = 1) -> np.ndarray:
    """`player`'s root values under a stored river strategy hsig[k] (n, L, H, A), from both
    players' root reaches ((H,) or (L, H), or (G, L, H) with `group`): the gadget's T values.
    mode "br" (best response within the subgame, the default) is what the safety guarantee
    needs; "cfr" gives the on-policy values."""
    g = np.zeros(street.num_deals, dtype=np.int64) if group is None else np.asarray(group)
    G = 1 if group is None else num_groups
    root = tuple(np.ascontiguousarray(_as_groups(r, G)[g]) for r in (reach0, reach1))
    return root_values(street, template, contrib, hsig, root, player, mode, group, num_groups)


def _solve(street, template, contrib, group, G: int, rp_root, ro_root, p: int, cfv, variant: str, iters: int,
           tol: float, check_every: int, alpha=1.5, beta=0.0, gamma=2.0) -> Subgame:
    """The CFR loop over G independent subgames at once: deal i of `street` belongs to subgame
    group[i]. A plain subgame (cfv None: o enters with reach ro_root) or the gadget (o's reach
    into the subgame is its F probability, T pays cfv). Alternating updates, player 0 first,
    with range_cfr's update kernel, so a plain solve from all-ones reach is RangeCFR on the
    subgame as its own public tree. Every operation is per deal or per subgame, so a subgame's
    iterates don't depend on the others in the batch; each subgame is frozen at the first check
    where its own exploitability is at most tol, so its result equals a solve on its own."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    start = time.perf_counter()
    o = 1 - p
    tp = template
    n = street.num_deals
    group = np.ascontiguousarray(group, dtype=np.int64)
    rp_root = _as_groups(rp_root, G)
    _, L, H = rp_root.shape
    rp = np.ascontiguousarray(rp_root[group])
    gadget = cfv is not None
    if gadget:
        cfv = _as_groups(cfv, G)
        g_regret = np.zeros((G, L, H, 2))
        g_cum = np.zeros((G, L, H, 2))
        g_sig = np.full((G, L, H, 2), 0.5)
    else:
        ro_root = _as_groups(ro_root, G)
        ro = np.ascontiguousarray(ro_root[group])
    regret = [np.zeros((n, L, H, len(a))) for a in tp.actions]
    cum = [np.zeros_like(r) for r in regret]
    sigma = [_matching(r) for r in regret]
    mass = _group_sum(street.fold(rp, o), group, G)  # (G, L, H): o's hands, chance x p's reach
    mass = (mass if gadget else mass * ro_root).sum(-1)
    v = VARIANTS[variant]
    history = [[] for _ in range(G)]
    t = 0

    def roots(sig_f):
        r_o = np.ascontiguousarray(sig_f[group]) if gadget else ro
        return (rp, r_o) if p == 0 else (r_o, rp)

    def evaluate():
        avg = [_average(c) for c in cum]
        g_avg = _average(g_cum) if gadget else None
        root = roots(g_avg[..., F] if gadget else None)
        br_o = root_values(street, tp, contrib, avg, root, o, "br", group, G)
        br_p = root_values(street, tp, contrib, avg, root, p, "br", group, G)
        on = [root_values(street, tp, contrib, avg, root, q, "cfr", group, G) for q in (0, 1)]
        best_p = (rp_root * br_p).sum(-1)  # p's best total: its root reach is fixed
        if gadget:
            best_o = np.maximum(br_o, cfv).sum(-1)
            best_p = best_p - (g_avg[..., T] * cfv).sum(-1)  # T pays o cfv whatever p does
        else:
            best_o = (ro_root * br_o).sum(-1)
        gap = best_o + best_p
        norm = np.divide(gap, mass, out=np.zeros_like(gap), where=mass > 0).max(axis=1)  # (G,)
        return avg, g_avg, br_o, br_p, on, gap, norm

    # The result, filled in per subgame as each one stops.
    out_sig = [np.empty_like(c) for c in cum]
    out_g = np.empty((G, L, H, 2)) if gadget else None
    out_br_o, out_br_p, out_gap = np.empty((G, L, H)), np.empty((G, L, H)), np.empty((G, L))
    out_on = [np.empty((G, L, H)) for _ in (0, 1)]
    out_expl, out_iters = np.empty(G), np.zeros(G, dtype=np.int64)
    done = np.zeros(G, dtype=bool)

    def freeze(which, result):
        avg, g_avg, br_o, br_p, on, gap, norm = result
        deals = which[group]
        for k in range(len(cum)):
            out_sig[k][deals] = avg[k][deals]
        if gadget:
            out_g[which] = g_avg[which]
        out_br_o[which], out_br_p[which], out_gap[which] = br_o[which], br_p[which], gap[which]
        for q in (0, 1):
            out_on[q][which] = on[q][which]
        out_expl[which], out_iters[which] = norm[which], t
        done[which] = True

    while t < iters and not done.all():
        t += 1
        weight = {"cfr": 1.0, "cfr+": float(t), "dcfr": float(t) ** gamma}[variant]
        pos = t ** alpha / (t ** alpha + 1.0)
        neg = t ** beta / (t ** beta + 1.0)
        for q in (0, 1):
            root = roots(g_sig[..., F] if gadget else None)
            nodes, _ = street_reach(tp, sigma, root)
            V, R = street_values(tp, street, contrib, sigma, nodes, q, "cfr", True)
            for k, owner in enumerate(tp.players):
                if owner != q:
                    continue
                A = regret[k].shape[-1]
                sigma[k] = np.ascontiguousarray(sigma[k])
                own = np.ascontiguousarray(nodes[k][q]).reshape(-1)
                _update(regret[k].size, regret[k].reshape(-1, A), np.ascontiguousarray(R[k]).reshape(-1, A),
                        sigma[k].reshape(-1, A), cum[k].reshape(-1, A), own, weight, pos, neg, v)
            if gadget and q == o:
                w = _group_sum(V[0], group, G)
                value = g_sig[..., F] * w + g_sig[..., T] * cfv
                inst = np.stack([w - value, cfv - value], axis=-1)
                _update(g_regret.size, g_regret.reshape(-1, 2), np.ascontiguousarray(inst).reshape(-1, 2),
                        g_sig.reshape(-1, 2), g_cum.reshape(-1, 2), np.ones(G * L * H), weight, pos, neg, v)
        if t % check_every == 0 or t == iters:
            result = evaluate()
            for g in np.flatnonzero(~done):
                history[g].append((t, float(result[-1][g])))
            stop = ~done & (result[-1] <= tol)
            if t == iters:
                stop = ~done
            if stop.any():
                freeze(stop, result)
    if not done.all():  # iters == 0
        freeze(~done, evaluate())
    seconds = time.perf_counter() - start
    br = (out_br_p, out_br_o) if p == 0 else (out_br_o, out_br_p)
    return Subgame(out_sig, tuple(out_on), br, out_gap, mass, out_expl, out_iters, seconds, out_g,
                   cfv if gadget else None, p if gadget else None, history)


def _single(sg: Subgame) -> Subgame:
    """A batch of one subgame as a plain Subgame: (L, H) root fields, scalar counts."""
    pick = lambda x: None if x is None else x[0]
    return Subgame(sg.strategy, tuple(x[0] for x in sg.root_cfv), tuple(x[0] for x in sg.br_cfv), sg.gap[0],
                   sg.mass[0], float(sg.exploitability[0]), int(sg.iterations[0]), sg.seconds, pick(sg.gadget),
                   pick(sg.opponent_cfv), sg.player, sg.history[0])


def solve_subgame(street, template, contrib, reach0, reach1, variant: str = "cfr+", iters: int = 1000,
                  tol: float = 0.0, check_every: int = 10) -> Subgame:
    """Solve one subgame from both players' root reaches ((H,) or (L, H) each). `street` holds
    the subgame's last-street deals (one turn deal's rivers), `template` its betting and
    `contrib` (L,) the chips each player has in on each line. Stops after `iters` iterations or
    once the average profile's NashConv per line, over its mass, is at most `tol` (chips),
    checked every `check_every` iterations. Root values are in solver units (module doc)."""
    group = np.zeros(street.num_deals, dtype=np.int64)
    return _single(_solve(street, template, contrib, group, 1, reach0, reach1, 0, None, variant, iters, tol,
                          check_every))


def resolve_batch(street, template, contrib, group, num_groups: int, reach_resolver, reach_opponent, opponent_cfv,
                  player: int, variant: str = "cfr+", iters: int = 1000, tol: float = 0.0, safe: bool = True,
                  check_every: int = 10) -> Subgame:
    """`resolve` for G = num_groups subgames at once (deal i of `street` in subgame group[i];
    reaches and opponent_cfv (G, L, H)). Each subgame's result equals `resolve` on it alone."""
    if safe:
        if opponent_cfv is None:
            raise ValueError("safe re-solving needs the opponent's blueprint values")
        out = _solve(street, template, contrib, group, num_groups, reach_resolver, None, player, opponent_cfv,
                     variant, iters, tol, check_every)
    else:
        r = (reach_resolver, reach_opponent) if player == 0 else (reach_opponent, reach_resolver)
        out = _solve(street, template, contrib, group, num_groups, r[0], r[1], 0, None, variant, iters, tol,
                     check_every)
        out.player = player
        if opponent_cfv is not None:
            out.opponent_cfv = _as_groups(opponent_cfv, num_groups)
    return out


def resolve(street, template, contrib, reach_resolver, reach_opponent, opponent_cfv, player: int,
            variant: str = "cfr+", iters: int = 1000, tol: float = 0.0, safe: bool = True,
            check_every: int = 10) -> Subgame:
    """Re-solve one subgame for `player` (p) from the blueprint's root reaches ((H,) or (L, H)).

    safe=True: the gadget. At the root the opponent o picks, per line and hand, F (enter, with
    reach equal to its F probability) or T (take opponent_cfv[l, h]); `reach_opponent` is not
    used by the solve. opponent_cfv (L, H) must be in solver units: o's root value with chance
    weight x p's reach x payoff, without o's reach, as `blueprint_cfv` gives it (mode "br" for
    the guarantee). safe=False: the subgame alone from both reaches (`solve_subgame`).

    Returns a Subgame whose `strategy[k]` at p's nodes is p's re-solved river strategy for every
    deal of the subgame, `gadget` o's (F, T) average, `br_cfv[o]` o's best-response root values
    to compare with opponent_cfv, and `gap` (L,) the NashConv of the game solved: for the gadget,
    sum_h max(0, br_cfv[o] - opponent_cfv) <= gap per line (module doc)."""
    group = np.zeros(street.num_deals, dtype=np.int64)
    return _single(resolve_batch(street, template, contrib, group, 1, reach_resolver, reach_opponent, opponent_cfv,
                                 player, variant, iters, tol, safe, check_every))


# A river provider for the exact evaluator (sampled_cfr.last_street_sums) --------------------


class Resolver:
    """provider(st, parents, children) -> hsig that re-solves the last street. For each deal d
    of the street before (each turn deal), the first time any of its children is asked for, it
    builds the subgame of all d's children, takes both players' root reaches from the
    blueprint's upper streets, computes the T values from the blueprint's last street
    (`blueprint`, a provider such as sampled_cfr.StoredStrategy) with blueprint_cfv, and
    re-solves once for each player in `players` (safe or unsafe); node k of the last template
    then plays the re-solve of its acting player (the blueprint for a player not in
    `players`). Deals are re-solved `batch` at a time (blocks d // batch), each subgame on its
    own terms (resolve_batch), so the result doesn't depend on the batch or on how chunks fall;
    the last `keep` blocks are cached, so one re-solve serves every child of d. With `deals`
    (a collection of deals of the street before) only those are re-solved, and every other
    deal plays the blueprint.
    `diagnostics[(d, p)]` records each re-solve:
      gap, mass         (L,) NashConv of the game solved and its normalizing mass (solver units)
      br, violation     (L, H) o's best-response root value, and it minus opponent_cfv
      iterations, exploitability (max over lines of gap / mass, chips),
      seconds           the block's solve time divided by the deals in it
    """

    def __init__(self, pt, last, sigma, blueprint, safe: bool = True, variant: str = "cfr+", iters: int = 1000,
                 tol: float = 0.0, check_every: int = 10, players=(0, 1), batch: int = 1, keep: int = 2,
                 deals=None):
        from bombpot.core.solvers.range_cfr import forward
        T = len(pt.templates)
        self.pt, self.last, self.blueprint = pt, last, blueprint
        self.tp, self.contrib = pt.templates[T - 1], pt.contrib[T - 1]
        self.safe, self.variant, self.iters, self.tol, self.check_every = safe, variant, iters, tol, check_every
        self.players, self.batch, self.keep = tuple(players), batch, keep
        self.num_parents = pt.streets[-1].num_deals
        self.deals = None if deals is None else {int(d) for d in deals}
        hsig_upper = [list(row) for row in sigma[:T - 1]]
        reach = forward(pt, hsig_upper)
        _, self.closing = street_reach(pt.templates[T - 2], hsig_upper[T - 2], reach[T - 2][0])
        self.cache: dict = {}
        self.diagnostics: dict = {}

    def root(self, d: int) -> tuple:
        """Both players' reach at deal d's last-street root, (L, H) each, laid out as next_root does."""
        H = self.closing[0][0].shape[-1]
        return tuple(np.stack([c[p][d] for c in self.closing], axis=1).reshape(-1, H) for p in (0, 1))

    def solve(self, deals) -> list:
        """Re-solve the subgames of `deals` (deals of the street before): hsig[k] shaped
        (len(deals) x children, L, H, A), deal by deal."""
        deals = np.asarray(deals, dtype=np.int64)
        G, n = len(deals), self.last.num_children
        parents, children = np.repeat(deals, n), np.tile(np.arange(n), G)
        group = np.repeat(np.arange(G), n)
        st = self.last.street(parents, children)
        bp = self.blueprint(st, parents, children)
        roots = [self.root(int(d)) for d in deals]
        r = tuple(np.stack([x[p] for x in roots]) for p in (0, 1))  # (G, L, H)
        out = [np.ascontiguousarray(s) for s in bp]
        for p in self.players:
            o = 1 - p
            cfv = blueprint_cfv(st, self.tp, self.contrib, bp, r[0], r[1], o, "br", group, G)
            res = resolve_batch(st, self.tp, self.contrib, group, G, r[p], r[o], cfv, p, self.variant, self.iters,
                                self.tol, self.safe, self.check_every)
            for k, q in enumerate(self.tp.players):
                if q == p:
                    out[k] = res.strategy[k]
            for g, d in enumerate(deals):
                self.diagnostics[(int(d), p)] = {
                    "gap": res.gap[g], "mass": res.mass[g], "violation": res.br_cfv[o][g] - cfv[g], "br": res.br_cfv[o][g],
                    "iterations": int(res.iterations[g]), "exploitability": float(res.exploitability[g]),
                    "seconds": res.seconds / G}
        return out

    def _block(self, b: int) -> list:
        if b not in self.cache:
            deals = np.arange(b * self.batch, min(self.num_parents, (b + 1) * self.batch))
            if self.deals is not None:
                deals = np.array([d for d in deals if int(d) in self.deals], dtype=np.int64)
            self.cache[b] = self.solve(deals)
            while len(self.cache) > self.keep:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[b]

    def __call__(self, st, parents, children) -> list:
        parents, children = np.asarray(parents), np.asarray(children)
        n = self.last.num_children
        out = None
        out = [np.empty((len(parents), len(self.contrib), len(self.pt.hands), len(a))) for a in self.tp.actions]
        for d in np.unique(parents):
            d = int(d)
            mask = parents == d
            if self.deals is not None and d not in self.deals:
                idx = np.flatnonzero(mask)
                hs = self.blueprint(st.subset(idx), parents[idx], children[idx])
                for k, s in enumerate(hs):
                    out[k][mask] = s
                continue
            b = d // self.batch
            hs = self._block(b)
            start = np.arange(b * self.batch, d)
            if self.deals is not None:
                start = [x for x in start if int(x) in self.deals]
            rows = len(start) * n + children[mask]
            for k, s in enumerate(hs):
                out[k][mask] = s[rows]
        return out
