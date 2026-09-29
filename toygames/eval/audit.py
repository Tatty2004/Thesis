"""Dominated-action audit of a strategy profile.

At an infoset where the player never acts again after either of two actions a and b
(fold and call facing a bet on the last street, say), b dominates a when b's worst
payoff beats a's best payoff over every terminal either can reach, so against every
opponent hand and every opponent continuation. Folding a hand that can't lose and
calling with a hand that can't win are the poker examples.

Moving the probability sigma(I, a) onto b leaves the player's reach to every other
infoset unchanged, and gains at least

    pi_i(I) * sigma(I, a) * pi_-i(I) * gap,     gap = min payoff(b) - max payoff(a) > 0,

where pi_i(I) is the player's own reach and pi_-i(I) the opponent-and-chance reach.
Such moves at different infosets add up, so over all dominated actions of player i

    sum of those bounds  <=  BR_i(sigma_-i) - u_i(sigma)  <=  NashConv.

A profile breaking this means a best response was computed wrongly; one meeting it
plays dominated actions only with tiny probability or where it is almost never reached.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toygames.tree import Tree, realization_plan


@dataclass(frozen=True)
class Dominated:
    player: int
    key: str
    action: str    # the dominated action
    better: str    # an action that dominates it (the one with the largest gap)
    prob: float    # sigma(I, action)
    gap: float     # min payoff after `better` - max payoff after `action`
    reach: float   # pi_i(I) * pi_-i(I), the probability of reaching I
    bound: float   # reach * prob * gap: the least the player gains by switching


def dominated_actions(tree: Tree, sigma) -> list[Dominated]:
    """Every (infoset, action) whose action is dominated by another final action there."""
    if tree.term_chance is None:
        raise ValueError("the audit needs per-terminal chance and payoff (a compiled full game)")
    found = []
    for p in (0, 1):
        o, n = 1 - p, tree.n_seqs[p]
        ts = tree.term_seq[p]
        u = tree.term_u0 if p == 0 else -tree.term_u0
        lo = np.full(n, np.inf)
        hi = np.full(n, -np.inf)
        np.minimum.at(lo, ts, u)
        np.maximum.at(hi, ts, u)
        final = np.ones(n, dtype=bool)  # the player never acts again after the sequence
        final[0] = False
        final[tree.parent_seq[p]] = False
        x_p = realization_plan(tree, p, sigma[p])
        x_o = realization_plan(tree, o, sigma[o])
        # For a final sequence, the opponent-and-chance weight of its terminals sums to pi_-i(I).
        opp_reach = np.bincount(ts, weights=tree.term_chance * x_o[tree.term_seq[o]], minlength=n)
        fs, na, ps = tree.first_seq[p], tree.num_actions[p], tree.parent_seq[p]
        candidates = np.flatnonzero(np.add.reduceat(final[1:].astype(np.int64), fs - 1) >= 2)
        for i in candidates:
            seqs = [s for s in range(fs[i], fs[i] + na[i]) if final[s]]
            for a in seqs:
                better = max(seqs, key=lambda b: lo[b] - hi[a])
                gap = lo[better] - hi[a]
                if gap <= 0:
                    continue
                reach = x_p[ps[i]] * opp_reach[a]
                prob = sigma[p][a]
                found.append(Dominated(p, tree.keys[p][i], tree.actions[p][i][a - fs[i]],
                                       tree.actions[p][i][better - fs[i]], float(prob), float(gap),
                                       float(reach), float(reach * prob * gap)))
    return found


def audit(tree: Tree, sigma, br_value: tuple, game_value: float) -> dict:
    """Summary of dominated_actions, checked against each player's best-response gain."""
    found = dominated_actions(tree, sigma)
    gains = (br_value[0] - game_value, br_value[1] + game_value)
    out = {}
    for p in (0, 1):
        mine = [d for d in found if d.player == p]
        bound = sum(d.bound for d in mine)
        kinds = {}
        for d in mine:
            kinds[f"{d.action}<{d.better}"] = kinds.get(f"{d.action}<{d.better}", 0) + 1
        reach = sum(d.reach for d in mine)
        out[f"p{p}"] = {
            "dominated": len(mine),
            "kinds": kinds,
            "reach": reach,  # chance per hand of standing at one of these spots
            # How often the dominated action is played there, weighted by how often each spot comes up.
            "mistake_rate": sum(d.reach * d.prob for d in mine) / reach if reach > 0 else 0.0,
            "bound_sum": bound,
            "bound_max": max((d.bound for d in mine), default=0.0),
            "br_gain": gains[p],
            "ok": bound <= gains[p] + 1e-12,
        }
    out["ok"] = out["p0"]["ok"] and out["p1"]["ok"]
    return out
