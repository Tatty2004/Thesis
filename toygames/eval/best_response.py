"""Exact best response on a compiled tree."""
from __future__ import annotations

import numpy as np

from toygames.tree import Tree, realization_plan


def best_response(tree: Tree, player: int, sigma_opp: np.ndarray, return_strategy: bool = False):
    """Value of the best response of `player` to the opponent's behavioural strategy.

    Works bottom-up over the player's own sequence tree: a sequence is worth the
    counterfactual payoff of the terminals it ends at (opponent reach x chance reach
    x payoff) plus the best action value of each infoset that follows it. Returns
    (value, pure best-response strategy or None).
    """
    p, o = player, 1 - player
    n = tree.n_seqs[p]
    x_o = realization_plan(tree, o, sigma_opp)
    v = np.bincount(tree.term_seq[p], weights=tree.term_cu[p] * x_o[tree.term_seq[o]], minlength=n)
    br = np.zeros(n) if return_strategy else None
    if br is not None:
        br[0] = 1.0
    fs, na, ps = tree.first_seq[p], tree.num_actions[p], tree.parent_seq[p]
    li, ls = tree.level_infosets[p], tree.level_seqs[p]
    for lvl in range(len(li) - 2, -1, -1):
        ilo, ihi, lo, hi = li[lvl], li[lvl + 1], ls[lvl], ls[lvl + 1]
        best = np.maximum.reduceat(v[lo:hi], fs[ilo:ihi] - lo)
        if br is not None:
            idx = np.arange(lo, hi)
            is_best = v[lo:hi] == np.repeat(best, na[ilo:ihi])
            chosen = np.minimum.reduceat(np.where(is_best, idx, hi), fs[ilo:ihi] - lo)
            br[chosen] = 1.0
        v += np.bincount(ps[ilo:ihi], weights=best, minlength=n)
    return float(v[0]), br
