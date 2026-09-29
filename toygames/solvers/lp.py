"""Sequence-form LP (Koller, Megiddo & von Stengel 1994), solved with HiGHS.

For player 0, with E x = e the realization-plan constraints of player 0, F y = f
those of player 1, and A the sequence-form payoff matrix of player 0:

    maximize f^T v   subject to  E x = e,  F^T v - A^T x <= 0,  x >= 0.

The optimum is the game value and x is a maximin realization plan. Player 1 solves
the same LP with the roles swapped and payoff -A^T.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from toygames.tree import Tree, behavioural_from_plan

HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def constraint_matrix(tree: Tree, p: int):
    """(E, e) with E x = e, x >= 0 exactly when x is a realization plan of player p."""
    n_inf, n = tree.n_infosets[p], tree.n_seqs[p]
    rows = np.concatenate(([0], 1 + np.arange(n_inf), 1 + tree.seq_infoset[p][1:]))
    cols = np.concatenate(([0], tree.parent_seq[p], np.arange(1, n)))
    vals = np.concatenate(([1.0], -np.ones(n_inf), np.ones(n - 1)))
    e = np.zeros(1 + n_inf)
    e[0] = 1.0
    return sp.csr_matrix((vals, (rows, cols)), shape=(1 + n_inf, n)), e


def payoff_matrix(tree: Tree) -> sp.csr_matrix:
    """A[s0, s1] = sum over terminals ending player 0 at s0 and player 1 at s1 of chance x u0."""
    return sp.csr_matrix((tree.term_cu0, (tree.term_seq[0], tree.term_seq[1])),
                         shape=(tree.n_seqs[0], tree.n_seqs[1]))


def solve_lp(tree: Tree, player: int = 0, options: dict | None = None):
    """Maximin strategy of `player`. Returns (value to `player`, behavioural strategy, plan)."""
    p, o = player, 1 - player
    E, e = constraint_matrix(tree, p)
    F, f = constraint_matrix(tree, o)
    A = payoff_matrix(tree)
    A_p = A if p == 0 else -A.T.tocsr()
    n_x, n_v = tree.n_seqs[p], F.shape[0]
    c = np.concatenate((np.zeros(n_x), -f))
    A_eq = sp.hstack([E, sp.csr_matrix((E.shape[0], n_v))]).tocsr()
    A_ub = sp.hstack([-A_p.T, F.T]).tocsr()
    bounds = [(0, None)] * n_x + [(None, None)] * n_v
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=e, bounds=bounds,
                  method="highs", options={**HIGHS_OPTIONS, **(options or {})})
    if res.status != 0:
        raise RuntimeError(f"LP for player {p} failed: {res.message}")
    x = np.maximum(res.x[:n_x], 0.0)
    return float(-res.fun), behavioural_from_plan(tree, p, x), x


def solve_game(tree: Tree, options: dict | None = None):
    """Solve both players' LPs. Returns (value to player 0 from each LP, equilibrium profile)."""
    v0, sigma0, _ = solve_lp(tree, 0, options)
    v1, sigma1, _ = solve_lp(tree, 1, options)
    return (v0, -v1), [sigma0, sigma1]
