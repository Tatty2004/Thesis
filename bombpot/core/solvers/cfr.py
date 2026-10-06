"""CFR, CFR+ and DCFR on a compiled tree, vectorised over sequences."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from bombpot.core.eval.exploitability import exploitability
from bombpot.core.tree import Tree, normalize, realization_plan

# Frozen after step 4: every solver log has exactly these columns.
LOG_FIELDS = ("iteration", "wall_time", "exploitability", "br_value_p0", "br_value_p1", "game_value")


def dcfr_discount(regret: np.ndarray, t: int, alpha: float, beta: float) -> np.ndarray:
    """After iteration t: positive regrets x t^alpha/(t^alpha+1), negative x t^beta/(t^beta+1)."""
    pos = t**alpha / (t**alpha + 1.0)
    neg = t**beta / (t**beta + 1.0)
    regret *= np.where(regret > 0, pos, neg)
    return regret


class IterativeSolver:
    """The shared driver: subclasses provide iteration(), log_row(), t and solve_seconds."""

    def run(self, iterations: int | None = None, tol: float | None = None, max_iterations: int = 100_000,
            log_every: int = 10, log_path=None, verbose: bool = False) -> list[dict]:
        """Iterate to `iterations`, or until the average strategy's exploitability is <= tol.

        Every `log_every` iterations (and at the end) it logs the iteration, wall time,
        exploitability, each player's best-response value and the game value, all
        for the average strategy. wall_time is seconds spent iterating, so the
        logging itself isn't counted. Rows go to `log_path` as CSV when given.
        Returns the rows.
        """
        if iterations is None and tol is None:
            raise ValueError("give iterations, tol, or both")
        target = iterations if iterations is not None else max_iterations
        rows = []
        f = writer = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            new = not Path(log_path).exists()
            f = open(log_path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                writer.writeheader()
        try:
            while self.t < target:
                self.iteration()
                if self.t % log_every and self.t != target:
                    continue
                row = self.log_row()
                rows.append(row)
                if writer is not None:
                    writer.writerow(row)
                    f.flush()
                if verbose:
                    print(f"  it {row['iteration']:>6}  {row['wall_time']:8.2f}s  "
                          f"expl {row['exploitability']:.3e}  value {row['game_value']:+.6f}", flush=True)
                if tol is not None and row["exploitability"] <= tol:
                    break
        finally:
            if f is not None:
                f.close()
        return rows

    def log_row(self) -> dict:
        raise NotImplementedError


class CFR(IterativeSolver):
    """Counterfactual regret minimisation with alternating updates.

    variant "cfr":  regret matching, uniform average
    variant "cfr+": regret matching+ (regrets floored at 0), average weighted by t
    variant "dcfr": Brown & Sandholm (2019). After iteration t, positive regrets are
                    scaled by t^alpha/(t^alpha+1) and negative ones by t^beta/(t^beta+1).
                    Iteration t's average-strategy contribution is weighted by t^gamma,
                    which equals scaling the running sum by (t/(t+1))^gamma every iteration.

    Counterfactual values use opponent reach x chance reach. Player 0 updates first,
    and player 1's update sees player 0's new strategy. Each player's average is
    accumulated from their own reach, at the strategy they held before updating.
    """

    def __init__(self, tree: Tree, variant: str = "dcfr", alpha: float = 1.5, beta: float = 0.0,
                 gamma: float = 2.0, alternating: bool = True):
        if variant not in ("cfr", "cfr+", "dcfr"):
            raise ValueError(f"unknown variant {variant!r}")
        self.tree, self.variant = tree, variant
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.alternating = alternating
        self.t = 0
        self.solve_seconds = 0.0
        self.regret = [np.zeros(n) for n in tree.n_seqs]
        self.cum_x = [np.zeros(n) for n in tree.n_seqs]
        self.sigma = tree.uniform()
        self.value = [0.0, 0.0]  # each player's payoff at their last update

    # Strategies -------------------------------------------------------------------

    def current_strategy(self) -> list[np.ndarray]:
        return [s.copy() for s in self.sigma]

    def average_strategy(self) -> list[np.ndarray]:
        return [normalize(self.tree, p, self.cum_x[p]) for p in (0, 1)]

    # Iterations -------------------------------------------------------------------

    def iteration(self) -> None:
        start = time.perf_counter()
        self.t += 1
        t = self.t
        weight = {"cfr": 1.0, "cfr+": float(t), "dcfr": float(t) ** self.gamma}[self.variant]
        if self.alternating:
            for p in (0, 1):
                r = self._instant_regret(p)
                self._accumulate(p, weight)
                self._update(p, r, t)
        else:
            rs = [self._instant_regret(p) for p in (0, 1)]
            for p in (0, 1):
                self._accumulate(p, weight)
                self._update(p, rs[p], t)
        self.solve_seconds += time.perf_counter() - start

    def _instant_regret(self, p: int) -> np.ndarray:
        """Counterfactual value of every sequence of p minus the value of its infoset."""
        tree, o, n = self.tree, 1 - p, self.tree.n_seqs[p]
        x_o = realization_plan(tree, o, self.sigma[o])
        v = np.bincount(tree.term_seq[p], weights=tree.term_cu[p] * x_o[tree.term_seq[o]], minlength=n)
        sig, r = self.sigma[p], np.zeros(n)
        fs, na, ps = tree.first_seq[p], tree.num_actions[p], tree.parent_seq[p]
        li, ls = tree.level_infosets[p], tree.level_seqs[p]
        for lvl in range(len(li) - 2, -1, -1):
            ilo, ihi, lo, hi = li[lvl], li[lvl + 1], ls[lvl], ls[lvl + 1]
            ev = np.add.reduceat(sig[lo:hi] * v[lo:hi], fs[ilo:ihi] - lo)
            r[lo:hi] = v[lo:hi] - np.repeat(ev, na[ilo:ihi])
            v += np.bincount(ps[ilo:ihi], weights=ev, minlength=n)
        self.value[p] = float(v[0])
        return r

    def _accumulate(self, p: int, weight: float) -> None:
        self.cum_x[p] += weight * realization_plan(self.tree, p, self.sigma[p])

    def _update(self, p: int, r: np.ndarray, t: int) -> None:
        regret = self.regret[p]
        regret += r
        if self.variant == "cfr+":
            np.maximum(regret, 0.0, out=regret)
        elif self.variant == "dcfr":
            dcfr_discount(regret, t, self.alpha, self.beta)
        self.sigma[p] = normalize(self.tree, p, np.maximum(regret, 0.0))

    def log_row(self) -> dict:
        rep = exploitability(self.tree, self.average_strategy())
        return {"iteration": self.t, "wall_time": self.solve_seconds, "exploitability": rep.exploitability,
                "br_value_p0": rep.br_value[0], "br_value_p1": rep.br_value[1], "game_value": rep.game_value}
