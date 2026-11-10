"""Public chance sampling with range vectors, for games whose last street is too big to
enumerate (the river of double-board Hold'em: 3.9 million deals per pair of flops).

Each iteration samples `batch` deals of the second-to-last street (without replacement)
and `samples` children of each on the last street, and runs the range solver's passes
(range_cfr.forward and backward) on that sample. Chance weights are divided by the
probability of being sampled, so sampled counterfactual values are unbiased (Johanson
et al. 2012, "Efficient Nash equilibrium approximation through Monte Carlo
counterfactual regret minimization"). The average strategy is accumulated with the
player's own reach at the sampled infosets only, as in that paper.

The last street's infosets come from an abstraction: `rows(parents, children)` picks the
stored strategy a deal uses, and `ids(street, ...)` each hand's infoset within that row.
`Lossless` gives every deal its own row (an exact game, for checks on small decks);
`abstraction/river.py` shares a row across all river cards of a turn deal and buckets
hands by equity, as poker solvers do for the river.

Regrets and averages use linear weighting (iteration t counts t), as in Pluribus (Brown
and Sandholm 2019), which needs no discounting of the infosets a sample skips.

This solver handles games with one line class per street (core/public.py), which is every
limit game. Its storage stays per street and node, regret[t][k] shaped (D, L, ids, A) (the
layout of M1's checkpoints); the range solver's passes, which take [t][c][k], see it as
class 0.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
from numba import prange

from bombpot.core.jit import kernel
from bombpot.core.public import PublicTree
from bombpot.core.solvers.cfr import IterativeSolver
from bombpot.core.solvers.range_cfr import backward, class_reach, class_values, close_values, forward, next_root


@kernel
def _update(regret, instant, sigma, cum, own, w):
    """regret += w x instant, cum += w x own x sigma (the strategy just played), then regret
    matching. Rows are infosets."""
    N, A = regret.shape
    for n in prange(N):
        total = 0.0
        for a in range(A):
            cum[n, a] += w * own[n] * sigma[n, a]
            regret[n, a] += w * instant[n, a]
            if regret[n, a] > 0.0:
                total += regret[n, a]
        for a in range(A):
            sigma[n, a] = (max(regret[n, a], 0.0) / total) if total > 0.0 else 1.0 / A


def one_class(pt: PublicTree) -> list:
    """Each street's single LineClass (ValueError if a street has several)."""
    if any(len(row) != 1 for row in pt.classes):
        raise ValueError("the sampled solver needs one line class per street")
    return [row[0] for row in pt.classes]


def _classed(x: list) -> list:
    """[t][k] -> [t][0][k], as the range solver's passes take them."""
    return [[row] for row in x]


def _unclassed(x: list) -> list:
    """[t][0][k] -> [t][k]."""
    return [row[0] for row in x]


def _matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    total = pos.sum(-1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / regret.shape[-1])


class Lossless:
    """Every last-street deal is its own row and every hand its own infoset: the exact game."""

    def __init__(self, num_parents: int, num_children: int, num_hands: int):
        self.num_children = num_children
        self.num_rows, self.num_ids = num_parents * num_children, num_hands

    def rows(self, parents, children) -> np.ndarray:
        return np.asarray(parents) * self.num_children + np.asarray(children)

    def ids(self, street, parents, children, player: int) -> np.ndarray:
        return np.broadcast_to(np.arange(self.num_ids), (len(parents), self.num_ids))


class SampledRangeCFR(IterativeSolver):

    def __init__(self, pt: PublicTree, last, abstraction, batch: int, samples: int = 1, seed: int = 0):
        """pt holds every street but the last (pt.classes covers all of them);
        last.street(parents, children, scale) builds any set of last-street deals, and
        last.num_children is how many each deal of the street before has."""
        self.pt, self.last, self.abs = pt, last, abstraction
        self.cls = one_class(pt)
        self.T = len(self.cls)
        if pt.num_streets != self.T - 1:
            raise ValueError("pt must hold every street except the last")
        self.batch, self.samples, self.seed = batch, samples, seed
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.solve_seconds = 0.0
        self.value = [0.0, 0.0]
        H = len(pt.hands)
        self.regret, self.cum = [], []
        for t, lc in enumerate(self.cls):
            tp = lc.template
            if t < self.T - 1:
                shape = pt.shape(t, 0)
            else:
                shape = (abstraction.num_rows, lc.num_lines, abstraction.num_ids)
            self.regret.append([np.zeros(shape + (len(a),)) for a in tp.actions])
            self.cum.append([np.zeros(shape + (len(a),)) for a in tp.actions])
        self.sigma = [[_matching(r) for r in row] for row in self.regret]
        self.H = H

    # Sampling -------------------------------------------------------------------------

    def sample(self):
        """Deals for one iteration: `batch` deals of the second-to-last street, and for
        each, `samples` of its children. Returns (positions, parents, children, q1, q2)."""
        D = self.pt.streets[-1].num_deals
        S = np.sort(self.rng.choice(D, size=self.batch, replace=False))
        n = self.last.num_children
        children = np.stack([self.rng.choice(n, size=self.samples, replace=False) for _ in S]).ravel()
        positions = np.repeat(np.arange(len(S)), self.samples)
        return S, positions, children, self.batch / D, self.samples / n

    def view(self, S, positions, children, q1: float, q2: float):
        """The public tree of one sample (chance weights over sampling probabilities), the
        last street's rows, and each player's infoset ids there."""
        pt = self.pt
        streets = list(pt.streets[:-1]) + [pt.streets[-1].subset(S, 1.0 / q1)]
        last = self.last.street(S[positions], children, 1.0 / (q1 * q2))
        last.parent = positions  # index into the sampled deals of the street before
        view = PublicTree(pt.name, pt.hands, streets + [last], pt.classes)
        rows = self.abs.rows(S[positions], children)
        ids = [np.asarray(self.abs.ids(last, S[positions], children, p)) for p in (0, 1)]
        return view, rows, ids

    def view_sigma(self, sigma, S, rows, ids) -> list:
        """Per-hand strategies on a sample's view."""
        T = self.T
        out = [list(row) for row in sigma[:T - 2]]
        out.append([s[S] for s in sigma[T - 2]])
        tp = self.cls[T - 1].template
        out.append([s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[T - 1], tp.players)])
        return out

    # Iterations -------------------------------------------------------------------------

    def iteration(self) -> None:
        start = time.perf_counter()
        self.t += 1
        S, positions, children, q1, q2 = self.sample()
        view, rows, ids = self.view(S, positions, children, q1, q2)
        for p in (0, 1):  # alternating, both on the same sample
            hsig = _classed(self.view_sigma(self.sigma, S, rows, ids))
            reach = forward(view, hsig)
            V, R = backward(view, hsig, reach, p)
            self.value[p] = float(V[0][0][0].sum())
            reach, R = _unclassed(reach), _unclassed(R)
            self._apply(p, float(self.t), view, S, rows, ids[p], reach, R)
        self.solve_seconds += time.perf_counter() - start

    def instant_regrets(self, p: int, S, positions, children, q1, q2) -> list:
        """Player p's sampled instant regrets in storage layout (zero where not sampled), under
        the current strategies: for testing that their expectation is the full-width regret."""
        view, rows, ids = self.view(S, positions, children, q1, q2)
        hsig = _classed(self.view_sigma(self.sigma, S, rows, ids))
        reach = forward(view, hsig)
        _, R = backward(view, hsig, reach, p)
        R = _unclassed(R)
        out = [[np.zeros_like(r) for r in row] for row in self.regret]
        T = self.T
        for t, lc in enumerate(self.cls):
            for k, q in enumerate(lc.template.players):
                if q != p:
                    continue
                if t < T - 2:
                    out[t][k] += R[t][k]
                elif t == T - 2:
                    out[t][k][S] += R[t][k]
                else:
                    touched, sums = self._scatter(R[t][k], rows, ids[p], view.streets[-1].possible(p))
                    out[t][k][touched] += sums
        return out

    def _scatter(self, x: np.ndarray, rows, ids, possible):
        """Per-hand (n, L, H, ...) values summed into their infosets: returns the touched
        rows and their sums, (rows, L, num_ids, ...). Hands that can't be held are left out."""
        n, L, H = x.shape[:3]
        touched, inv = np.unique(rows, return_inverse=True)
        keep = possible.ravel()
        out = np.zeros((len(touched), self.abs.num_ids, L) + x.shape[3:])
        vals = np.moveaxis(x, 2, 1).reshape((n * H, L) + x.shape[3:])[keep]
        np.add.at(out, (np.repeat(inv.ravel(), H)[keep], ids.ravel()[keep]), vals)
        return touched, np.moveaxis(out, 1, 2)

    def _apply(self, p, w, view, S, rows, ids, reach, R) -> None:
        T = self.T
        for t, lc in enumerate(self.cls):
            tp = lc.template
            for k, q in enumerate(tp.players):
                if q != p:
                    continue
                A = len(tp.actions[k])
                own = reach[t][k][p]
                if t < T - 2:
                    reg, sig, cum, inst = self.regret[t][k], self.sigma[t][k], self.cum[t][k], R[t][k]
                elif t == T - 2:
                    reg, sig, cum, inst = (self.regret[t][k][S], self.sigma[t][k][S], self.cum[t][k][S], R[t][k])
                else:  # sum hands into their infosets; own reach sums too (the average weights histories)
                    possible = view.streets[-1].possible(p)
                    touched, inst = self._scatter(R[t][k], rows, ids, possible)
                    _, own = self._scatter(own, rows, ids, possible)
                    reg, sig, cum = self.regret[t][k][touched], self.sigma[t][k][touched], self.cum[t][k][touched]
                reg, sig, cum = np.ascontiguousarray(reg), np.ascontiguousarray(sig), np.ascontiguousarray(cum)
                _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(inst).reshape(-1, A), sig.reshape(-1, A),
                        cum.reshape(-1, A), np.ascontiguousarray(own).reshape(-1), w)
                if t < T - 2:
                    self.regret[t][k], self.sigma[t][k], self.cum[t][k] = reg, sig, cum
                elif t == T - 2:
                    self.regret[t][k][S], self.sigma[t][k][S], self.cum[t][k][S] = reg, sig, cum
                else:
                    self.regret[t][k][touched], self.sigma[t][k][touched], self.cum[t][k][touched] = reg, sig, cum

    def average_strategy(self) -> list:
        out = []
        for row in self.cum:
            out.append([])
            for c in row:
                total = c.sum(-1, keepdims=True)
                out[-1].append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
        return out

    def log_row(self) -> dict:
        """Exact exploitability over every last-street deal: about 2 s on the small test game,
        but about 14 hours (8 cores) for the real deck's 3.9 million river deals."""
        return {"iteration": self.t, "wall_time": self.solve_seconds,
                **exact_exploitability(self.pt, self.last, self.abs, self.average_strategy())}

    # Checkpoints ------------------------------------------------------------------------

    _SCALARS = ("t", "solve_seconds", "value", "rng", "batch", "samples", "seed")

    def _arrays(self) -> dict:
        """Every regret, average and strategy array, keyed by name, street and node."""
        return {f"{name}_t{t}_k{k}": a for name in ("regret", "cum", "sigma")
                for t, row in enumerate(getattr(self, name)) for k, a in enumerate(row)}

    def save(self, path) -> None:
        """Write the full state to one .npz: every regret, average and strategy array (the
        strategy itself, not recomputed, so a resumed run is bit-identical), t, solve_seconds,
        value, the random generator's state (as JSON), batch, samples and seed. Uncompressed,
        since checkpoints are large. It goes to a temporary file that is then renamed, so a
        crash never leaves a half-written checkpoint."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, t=self.t, solve_seconds=self.solve_seconds, value=np.array(self.value),
                     rng=np.array(json.dumps(self.rng.bit_generator.state)), batch=self.batch,
                     samples=self.samples, seed=np.array(json.dumps(self.seed, default=int)), **self._arrays())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load(self, path) -> None:
        """Restore a state written by save() into this solver, which must be built with the
        same public tree, last street and abstraction (batch, samples and seed are restored
        from the file). Refuses with a ValueError, before changing anything, if the file's
        arrays differ from this solver's in number or shape."""
        want = self._arrays()
        with np.load(path) as z:
            missing = [key for key in self._SCALARS if key not in z.files]
            names = {key for key in z.files if key not in self._SCALARS}
            if missing or names != set(want):
                raise ValueError(f"{path}: not a checkpoint of this solver: it has {len(names)} arrays and this "
                                 f"solver {len(want)}{f', and it lacks {missing}' if missing else ''}")
            arrays = {}
            for key, a in want.items():
                b = z[key]
                if b.shape != a.shape or b.dtype != a.dtype:
                    raise ValueError(f"{path}: {key} is {b.shape} {b.dtype} in the checkpoint but {a.shape} "
                                     f"{a.dtype} in this solver: built with another public tree or abstraction?")
                arrays[key] = b
            scalars = {key: z[key] for key in self._SCALARS}
        for name in ("regret", "cum", "sigma"):
            setattr(self, name, [[arrays[f"{name}_t{t}_k{k}"] for k in range(len(row))]
                                 for t, row in enumerate(getattr(self, name))])
        self.t = int(scalars["t"])
        self.solve_seconds = float(scalars["solve_seconds"])
        self.value = [float(v) for v in scalars["value"]]
        self.rng.bit_generator.state = json.loads(str(scalars["rng"]))
        self.batch, self.samples = int(scalars["batch"]), int(scalars["samples"])
        self.seed = json.loads(str(scalars["seed"]))


# Exact evaluation, streaming the last street ------------------------------------------------
# The last street's values are summed into the closing lines of the street before, per deal of
# that street (last_street_sums), then the streets above are walked bottom-up
# (finish_exploitability). The streaming can be split into shards (split_chunks) whose sums add
# up to the unsplit ones bit for bit, so long evaluations can run as parallel jobs.

_JOBS = ((0, "br"), (1, "br"), (0, "cfr"))  # player 0's and player 1's best responses, then the value


class StoredStrategy:
    """The default river provider: a stored last-street strategy looked up through the
    abstraction's rows and ids. provider(st, parents, children) -> hsig[c][k], the per-hand
    strategy (n, L_c, H, A) at node k of each line class c of the last street, for the
    last-street deals (parents[i], children[i]) whose CardStreet is st. `sigma_last[c][k]` is
    (rows, L_c, ids, A) and `players[c]` the acting players of class c's template; the sampled
    solver's one-class storage sigma[-1][k] is StoredStrategy.one_class(...)."""

    def __init__(self, abstraction, sigma_last, players):
        self.abs, self.sigma, self.players = abstraction, sigma_last, [list(p) for p in players]

    @classmethod
    def one_class(cls, abstraction, sigma_last_k, players_k) -> "StoredStrategy":
        return cls(abstraction, [sigma_last_k], [players_k])

    def __call__(self, st, parents, children) -> list:
        rows = self.abs.rows(parents, children)
        ids = [np.asarray(self.abs.ids(st, parents, children, p)) for p in (0, 1)]
        return [[s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sig, players)]
                for sig, players in zip(self.sigma, self.players)]


def exact_exploitability(pt: PublicTree, last, abstraction, sigma, chunk: int = 256, progress=None,
                         progress_every: int = 1, provider=None) -> dict:
    """Best responses and value of a stored strategy in the full game, every last-street
    deal included, streamed `chunk` deals at a time so memory stays bounded. If given,
    progress(chunks done, total chunks) is called every `progress_every` chunks and after
    the last, so long evaluations can log an ETA. `provider` (see last_street_sums) supplies
    the last street's strategy; by default the stored one, sigma[-1] through `abstraction`."""
    sums = last_street_sums(pt, last, abstraction, sigma, chunk, progress=progress, progress_every=progress_every,
                            provider=provider)
    return finish_exploitability(pt, sigma, sums)


def split_chunks(num_parents: int, num_children: int, chunk: int, shards: int) -> list:
    """The chunks of an exact evaluation (`chunk` last-street deals each, in order) split
    into `shards` ranges of about equal size. Every range starts at a chunk that begins a
    new deal of the street before, so no deal's sums are split between shards, and the
    shards' sums add up to the unsplit ones bit for bit."""
    num_chunks = -(-num_parents * num_children // chunk)
    step = math.lcm(chunk, num_children) // chunk  # chunks from one such start to the next
    edges = [min(num_chunks, round(i * num_chunks / (shards * step)) * step) for i in range(shards)]
    ranges = [range(a, b) for a, b in zip(edges, edges[1:] + [num_chunks])]
    if shards < 1 or any(len(r) == 0 for r in ranges):
        raise ValueError(f"{num_chunks} chunks of {chunk} can't be split into {shards} shards that each start "
                         f"with a new deal of the street before (one every {step} chunks)")
    return ranges


def zero_sums(pt: PublicTree) -> list:
    """Empty last-street sums, one per job, each (D1, L, closes, H)."""
    cls = one_class(pt)
    T = len(cls)
    shape = pt.shape(T - 2, 0) + (cls[T - 2].template.num_closes,)
    return [np.moveaxis(np.zeros(shape), 3, 2) for _ in _JOBS]


def last_street_sums(pt: PublicTree, last, abstraction, sigma, chunk: int = 256, chunks=None, progress=None,
                     progress_every: int = 1, provider=None) -> list:
    """The streamed part of exact_exploitability: for each job, the last street's values
    summed into the closing lines of the street before, per deal of that street (as in
    zero_sums). `chunks`, a range of chunk numbers (default every chunk), streams only
    those: one shard of a split evaluation.

    provider(st, parents, children) -> hsig[c][k], per line class c of the last street and node
    k of its template, (n, L_c, H, A) per-hand strategies for the chunk's deals. The default is
    StoredStrategy.one_class(abstraction, sigma[-1], players), today's lookup; a re-solving
    provider (core/solvers/resolve.Resolver) re-solves each deal of the street before once and
    serves all of its children. Chunks are streamed in order, so a deal's children arrive
    together (split_chunks starts every shard on a new deal). This evaluator itself still needs
    one class per street (one_class)."""
    cls = one_class(pt)
    T = len(cls)
    D1 = pt.streets[-1].num_deals
    hsig_upper = [list(row) for row in sigma[:T - 1]]
    reach = _unclassed(forward(pt, _classed(hsig_upper)))
    _, closing = class_reach(cls[T - 2].template, hsig_upper[T - 2], reach[T - 2][0])
    tp_last = cls[T - 1].template
    if provider is None:
        provider = StoredStrategy.one_class(abstraction, sigma[T - 1], tp_last.players)
    agg = zero_sums(pt)
    total = D1 * last.num_children
    if chunks is None:
        chunks = range(-(-total // chunk))
    for done, c in enumerate(chunks, 1):
        flat = np.arange(c * chunk, min(total, (c + 1) * chunk))
        parents, children = np.divmod(flat, last.num_children)
        st = last.street(parents, children)
        root = next_root([closing], cls[T - 2:T - 1], cls[T - 1:], st.parent)[0]
        (hsig,) = provider(st, parents, children)
        nodes, _ = class_reach(tp_last, hsig, root)
        for j, (p, mode) in enumerate(_JOBS):
            V, _ = class_values(tp_last, st, cls[T - 1].contrib, hsig, nodes, p, mode, True)
            agg[j] += close_values([V[0]], cls[T - 2:T - 1], cls[T - 1:], st.parent, D1)[0]
        if progress is not None and (done % progress_every == 0 or done == len(chunks)):
            progress(done, len(chunks))
    return agg


def finish_exploitability(pt: PublicTree, sigma, sums) -> dict:
    """exact_exploitability's result from the last street's sums (last_street_sums, or the
    shards' sums added into zero_sums): the streets above, bottom-up."""
    cls = one_class(pt)
    T = len(cls)
    hsig_upper = [list(row) for row in sigma[:T - 1]]
    reach = _unclassed(forward(pt, _classed(hsig_upper)))
    out = []
    for j, (p, mode) in enumerate(_JOBS):
        a = sums[j]
        for t in range(T - 2, -1, -1):
            V, _ = class_values(cls[t].template, pt.streets[t], cls[t].contrib, hsig_upper[t], reach[t], p, mode,
                                False, a)
            if t > 0:
                a = close_values([V[0]], cls[t - 1:t], cls[t:t + 1], pt.streets[t].parent,
                                 pt.streets[t - 1].num_deals)[0]
        out.append(float(V[0].sum()))
    br0, br1, value = out
    return {"exploitability": (br0 + br1) / 2, "br_value_p0": br0, "br_value_p1": br1, "game_value": value}
