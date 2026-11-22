"""Live re-solving for the strategy viewer (viewer_server --live): the flop from the stored
trunk, the turn safely re-solved when its cards are known, and the river re-solved on top of
that turn re-solve, DeepStack's continual re-solving one street at a time.

Turn (LiveSolver.turn). For a turn deal d and the flop line that reaches it, each player p's
turn is re-solved with the gadget (core/solvers/leaf_resolve.LeafGadget via turn_resolve.Spot):
p's reach at the turn root from the stored flop strategy, the value network valuing every
turn end, the opponent o held to its T values at the turn root. T values (turn_resolve's
--cfv): 'avg', the trunk's turn-root values averaged over its iterations (turncfv.npy, written
by real_trunk --turn-cfv; the source that passed the small-deck gate T), else stand-ins from
what every trunk stores: 'backup' (the averaged leaf values, leafcfv.npy, backed up through the
average turn strategy) or 'onpolicy' (the average turn strategy's values, one network call).
Node k of the turn plays the re-solve of its acting player. `keep_best` ends each re-solve on
its check with the smallest gadget gap (LeafGadget.solve).

River. A river re-solve for p after this turn starts from p's reach under p's re-solved turn
and holds o to the values o got from the network at that turn close in p's turn re-solve,
averaged over the turn re-solve's iterations (LeafGadget.leaf_average: the turn re-solve's own
values, not the trunk's). `river_inputs` gives both; viewer_server runs the river gadget (all
river cards of the line at once, as the grade does) or its quick single-river solve with them.
The turn-end network gives one value per hand at the turn close, an average over the river
cards, so the safe river gadget is over all the river cards of the line together (the
opponent chooses between entering and its T value before the river card is known); a
single-river gadget would need values per river card, which this network does not give.

Flop (the hook). The flop is the stored trunk's (`flop_strategy`). With a network at the end
of the flop, the flop would be solved here from the root (no gadget: nothing comes before it),
and its averaged values at the turn roots would become the turn re-solve's T values the way
leaf_average feeds the river; `flop_values` is where they would come in.

Needs torch. Device-agnostic: on Apple's GPU (mps) the network runs on the device and NetLeaf's
float64 work on the host (turn_resolve.OnDevice); the turn tree is numpy on the CPU anyway.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.leaf_resolve import LeafGadget
from bombpot.experiments.valuenet.turn_resolve import Spot


class PairedLeaf:
    """One network pass for two LeafGadgets run in lockstep on two threads (both players'
    re-solves of one turn root): each gadget's leaf call waits for the other's, the two closing
    reaches go to the network as two lines of one batch, and each gets its half back. The two
    gadgets must make the same calls in the same order (same iterations, checks and update
    mode), which both players' re-solves of one root do. On a GPU a network call of 26 rows
    costs about what one of 13 does, so this nearly halves the time of the pair; the values are
    those of two separate calls (rows are independent; to the device's rounding)."""

    def __init__(self, leaf):
        self.leaf = leaf  # NetLeaf over the same line twice
        self.cv = threading.Condition()
        self.req, self.res, self.gen, self.failed = [None, None], [None, None], 0, None

    def side(self, i: int):
        pair = self

        class Side:
            def values(self, q, t, closing):
                return pair._call(i, ("one", q, t), closing)

            def values_both(self, t, closing):
                return pair._call(i, ("both", None, t), closing)

        return Side()

    def fail(self, err) -> None:
        with self.cv:
            self.failed = err
            self.cv.notify_all()

    def _call(self, i: int, kind, closing):
        with self.cv:
            if self.failed is not None:
                raise RuntimeError("the paired re-solve failed") from self.failed
            self.req[i] = (kind, closing)
            if self.req[1 - i] is None:
                gen = self.gen
                self.cv.wait_for(lambda: self.gen != gen or self.failed is not None)
                if self.failed is not None:
                    raise RuntimeError("the paired re-solve failed") from self.failed
                return self.res[i]
            (k0, c0), (k1, c1) = self.req
            if k0 != k1:
                err = RuntimeError(f"paired gadgets out of step: {k0} vs {k1}")
                self.failed = err
                self.cv.notify_all()
                raise err
            cl = [tuple(np.concatenate([c0[0][j], c1[0][j]], axis=1) for j in (0, 1))]
            kind, q, t = k0
            try:
                out = self.leaf.values_both(t, cl) if kind == "both" else (self.leaf.values(q, t, cl),)
            except BaseException as e:
                self.failed = e
                self.cv.notify_all()
                raise
            halves = [[[x[0][:, g:g + 1]] for x in out] for g in (0, 1)]  # [gadget][player][class]
            self.res = [tuple(h) if kind == "both" else h[0] for h in halves]
            self.req = [None, None]
            self.gen += 1
            self.cv.notify_all()
            return self.res[i]


def file_tag(path, n: int = 8) -> str:
    """A short content hash of a file (the network checkpoint) for cache names."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


class LiveSolver:
    """Live turn re-solves of one trunk directory (module doc).

    iters, check_every   turn re-solve length and how often its gap is checked (and kept,
                         with keep_best); progress is reported at the checks
    cfv                  T-value source: auto (avg if turncfv.npy, else backup if leafcfv,
                         else onpolicy), avg, backup, onpolicy or br
    simultaneous         one network pass per iteration (LeafGadget's)
    paired               both players' re-solves share their network calls (PairedLeaf)
    """

    def __init__(self, trunk, device="cuda", model=None, cfv="auto", iters=300, check_every=20,
                 simultaneous=False, keep_best=True, amp=None, built=None, paired=True):
        self.spot = Spot(trunk, device, model, amp, built=built)
        self.cfv = self.spot.cfv_source(cfv)
        self.iters, self.check_every = int(iters), int(check_every)
        self.simultaneous, self.keep_best, self.paired = bool(simultaneous), bool(keep_best), bool(paired)
        self.device = device
        mode = "sim" if self.simultaneous else "alt"
        self.tag = (f"{self.cfv}_{self.iters}{mode}{'_best' if self.keep_best else ''}_"
                    f"{Path(self.spot.model_path).parent.name}-{file_tag(self.spot.model_path)}")

    def describe(self) -> dict:
        return {"cfv": self.cfv, "iterations": self.iters, "check_every": self.check_every,
                "simultaneous": self.simultaneous, "keep_best": self.keep_best, "paired": self.paired,
                "model": self.spot.model_path,
                "tag": self.tag, "device": str(self.device), "amp": self.spot.amp}

    # The flop: the hook for a flop-end network --------------------------------------------

    def flop_strategy(self) -> list:
        """[c][k] (1, 1, H, A): the stored flop strategy (real_grade.rounded). A live flop solve
        would replace this: a LeafGadget-style solve of the flop from the root (cfv=None,
        reach_o = ones: nothing comes before the flop) with a flop-end network as the leaf."""
        return self.spot.flop_sigma

    def flop_values(self, d: int, c: int, l: int):
        """T values for the turn re-solve from a live flop solve (the flop solve's leaf_average
        at turn root (d, c, l)); None: use the trunk's (`cfv`). There is no flop-end network
        yet, so this is always None."""
        return None

    # The turn -------------------------------------------------------------------------------

    def turn(self, d: int, c: int, l: int, progress=None) -> dict:
        """Both players' re-solves of turn deal d after turn class c's line l (the flop line).

        Returns arrays (npz-ready): "p{p}_k{k}" (H, A) float32 p's re-solved strategy at each
        of p's nodes k; "p{p}_leaf" (closes, H) float64 o = 1 - p's averaged network values at
        every turn close in p's re-solve (the T values of p's river re-solves there, solver
        units as real_trunk's leafcfv.npy); "stats" (2, 7): per p iterations run, the
        iteration kept, gadget gap / mass (chips), the same in pots of the turn root, seconds,
        network seconds, network calls (zeros when p can't reach this turn root).
        progress(p, t, exploitability_chips) is called at every check."""
        s = self.spot.setup(d, c, l, self.cfv)
        fv = self.flop_values(d, c, l)
        if fv is not None:
            s["cfv"], s["cfv_source"] = fv, "flop"
        tp = s["cls"].template
        out = {"stats": np.zeros((2, 7))}
        players = [p for p in (0, 1) if float(s["reach"][p].sum()) > 0]
        runs = self._solve(d, c, l, s, players, progress)
        for p in (0, 1):
            closes = len(tp.closes)
            if p not in players:  # p never gets here
                for k, q in enumerate(tp.players):
                    if q == p:
                        out[f"p{p}_k{k}"] = np.full((self.spot.H, len(tp.actions[k])), 1.0 / len(tp.actions[k]),
                                                    np.float32)
                out[f"p{p}_leaf"] = np.zeros((closes, self.spot.H))
                continue
            g, ev = runs[p]
            avg = g.average_strategy()[0]
            for k, q in enumerate(tp.players):
                if q == p:
                    out[f"p{p}_k{k}"] = np.asarray(avg[k][0, 0], dtype=np.float32)
            out[f"p{p}_leaf"] = np.asarray(g.leaf_average(1 - p)[0][0, 0], dtype=np.float64)
            gap = float(ev["exploitability"])
            out["stats"][p] = (g.t, ev.get("best_iteration", g.t), gap, gap / s["pot"], g.seconds, g.leaf_seconds,
                               g.leaf_calls)
        return out

    def _solve(self, d, c, l, s, players, progress) -> dict:
        """{p: (gadget, evaluation)} for the players that reach the root: one after the other,
        or both at once on two threads sharing the network (paired)."""
        cb = lambda p: None if progress is None else (lambda t, e: progress(p, t, e))
        if not (self.paired and len(players) == 2):
            return {p: self.spot.resolve(s, p, self.iters, None, self.check_every, self.simultaneous,
                                         keep_best=self.keep_best, progress=cb(p)) for p in players}
        pair = PairedLeaf(self.spot.leaf(d, c, [l, l]))
        gad = {p: LeafGadget(s["street"], [s["cls"]], [s["reach"][p]], [s["cfv"][1 - p]], p, pair.side(p),
                             simultaneous=self.simultaneous) for p in (0, 1)}
        res, errs = {}, []

        def run(p):
            try:
                res[p] = gad[p].solve(self.iters, None, self.check_every, keep_best=self.keep_best, progress=cb(p))
            except BaseException as e:  # noqa: BLE001 (re-raised below)
                errs.append(e)
                pair.fail(e)

        threads = [threading.Thread(target=run, args=(p,), daemon=True) for p in (0, 1)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errs:
            raise errs[0]
        return {p: (gad[p], res[p]) for p in (0, 1)}

    @staticmethod
    def river_inputs(turn: dict, close: int) -> np.ndarray:
        """(2, H): cfv[o] = o's T values at the river root after turn close `close`, for the
        river re-solve of p = 1 - o (o's values in p's turn re-solve)."""
        return np.stack([turn[f"p{1 - o}_leaf"][close] for o in (0, 1)])


def timed_turn(solver: LiveSolver, d: int, c: int, l: int) -> tuple:
    t0 = time.perf_counter()
    out = solver.turn(d, c, l)
    return out, time.perf_counter() - t0
