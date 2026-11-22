"""Flop-end samples for a flop-end value network (cost pilot, 2026-11-22): per-hand counterfactual
values at the end of the flop, before the turn cards, as DeepStack's flop network learns them.

A sample: one flop pair, one flop-end line of the real game (the 13 lines that close the flop:
its pot and chips, and the turn betting that follows), random ranges for both players over the
hands (datagen.random_range: recursive random splitting, ordered by flop-end strength = the mean
turn-end equity over the turn deals). Below that state: the turn deal (chance), turn betting, and
at every turn close the turn-end network (NetLeaf) values the river. That subgame is solved with
DCFR (gate_f/real_trunk's machinery: SubgameCFR = RangeCFR on one street from root ranges, with
the network as its leaf) and the root counterfactual values are the label, in pots, in datagen's
units: cfv_p[i] = sum over compatible j of range_o[j] E[payoff of p | i, j] / pot, the chance
weight per valid turn deal 1 / (rest (rest - 1)) (rest = cards left after the flops and both
hands), so a hand pair's valid turn deals weigh 1 in all.

**No decision sits above the turn deal**, so the turn subgame is one independent betting game per
turn deal (regrets live per deal, line and hand), and the root value is the sum of the per-deal
values. Solving a subset S of the deals gives exactly those deals' values of the full solve (up
to the network's rounding in other batch shapes). So the cheap variants are estimators of that
sum from a sample of deals (flop_pilot.py measures them against the full solve):
  plain   (D / K) sum over S of v_d (unbiased, D = all turn deals)
  cv      a control variate: sum over all D of f_d + (D / K) sum over S of (v_d - f_d), with
          f_d = the network's own value at deal d's turn end when both players check the turn
          with their root ranges (`--proxy`: one network row per deal, ~1/26 of an iteration)

    python -m bombpot.experiments.valuenet.flop_datagen --model .../best.pt --flops Ah7c2d,KsKd9h \
        --equity results/valuenet/real/equity_Ah7c2d_KsKd9h_13x4.npy --seed 0 --deals all \
        --iters 300 --checks 25,50,100,200,300 --amp --proxy --out results/valuenet/flopnet_pilot/ref

writes <out>/sample_<flops>_s<seed>_<deals>.npz: the state (flops, line, pot, stack, ranges), the
deals solved, and at every check: label (2, H) in pots with the plain estimator (exact for
--deals all), per-deal values (2, K, H) in pots x valid deals per hand pair (so a deal's array is
on the label's scale) when --per-deal, the subgame's exploitability against the network (pots),
and the seconds so far (all, and in the network). --simultaneous updates both players from one
network pass per iteration (half the network rows; the alternating default is gate_f's).

Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json
import time
from math import comb
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import VARIANTS, _update, backward, stack_closing, street_reach
from bombpot.core.solvers.subgame import SubgameCFR, line_template
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet.gate_f import turn_pots
from bombpot.experiments.valuenet.real_trunk import river_street
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

RANKS, SUITS = "23456789TJQKA", "cdhs"


def parse_flops(text: str, num_ranks: int, num_suits: int) -> tuple:
    ranks = RANKS[:num_ranks]
    return tuple(tuple(sorted(ranks.index(f[i]) * num_suits + SUITS.index(f[i + 1]) for i in (0, 2, 4)))
                 for f in text.split(","))


class FlopEnd:
    """The pieces built once per flop pair: game, flop+turn public tree, the turn street with
    chance weights for normalized root ranges, the flop-end lines (turn class 0), the pots at
    every turn close, boards per turn deal, and turn-end equities (all deals from `equity`, a
    (D, H, 2) array as real_trunk caches it, or computed per deal on demand)."""

    def __init__(self, flops, num_ranks: int = 13, num_suits: int = 4, equity=None):
        self.flops = flops
        self.game = dg.make_game(flops, num_ranks, num_suits, "pot")
        self.upper = holdem_public_tree(self.game, upto=2)
        self.river = HoldemRiver(self.game, self.upper)
        if len(self.upper.classes[1]) != 1:
            raise NotImplementedError("one turn line class expected (the real game's 13 flop-end lines)")
        self.lc = self.upper.classes[1][0]
        turn = self.upper.streets[1]
        n = num_ranks * num_suits - 6  # cards after the flops
        hole = comb(n, 2) * comb(n - 2, 2)  # the tree's turn kappa = hole-card weight x turn weight
        self.turn = turn.subset(np.arange(turn.num_deals), scale=hole)
        rest = n - 4
        assert np.allclose(self.turn.kappa, 1.0 / (rest * (rest - 1))), "turn chance weights"
        self.per_pair = rest * (rest - 1)  # valid turn deals of one hand pair
        self.D, self.H = turn.valid.shape
        self.pots = turn_pots(self.upper)[0]  # (lines, closes)
        self.hands = np.array(self.upper.hands, dtype=np.int64)
        self.boards = np.concatenate([np.broadcast_to(np.array(self.game.flops), (self.D, 2, 3)),
                                      np.asarray(self.river.turn_cards)[:, :, None]], axis=-1)
        self._eq = {} if equity is None else None
        self.equity_all = None if equity is None else np.asarray(equity, dtype=np.float32)
        if self.equity_all is not None and self.equity_all.shape != (self.D, self.H, 2):
            raise ValueError(f"equity {self.equity_all.shape} for {self.D} deals x {self.H} hands")
        self.equity_seconds = 0.0

    def equity(self, deals) -> np.ndarray:
        """(K, H, 2) turn-end equities of `deals`."""
        if self.equity_all is not None:
            return self.equity_all[deals]
        t0 = time.perf_counter()
        for d in deals:
            if int(d) not in self._eq:
                self._eq[int(d)] = dg.turn_equity(river_street(self.river, int(d))).astype(np.float32)
        self.equity_seconds += time.perf_counter() - t0
        return np.stack([self._eq[int(d)] for d in deals])

    def strength(self, deals=None) -> np.ndarray:
        """(H,) flop-end strength for ordering random ranges: the mean over `deals` of the mean
        turn-end equity over both boards, where the hand can be held. Default: a fixed random
        64 of the turn deals (seeded), so a state doesn't depend on which deals get solved."""
        if deals is None:
            deals = np.sort(np.random.default_rng(12345).permutation(self.D)[:64])
        deals = np.asarray(deals)
        eq = self.equity(deals).mean(-1)
        valid = self.turn.valid[deals]
        return np.where(valid, eq, 0).sum(0) / np.maximum(valid.sum(0), 1)

    def leaf(self, model, deals, line: int, closes=None, device="cuda", devices=None, amp: bool = False):
        """NetLeaf over `deals` at flop-end line `line`'s turn closes (all, or the indices in
        `closes`), with the turn street's chance weights."""
        from bombpot.valuenet.leaf import NetLeaf

        pots = self.pots[line][None] if closes is None else self.pots[line][None, closes]
        return NetLeaf(model, self.equity(deals), [pots], [dg.STACK - pots / 2], self.turn.kappa[deals], device,
                       boards_by_deal=self.boards[deals], possible_by_deal=self.turn.valid[deals], hands=self.hands,
                       devices=devices, amp=amp)


def sample_state(fe: FlopEnd, rng, strength=None, line: int | None = None) -> dict:
    """A flop-end line (uniform over the 13, unless given) and random ranges for both players."""
    strength = fe.strength() if strength is None else strength
    line = int(rng.integers(fe.lc.num_lines)) if line is None or line < 0 else int(line)
    valid = np.ones(fe.H, bool)  # every hand can be held at the flop's end (the flops fixed)
    r0, r1 = dg.random_range(rng, valid, strength), dg.random_range(rng, valid, strength)
    contrib = float(fe.lc.contrib[line])
    return {"line": line, "pot": 2 * contrib, "stack": dg.STACK - contrib, "range0": r0, "range1": r1}


class _Fixed:
    """A leaf that hands out precomputed values (both players' from one network pass)."""

    def __init__(self, vals):
        self.vals = vals

    def values(self, p: int, t: int, closing):
        return self.vals[p]


class Timed:
    """A leaf wrapper that counts seconds and rows (deal x line x close) per call."""

    def __init__(self, leaf):
        self.leaf, self.seconds, self.rows, self.calls = leaf, 0.0, 0, 0

    def _count(self, closing):
        self.rows += sum(int(np.prod(c[0].shape[:3])) for c in closing)
        self.calls += 1

    def values(self, p: int, t: int, closing):
        t0 = time.perf_counter()
        out = self.leaf.values(p, t, closing)
        self.seconds += time.perf_counter() - t0
        self._count(closing)
        return out

    def values_both(self, t: int, closing):
        t0 = time.perf_counter()
        out = self.leaf.values_both(t, closing)
        self.seconds += time.perf_counter() - t0
        self._count(closing)
        return out


class TurnCFR(SubgameCFR):
    """The turn below one flop-end state, over `deals` (indices of fe's turn deals), every turn
    close valued by `leaf` (a NetLeaf over the same deals, ideally Timed). Alternating DCFR by
    default (as the trunk); simultaneous=True updates both players from one leaf pass."""

    def __init__(self, fe: FlopEnd, state: dict, deals, leaf, variant: str = "dcfr", simultaneous: bool = False):
        self.fe, self.deals = fe, np.asarray(deals, dtype=np.int64)
        st = fe.turn.subset(self.deals)
        tp = line_template(fe.lc.template, state["line"])
        contrib = float(fe.lc.contrib[state["line"]])
        super().__init__(st, tp, [contrib], (state["range0"], state["range1"]), variant=variant, leaf=leaf,
                         stack=[dg.STACK - contrib])
        self.simultaneous = simultaneous
        self.pot = 2 * contrib

    def _fixed(self, hsig, reach) -> _Fixed:
        _, close = street_reach(self.pt.classes[0], hsig[0], [r[0] for r in reach[0]])
        return _Fixed(self.leaf.values_both(0, stack_closing(close)))

    def iteration(self) -> None:
        if not self.simultaneous:
            return super().iteration()
        start = time.perf_counter()
        self.t += 1
        t_ = self.t
        weight = {"cfr": 1.0, "cfr+": float(t_), "dcfr": float(t_) ** self.gamma}[self.variant]
        pos = t_ ** self.alpha / (t_ ** self.alpha + 1.0)
        neg = t_ ** self.beta / (t_ ** self.beta + 1.0)
        hsig = self.hand_sigma()
        reach = self.reach(hsig)
        fixed = self._fixed(hsig, reach)
        regrets = [backward(self.pt, hsig, reach, p, "cfr", leaf=fixed)[1] for p in (0, 1)]
        for p in (0, 1):  # both from the same profile, then both updated
            for t, c, k in self._nodes(p):
                reg = self.regret[t][c][k]
                A = reg.shape[-1]
                self.sigma[t][c][k] = np.ascontiguousarray(self.sigma[t][c][k])
                own = np.ascontiguousarray(self.own_reach(t, c, k, reach)).reshape(-1)
                _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(regrets[p][t][c][k]).reshape(-1, A),
                        self.sigma[t][c][k].reshape(-1, A), self.cum[t][c][k].reshape(-1, A), own, weight, pos, neg,
                        VARIANTS[self.variant])
        self.solve_seconds += time.perf_counter() - start

    def per_deal(self) -> dict:
        """The average strategy's values per deal, from one leaf pass: cfv[p] and br[p] (K, H)
        in solver units (chips), and the subgame's exploitability against the leaf (pots per
        unit of root mass)."""
        hsig = self.hand_sigma(self.average_strategy())
        reach = self.reach(hsig)
        fixed = self._fixed(hsig, reach)
        cfv = [backward(self.pt, hsig, reach, p, "cfr", leaf=fixed)[0][0][0][0][:, 0] for p in (0, 1)]
        br = [backward(self.pt, hsig, reach, p, "br", leaf=fixed)[0][0][0][0][:, 0] for p in (0, 1)]
        r0, r1 = self.root_ranges[0][0, 0], self.root_ranges[1][0, 0]
        mass = float(self.mass()[0, 0])
        raw = (br[0].sum(0) @ r0 + br[1].sum(0) @ r1) / 2
        return {"cfv": cfv, "br": br, "exploitability": raw / mass / self.pot if mass > 0 else 0.0, "mass": mass}


def proxy_values(fe: FlopEnd, model, state: dict, deals, device="cuda", devices=None, amp=False) -> tuple:
    """f_d (2, K, H) in pots x per_pair: the network's values at each deal's turn end when both
    players check the turn with their root ranges (the control variate), and the seconds."""
    tp = line_template(fe.lc.template, state["line"])
    chips = [float(np.asarray(x).reshape(-1)[0]) for _, x in tp.closes]
    cc = int(np.argmin(chips))  # check-check: the only close that adds no chips
    assert chips[cc] == 0 and sorted(chips)[1] > 0, chips
    leaf = fe.leaf(model, deals, state["line"], closes=[cc], device=device, devices=devices, amp=amp)
    K = len(deals)
    t0 = time.perf_counter()
    root = [np.broadcast_to(state[f"range{p}"], (K, 1, 1, fe.H)).copy() for p in (0, 1)]
    v = leaf.values_both(1, [tuple(root)])
    out = np.stack([x[0][:, 0, 0] for x in v]) / state["pot"] * fe.per_pair
    return out, time.perf_counter() - t0


def solve_state(fe: FlopEnd, model, state: dict, deals, iters: int, checks, *, device="cuda", devices=None,
                amp: bool = False, simultaneous: bool = False, per_deal: bool = True, log=print) -> dict:
    """Solve one flop-end state's turn over `deals`; the record of every check (module docstring)."""
    deals = np.asarray(deals, dtype=np.int64)
    t0 = time.perf_counter()
    leaf = Timed(fe.leaf(model, deals, state["line"], device=device, devices=devices, amp=amp))
    setup = time.perf_counter() - t0
    s = TurnCFR(fe, state, deals, leaf, simultaneous=simultaneous)
    scale = fe.D / len(deals)  # the plain estimator's importance weight
    out = {"setup_seconds": setup, "checks": [], "exploitability": [], "seconds": [], "leaf_seconds": [],
           "leaf_rows": [], "label_cfv": [], "label_br": [], "deal_cfv": [], "deal_br": []}
    t0 = time.perf_counter()
    for it in range(1, iters + 1):
        s.iteration()
        if it in checks:
            ev = s.per_deal()
            out["checks"].append(it)
            out["exploitability"].append(ev["exploitability"])
            out["seconds"].append(time.perf_counter() - t0)
            out["leaf_seconds"].append(leaf.seconds)
            out["leaf_rows"].append(leaf.rows)
            for key in ("cfv", "br"):
                out[f"label_{key}"].append(np.stack([v.sum(0) * scale / state["pot"] for v in ev[key]]).astype(
                    np.float32))
                if per_deal:
                    out[f"deal_{key}"].append(np.stack([v / state["pot"] * fe.per_pair for v in ev[key]]).astype(
                        np.float32))
            log(f"  it {it:4d}  {out['seconds'][-1]:8.1f}s (network {leaf.seconds:8.1f}s, {leaf.rows} rows)  "
                f"exploitability vs net {ev['exploitability']:.2e} pots")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--flops", required=True, help="e.g. Ah7c2d,KsKd9h")
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--equity", help="(D, H, 2) turn-end equities of every turn deal (real_trunk's cache); "
                                     "default: computed for the deals solved")
    ap.add_argument("--seed", type=int, default=0, help="the state (line and ranges)")
    ap.add_argument("--samples", type=int, default=1, help="states from this seed, one after the other")
    ap.add_argument("--line", type=int, default=-1, help="flop-end line (default random)")
    ap.add_argument("--deals", default="all", help="'all', or K: a uniform random subset of K turn deals")
    ap.add_argument("--deal-seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--checks", default="25,50,100,200,300")
    ap.add_argument("--simultaneous", action="store_true", help="both players from one network pass")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--proxy", action="store_true", help="also the check-down control variate at every deal")
    ap.add_argument("--no-per-deal", action="store_true", help="labels only (no per-deal arrays)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpus", type=int, default=1, help="network copies (cuda:0 .. cuda:n-1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    from bombpot.valuenet.model import load_model

    nr, ns = (int(x) for x in a.deck.split(","))
    t0 = time.perf_counter()
    fe = FlopEnd(parse_flops(a.flops, nr, ns), nr, ns, None if a.equity is None else np.load(a.equity))
    print(f"flop pair {a.flops}: {fe.D} turn deals, {fe.H} hands, built in {time.perf_counter() - t0:.1f}s", flush=True)
    model = load_model(a.model, a.device)
    devices = [f"cuda:{i}" for i in range(a.gpus)] if a.device == "cuda" and a.gpus > 1 else None
    if a.device == "cuda":
        print("gpu", torch.cuda.get_device_name(0), flush=True)
    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    rng = np.random.default_rng(a.seed)
    if a.deals == "all":
        deals = np.arange(fe.D)
    else:
        deals = np.sort(np.random.default_rng([a.deal_seed, a.seed]).choice(fe.D, int(a.deals), replace=False))
    strength = fe.strength()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in range(a.samples):
        state = sample_state(fe, rng, strength, a.line)
        print(f"sample {n}: line {state['line']} pot {state['pot']:g}, {len(deals)} deals, {a.iters} iterations"
              f"{' simultaneous' if a.simultaneous else ''}", flush=True)
        rec = solve_state(fe, model, state, deals, a.iters, checks, device=a.device, devices=devices, amp=a.amp,
                          simultaneous=a.simultaneous, per_deal=not a.no_per_deal,
                          log=lambda m: print(m, flush=True))
        extra = {}
        if a.proxy:
            all_deals = np.arange(fe.D)
            f, sec = proxy_values(fe, model, state, all_deals, a.device, devices, a.amp)
            extra = {"proxy": f.astype(np.float32), "proxy_seconds": sec}
            print(f"  proxy over {fe.D} deals: {sec:.1f}s", flush=True)
        name = f"sample_{a.flops.replace(',', '_')}_s{a.seed}_{n}_{a.deals}{'_sim' if a.simultaneous else ''}"
        info = {"args": vars(a), "git": dg.git_hash(), "gpu": torch.cuda.get_device_name(0) if a.device == "cuda"
                else "cpu", "flops": a.flops, "num_deals": fe.D, "per_pair": fe.per_pair,
                "equity_seconds": fe.equity_seconds, "setup_seconds": rec["setup_seconds"]}
        arrays = {k: np.asarray(v) for k, v in rec.items() if k != "setup_seconds" and len(v)}
        np.savez_compressed(out / f"{name}.npz", info=np.array(json.dumps(info, default=str)), deals=deals,
                            flops=np.array(fe.game.flops), line=state["line"], pot=state["pot"], stack=state["stack"],
                            range0=state["range0"].astype(np.float32), range1=state["range1"].astype(np.float32),
                            **arrays, **extra)
        print(f"wrote {out / name}.npz", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
