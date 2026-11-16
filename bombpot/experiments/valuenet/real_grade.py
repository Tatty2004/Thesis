"""Exact grade of a real-deck flop+turn solve whose river is safely re-solved (PLAN_valuenet.md,
G.3; the "deepstack" grade of gate_f, for a game too big for the full tree).

The strategy graded: the trunk's average strategy on the flop and the turn (real_trunk's
strategy.npz), and on the river, at every (turn deal, river line), each player p's re-solve with
the gadget (valuenet/gpu_subgame.GpuGadget = core/solvers/resolve.py's safe re-solve) from p's
closing reach under the trunk average, the opponent's T values being its leaf values averaged
over the trunk's iterations (real_trunk's leafcfv.npy). Node k of a river line plays the
re-solve of its acting player.

Why no pass over the river's 3.9M deals is needed afterwards: the opponent o's best-response
value at a river root against p's re-solved strategy is a by-product of p's re-solve (the
gadget's evaluation computes it for every hand of o), and it depends only on p's reach and p's
river strategy. So the flop+turn tree with these values as its leaf (a fixed value per turn-end
state and hand) gives o's best response in the full game exactly: one best-response pass of
RangeCFR on the upper tree per player.

    python -m bombpot.experiments.valuenet.real_grade --trunk results/valuenet/real/<name> --part i/n
        Re-solve the turn deals d with d % n == i (both players, every river line) and write
        <trunk>/grade/part_<i>.npz: o's best-response root values per (deal, river line, hand) in
        solver units, and each re-solve's iterations and normalized gap. Resumable: deals already
        in the file are skipped (the file is rewritten after every deal).
    python -m bombpot.experiments.valuenet.real_grade --trunk results/valuenet/real/<name> --finish
        Needs every part. Writes <trunk>/grade.json: both best-response values and the
        exploitability (chips per hand, the ante is 1).

Units. A re-solve runs on the turn deal's river street with the river's chance weights only
(kappa = river.scale, as datagen) and p's reach normalized to sum 1, so its numbers are those of
a datagen sample; solver units = turn chance weight x p's reach mass x that. `--tol` is in pots
like datagen's: a re-solve stops when its gadget NashConv / 2 over its mass is at most tol x pot.

Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import RangeCFR, backward, forward
from bombpot.core.solvers.subgame import line_template
from bombpot.experiments.valuenet import datagen as dg
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def river_lines(upper) -> list:
    """Every river line as (class, line), in class then line order: the L axis of the arrays here."""
    return [(c, l) for c, rc in enumerate(upper.classes[upper.num_streets]) for l in range(rc.num_lines)]


def per_line(upper, per_class) -> np.ndarray:
    """[c] (D, L_c, H) -> (D, L, H) over river_lines."""
    return np.concatenate(list(per_class), axis=1)


def roots_of(upper, closing) -> np.ndarray:
    """(2, D, L, H): both players' reach at every river root, from the turn's closing reach
    (ExactRiverLeaf.roots)."""
    D, H = closing[0][0].shape[0], closing[0][0].shape[-1]
    out = []
    for q in (0, 1):
        rows = []
        for rc in upper.classes[upper.num_streets]:
            x = np.empty((D, rc.num_lines, H))
            for pc in np.unique(rc.parent_class):
                m = np.flatnonzero(rc.parent_class == pc)
                x[:, m] = closing[pc][q][:, rc.parent_line[m], rc.parent_close[m]]
            rows.append(x)
        out.append(per_line(upper, rows))
    return np.stack(out)


def rounded(avg) -> list:
    """A strategy as stored: float32, then each row renormalized in float64. The graded strategy
    is this one, so the file and the grade agree to the last bit."""
    def one(x):
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        return x / x.sum(-1, keepdims=True)
    return [[[one(x) for x in cls] for cls in row] for row in avg]


def save_strategy(path, avg) -> None:
    np.savez(path, **{f"t{t}_c{c}_k{k}": np.asarray(x, dtype=np.float32)
                      for t, row in enumerate(avg) for c, cls in enumerate(row) for k, x in enumerate(cls)})


def load_strategy(path, upper) -> list:
    with np.load(path) as f:
        return rounded([[[f[f"t{t}_c{c}_k{k}"] for k in range(len(lc.template.actions))]
                         for c, lc in enumerate(upper.classes[t])] for t in range(upper.num_streets)])


def resolve_deal(upper, river, d: int, roots, cfv, tol: float, iters: int, variant: str = "dcfr", device="cuda",
                 dtype=None, keep_strategy: bool = False, check_every: int = 10):
    """Every river line of turn deal d re-solved for both players.

    roots, cfv   (2, L, H): both players' reach at d's river roots, and their averaged leaf values
                 there (solver units)
    Returns br (2, L, H): br[o, l] = o's best-response root value against p = 1 - o's re-solved
    strategy at line l, solver units; stats (2, L, 3): per re-solving player and line the
    iterations, the final gap / mass (chips) and the seconds; and with keep_strategy the
    strategies [l][p][k] (n, H, A) (small decks, for checks)."""
    import torch

    from bombpot.valuenet.gpu_subgame import GpuGadget, StreetSums

    dtype = torch.float32 if dtype is None else dtype
    turn = upper.streets[upper.num_streets - 1]
    st = river.street(*river.all_children([d]))
    kt = float(turn.kappa[d])
    st.kappa = np.full(st.num_deals, river.scale)
    sums = StreetSums(st, device, dtype)
    valid = np.asarray(turn.valid[d], bool)
    lines = river_lines(upper)
    H = roots.shape[-1]
    br, stats, strategies = np.zeros((2, len(lines), H)), np.zeros((2, len(lines), 3)), []
    for i, (c, l) in enumerate(lines):
        rc = upper.classes[upper.num_streets][c]
        if rc.template.num_nodes == 0:
            raise NotImplementedError("river classes without betting (all-in) need exact showdown values")
        tp, contrib = line_template(rc.template, l), float(rc.contrib[l])
        strategies.append([None, None])
        for p in (0, 1):
            o = 1 - p
            rp = np.where(valid, roots[p, i], 0.0)
            m = float(rp.sum())
            if m <= 0:  # p never gets here: o's values are 0 whatever p would play
                continue
            g = GpuGadget(sums, tp, contrib, rp / m, np.where(valid, cfv[o, i], 0.0) / (kt * m), p, variant,
                          dtype=dtype, device=device)
            ev = g.solve(iters, 2.0 * tol * 2.0 * contrib, check_every)
            br[o, i] = np.where(valid, ev["br_values"][o], 0.0) * (kt * m)
            stats[p, i] = (ev["iterations"], ev["exploitability"], g.seconds)
            if keep_strategy:
                strategies[-1][p] = [s.double().cpu().numpy() for s in g.average_strategy()]
    return (br, stats, strategies) if keep_strategy else (br, stats)


class StoredLeaf:
    """Turn-end values read from an array: br (2, D, L, H) over river_lines, scattered to the
    turn's closing lines (ExactRiverLeaf.values)."""

    def __init__(self, upper, br):
        self.upper, self.br = upper, br

    def values(self, p: int, t: int, closing) -> list:
        agg = [np.zeros(c[p].shape) for c in closing]
        a = 0
        for rc in self.upper.classes[self.upper.num_streets]:
            v = self.br[p][:, a:a + rc.num_lines]
            a += rc.num_lines
            for pc in np.unique(rc.parent_class):
                m = np.flatnonzero(rc.parent_class == pc)
                agg[pc][:, rc.parent_line[m], rc.parent_close[m]] = v[:, m]
        return agg


def finish(upper, avg, br) -> dict:
    """Both players' best-response values in the full game against `avg` on the flop and turn
    and the re-solved river behind `br`, and the exploitability (NashConv / 2)."""
    leaf = StoredLeaf(upper, br)
    reach = forward(upper, avg)
    v = [float(backward(upper, avg, reach, p, "br", leaf=leaf)[0][0][0][0].sum()) for p in (0, 1)]
    return {"exploitability": (v[0] + v[1]) / 2, "br_value_p0": v[0], "br_value_p1": v[1]}


def build(trunk: Path):
    args = json.loads((trunk / "trunk.json").read_text())["args"]
    nr, ns = (int(x) for x in args["deck"].split(","))
    ranks, suits = "23456789TJQKA"[:nr], "cdhs"
    flops = tuple(tuple(sorted(ranks.index(f[i]) * ns + suits.index(f[i + 1]) for i in (0, 2, 4)))
                  for f in args["flops"].split(","))
    game = dg.make_game(flops, nr, ns, "pot")
    upper = holdem_public_tree(game, upto=2)
    return game, upper, HoldemRiver(game, upper)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True, help="a real_trunk output directory (strategy.npz, leafcfv.npy, roots.npy)")
    ap.add_argument("--part", help="i/n: re-solve the turn deals d with d %% n == i")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-2, help="re-solve tolerance, pots")
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-minutes", type=float, help="stop starting new deals after this long (resume later)")
    a = ap.parse_args(argv)
    trunk = Path(a.trunk)
    out = trunk / "grade"
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    game, upper, river = build(trunk)
    D = upper.streets[1].num_deals
    L = len(river_lines(upper))
    print(f"game built in {time.perf_counter() - t0:.0f}s: {D} turn deals, {L} river lines", flush=True)
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        deals = [d for d in range(D) if d % n == i]
        path = out / f"part_{i:04d}.npz"
        done = {}
        if path.exists():
            with np.load(path) as f:
                done = {int(d): (f["br"][j], f["stats"][j]) for j, d in enumerate(f["deals"])}
        roots = np.load(trunk / "roots.npy", mmap_mode="r")
        cfv = np.load(trunk / "leafcfv.npy", mmap_mode="r")
        for d in deals:
            if d in done:
                continue
            if a.max_minutes and time.perf_counter() - t0 > 60 * a.max_minutes:
                print(f"stopping before deal {d}: --max-minutes reached", flush=True)
                break
            t1 = time.perf_counter()
            done[d] = resolve_deal(upper, river, d, np.array(roots[:, d]), np.array(cfv[:, d]), a.tol, a.sub_iters,
                                   device=a.device)
            s = done[d][1]
            print(f"deal {d}: {time.perf_counter() - t1:.0f}s, {int(s[..., 0].sum())} iterations, "
                  f"largest gap / mass {s[..., 1].max():.4g} chips", flush=True)
            keys = sorted(done)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, deals=np.array(keys), br=np.stack([done[k][0] for k in keys]),
                     stats=np.stack([done[k][1] for k in keys]), tol=a.tol, sub_iters=a.sub_iters)
            tmp.replace(path)
        print("DONE" if all(d in done for d in deals) else "INCOMPLETE", flush=True)
        return
    if a.finish:
        br = np.zeros((2, D, L, len(upper.hands)))
        seen = np.zeros(D, bool)
        its, gaps, secs = 0, 0.0, 0.0
        for path in sorted(out.glob("part_*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            with np.load(path) as f:
                br[:, f["deals"]] = np.moveaxis(f["br"], 0, 1)
                seen[f["deals"]] = True
                its += int(f["stats"][..., 0].sum())
                gaps = max(gaps, float(f["stats"][..., 1].max()))
                secs += float(f["stats"][..., 2].sum())
        if not seen.all():
            raise SystemExit(f"{int((~seen).sum())} turn deals are missing, e.g. {np.flatnonzero(~seen)[:10]}")
        avg = load_strategy(trunk / "strategy.npz", upper)
        res = finish(upper, avg, br)
        res.update(resolve_iterations=its, resolve_largest_gap_over_mass=gaps, resolve_gpu_seconds=secs,
                   git=dg.git_hash(), trunk=str(trunk))
        (trunk / "grade.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
