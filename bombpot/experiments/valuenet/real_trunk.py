"""The real game's flop+turn solve with a value network as the turn-end leaf (PLAN_valuenet.md,
G.2), and the network's error where that solve uses it.

    python -m bombpot.experiments.valuenet.real_trunk --model results/valuenet/net/<run>/best.pt \
        --flops Ah7c2d,KsKd9h --iters 100 --checks 25,50,100 --samples 2000 --out results/valuenet/real/<name>

1. Solve: gate_f.LeafCFR on the real no-limit game (datagen.make_game "pot") for one flop pair,
   every turn-end state valued by the network (NetLeaf). At each check: the solve's own
   exploitability against the network's values (what the solver believes, not a grade).
   Written to <out>/trunk.json; the average strategy to <out>/strategy.npz (float32, one array
   per street, class and node: t{t}_c{c}_k{k}) unless --no-strategy, with real_grade's inputs:
   roots.npy (both reaches at every river root) and leafcfv.npy (both players' leaf values averaged
   over the iterations), each (2, turn deals, river lines, hands).
2. On-policy leaf check: `--samples` turn-end states (turn deal, turn line, close) drawn
   uniformly from the final average strategy's, each with both players' ranges there (masked to
   the possible hands, normalized to sum 1; uniform where a player has no mass, as NetLeaf feeds
   the network). The river subgame of each is solved to `--tol` pots (the torch solver on
   --device, or the compiled one with --solver fused) and stored in datagen's shard layout at
   <out>/onpolicy.h5, so eval_common measures any network on it. <out>/onpolicy_meta.npz has, per
   sample, deal, line, close and each player's reach mass at that state (for weighting).

   The drawn states (with both reaches) are also written to <out>/onpolicy_todo.npz. With --defer
   the run stops there, and `--part i/n` (no trunk solve, same --out) solves slice i of n into
   <out>/onpolicy_<i>.h5, so the river solves can run as an array of GPU tasks.

The per-deal turn-end equities (a river street per turn deal) are cached in <out>/../equity_<flops>.npy.
Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import forward
from bombpot.core.solvers.subgame import line_template
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet import real_grade as rg
from bombpot.experiments.valuenet.gate_f import LeafCFR, turn_pots
from bombpot.experiments.valuenet.onpolicy import normalized
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def river_street(river, d: int):
    st = river.street(*river.all_children([d]))  # as FlopPair.river_street: river-only chance weights
    st.kappa = np.full(st.num_deals, river.scale)
    return st


def turn_equities(river, D: int, cache: Path) -> np.ndarray:
    """(D, H, 2) datagen.turn_equity of every turn deal, cached."""
    if cache.exists():
        return np.load(cache)
    t0 = time.perf_counter()
    eq = []
    for d in range(D):
        eq.append(dg.turn_equity(river_street(river, d)).astype(np.float32))
        if (d + 1) % 200 == 0:
            print(f"equity: {d + 1}/{D} turn deals, {time.perf_counter() - t0:.0f}s", flush=True)
    eq = np.stack(eq)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, eq)
    return eq


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="required unless --part")
    ap.add_argument("--flops", default="Ah7c2d,KsKd9h")
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--iters", type=int, default=300, help="300 since 2026-11-18 (README: real-deck solves)")
    ap.add_argument("--checks", default="50,100,200,300")
    ap.add_argument("--trunk", default="dcfr", choices=["cfr", "cfr+", "dcfr"])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-2, help="river subgame exploitability target, pots")
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--solver", default="gpu", choices=["gpu", "fused"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-strategy", action="store_true")
    ap.add_argument("--amp", action="store_true", help="mixed precision for the network leaf")
    ap.add_argument("--defer", action="store_true",
                    help="write the drawn states to <out>/onpolicy_todo.npz and stop; solve them with --part")
    ap.add_argument("--part", help="i/n: no trunk solve; solve slice i of n of <out>/onpolicy_todo.npz "
                                   "into <out>/onpolicy_<i>.h5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    from bombpot.valuenet.leaf import NetLeaf
    from bombpot.valuenet.model import load_model

    nr, ns = (int(x) for x in a.deck.split(","))
    ranks, suits = "23456789TJQKA"[:nr], "cdhs"
    flops = tuple(tuple(sorted(ranks.index(f[i]) * ns + suits.index(f[i + 1]) for i in (0, 2, 4)))
                  for f in a.flops.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    t0 = time.perf_counter()
    game = dg.make_game(flops, nr, ns, "pot")
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    turn = upper.streets[1]
    D, H = turn.valid.shape
    print(f"game built in {time.perf_counter() - t0:.0f}s: {D} turn deals, {H} hands, river classes "
          f"{[rc.num_lines for rc in upper.classes[2]]}", flush=True)
    if any(rc.template.num_nodes == 0 for rc in upper.classes[2]):
        raise NotImplementedError("river classes without betting (all-in) need exact showdown values")
    equity = turn_equities(river, D, out.parent / f"equity_{a.flops.replace(',', '_')}_{nr}x{ns}.npy")
    todo_path = out / "onpolicy_todo.npz"
    if a.part:
        return solve_states(a, game, upper, river, turn, equity, np.load(todo_path), out)
    model = load_model(a.model, a.device)
    boards = np.concatenate([np.broadcast_to(np.array(game.flops), (D, 2, 3)),
                             np.asarray(river.turn_cards)[:, :, None]], axis=-1)
    pots = turn_pots(upper)
    leaf = NetLeaf(model, equity, pots, [dg.STACK - p / 2 for p in pots], turn.kappa, a.device,
                   boards_by_deal=boards, possible_by_deal=turn.valid, hands=np.array(upper.hands, dtype=np.int64),
                   devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())] if a.device == "cuda" else None,
                   amp=a.amp)
    print(f"leaf on {len(leaf.replicas)} device(s), amp {a.amp}, batch {leaf.batch_size}", flush=True)
    solver = LeafCFR(upper, leaf, a.trunk)
    result = {"args": vars(a), "features": model.features.to_dict(), "turn_deals": D, "hands": H, "checks": {}}
    t0 = time.perf_counter()
    while solver.t < a.iters:
        solver.iteration()
        print(f"iteration {solver.t}: {time.perf_counter() - t0:.0f}s", flush=True)
        if solver.t in checks:
            ft = solver.exploitability()
            result["checks"][solver.t] = {"wall_time": time.perf_counter() - t0, "flop_turn": ft}
            print(f"check {solver.t}: flop+turn exploitability against the network {ft['exploitability']:.5f} "
                  f"(values {ft.get('br_value_p0')}, {ft.get('br_value_p1')})", flush=True)
            (out / "trunk.json").write_text(json.dumps(result, indent=1, default=float))
    avg = rg.rounded(solver.average_strategy())  # the strategy as stored (float32): what real_grade grades
    if not a.no_strategy:
        rg.save_strategy(out / "strategy.npz", avg)
    _, closing = forward(upper, solver.hand_sigma(avg), closing=True)
    closing = [tuple(np.array(x) for x in cl) for cl in closing]  # [turn class] (r0, r1) each (D, L, C, H)
    if not a.no_strategy:  # real_grade's inputs: both reaches and both averaged leaf values at every river root
        np.save(out / "roots.npy", rg.roots_of(upper, closing))
        np.save(out / "leafcfv.npy", np.stack([rg.per_line(upper, [solver.cfv_average(c, q)
                                                                   for c in range(len(upper.classes[2]))])
                                               for q in (0, 1)]))
    del solver, avg

    # the states to solve: (turn deal, turn class, line, close) and both players' reach there
    keys = sorted(continuations(upper))
    rng = np.random.default_rng(a.seed)
    picks = sorted((int(rng.integers(D)), keys[int(rng.integers(len(keys)))]) for _ in range(a.samples))
    todo = {"deal": np.array([d for d, _ in picks], dtype=np.int64),
            "key": np.array([k for _, k in picks], dtype=np.int64).reshape(len(picks), 3),
            "raw0": np.array([closing[pc][0][d, l, c] for d, (pc, l, c) in picks]).reshape(len(picks), H),
            "raw1": np.array([closing[pc][1][d, l, c] for d, (pc, l, c) in picks]).reshape(len(picks), H)}
    del closing
    np.savez(todo_path, **todo)
    if a.defer:
        print(f"DONE (deferred {len(picks)} states to {todo_path})", flush=True)
        return
    solve_states(a, game, upper, river, turn, equity, todo, out)


def continuations(upper) -> dict:
    """(turn class, line, close) -> (river class, river line) that continues it."""
    cont = {}
    for ci, rc in enumerate(upper.classes[2]):
        for l in range(rc.num_lines):
            cont[(int(rc.parent_class[l]), int(rc.parent_line[l]), int(rc.parent_close[l]))] = (ci, l)
    return cont


def solve_states(a, game, upper, river, turn, equity, todo, out: Path) -> None:
    """Exact river solves of the drawn states (all, or slice i of n with --part i/n: contiguous
    in the deal-sorted order), in datagen's shard layout."""
    import torch

    nr, ns = (int(x) for x in a.deck.split(","))
    H = turn.valid.shape[1]
    cont = continuations(upper)
    S = len(todo["deal"])
    name, lo, hi = "onpolicy", 0, S
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        name, lo, hi = f"onpolicy_{i:03d}", S * i // n, S * (i + 1) // n
    attrs = {"num_ranks": nr, "num_suits": ns, "board_cards": "3,1,1", "num_boards": 2, "ante": 1.0,
             "total_chips": dg.STACK, "template": "pot", "source": "real_trunk onpolicy (average strategy)",
             "model": str(a.model), "trunk_iters": a.iters, "tol_pots": a.tol, "iters_cap": a.sub_iters, "variant": "dcfr",
             "solver": a.solver, "flops_fixed": a.flops, "git": dg.git_hash(), "units": dg.UNITS}
    shard = dg.Shard(out / f"{name}.h5", H, attrs)
    hset = shard.add_hands(np.array(upper.hands, dtype=np.int64))
    meta = {"deal": [], "turn_class": [], "line": [], "close": [], "mass0": [], "mass1": [], "river_class": []}
    t0 = time.perf_counter()
    st, st_d = None, -1
    for n in range(lo, hi):
        d, (pc, l, c) = int(todo["deal"][n]), (int(x) for x in todo["key"][n])
        if d != st_d:
            st, st_d = river_street(river, d), d
        ci, lr = cont[(pc, l, c)]
        rc = upper.classes[2][ci]
        P = 2.0 * float(rc.contrib[lr])
        valid = turn.valid[d]
        raw = [todo["raw0"][n], todo["raw1"][n]]
        r0, r1 = (normalized(r, valid) for r in raw)
        tp = line_template(rc.template, lr)
        if a.solver == "gpu":
            from bombpot.valuenet.gpu_subgame import solve_subgame_gpu

            sub = solve_subgame_gpu(st, tp, P / 2, r0, r1, "dcfr", a.sub_iters, a.tol * P, dtype=torch.float32,
                                    device=a.device)
        else:
            from bombpot.core.solvers.fused import solve_subgame_fused

            sub = solve_subgame_fused(st, tp, P / 2, r0, r1, "dcfr", a.sub_iters, a.tol * P, keep_strategy=False)
        v = [sub.values[q] / P for q in (0, 1)]
        br = [sub.br_values[q] / P for q in (0, 1)]
        shard.add({"flops": np.array(game.flops), "turn": np.asarray(river.turn_cards[d]), "hand_set": hset, "pot": P,
                   "stack": dg.STACK - P / 2, "line": int(rc.lines[lr]), "pot_spread": 1.0, "range0": r0,
                   "range1": r1, "possible": valid, "equity": equity[d], "cfv0": v[0], "cfv1": v[1], "br0": br[0],
                   "br1": br[1], "zero_sum_residual": float(r0 @ v[0] + r1 @ v[1]),
                   "subgame_exploitability": sub.exploitability / P, "iters": sub.iterations,
                   "seconds": sub.seconds, "mass": sub.mass})
        for key, val in zip(meta, (d, pc, l, c, float(np.where(valid, raw[0], 0).sum()),
                                   float(np.where(valid, raw[1], 0).sum()), ci)):
            meta[key].append(val)
        if (n + 1 - lo) % 50 == 0 or n + 1 == hi:
            print(f"on-policy sample {n + 1 - lo}/{hi - lo}, {time.perf_counter() - t0:.0f}s", flush=True)
            np.savez(out / f"{name}_meta.npz", **{k: np.array(v) for k, v in meta.items()})
    shard.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
