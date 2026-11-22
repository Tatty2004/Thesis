"""Gate T: safe turn re-solving against an exact reference on the small deck (5 ranks x 3
suits, the real no-limit betting of datagen.make_game "pot": one turn class of 13 lines, 169
river lines in 6 classes).

1. Trunk: gate_f.LeafCFR with an exact river leaf (ExactRiverLeaf, best-response values), DCFR
   alternating as real_trunk runs it, keeping both players' averaged leaf values (the river
   gadget's T values) and averaged turn-root values (the turn gadget's, `--turn-cfv`).
2. Baseline grade ("trunk"): the trunk's flop and turn, every river safely re-solved from the
   trunk average's closing reach with the averaged leaf values (gate_c1.deepstack_grade, the
   strategy real_grade grades), exact exploitability on the full three-street tree.
3. Turn re-solve grade: for each player p, every turn root (turn deal, flop line) re-solved for
   p with the gadget (core/solvers/leaf_resolve.LeafGadget, the same exact leaf at the turn's
   end), then every river below safely re-solved for p from p's closing reach under the
   re-solved turn, the opponent's T values being its leaf values averaged over the turn
   re-solve's iterations. In the composite each turn and river node plays the re-solve of its
   acting player; the flop is the trunk's. Exact exploitability on the full tree. T values at
   the turn root (`--sources`):
     avg     the trunk's averaged turn-root values (real_trunk --turn-cfv)
     backup  the trunk's averaged leaf values backed up through its average turn strategy
             (turn_resolve --cfv backup: what a trunk without --turn-cfv still allows)
     onpolicy  the values of the trunk's average turn strategy, the leaf at its closing reach
     br      the opponent's best response to the trunk's turn at each root (leaf-valued)
     unsafe  no gadget: the turn alone from both trunk reaches (for contrast)
   Also, per source: the summed gadget gaps (chips per hand of the whole game), the largest
   gap / mass (chips), and how close the re-solve's root values are to the trunk's
   (reproduction): o's best response to the re-solve minus o's T values, and p's re-solved
   on-policy root value minus p's trunk value, each summed with the player's trunk reach.

    python -m bombpot.experiments.valuenet.gate_t --trunk-iters 100 --resolve-iters 100,300 \
        --out results/valuenet/turn/gate_t.json [--trunk-file trunk.pkl]

--trunk-file: load the trunk (and its grade) from this pickle when it exists, else run it and
save it there, so several runs (e.g. one per source) share one trunk. Each source's T values are
also compared with avg's ("t_values_vs_avg": the opponent-reach-weighted L1 distance and signed
difference over the reach-weighted L1 size of avg's, per player).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.leaf_resolve import LeafGadget
from bombpot.core.solvers.range_cfr import RangeCFR, forward
from bombpot.core.solvers.resolve import resolve_batch
from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet.gate_f import LeafCFR
from bombpot.experiments.valuenet.turn_resolve import backup_values, blueprint_values
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

FLOPS = ((0, 3, 7), (9, 11, 6))  # 2c3c4d / 5c6d4c on 5 ranks x 3 suits


def to_river_classes(upper, per_turn_class) -> list:
    """[turn class] (D, L_c, closes_c, H) -> [river class] (D, L_rc, H), each river line taking
    the close it continues (LeafCFR.cfv_average's layout)."""
    out = []
    for rc in upper.classes[upper.num_streets]:
        x = np.empty((per_turn_class[0].shape[0], rc.num_lines, per_turn_class[0].shape[-1]))
        for pc in np.unique(rc.parent_class):
            m = np.flatnonzero(rc.parent_class == pc)
            x[:, m] = per_turn_class[pc][:, rc.parent_line[m], rc.parent_close[m]]
        out.append(x)
    return out


def composite(full, upper, geom, flop, turn_by_player, river_cfv_by_player, tol, iters, variant) -> dict:
    """Exact exploitability of: the flop `flop`, at every turn node the strategy of its acting
    player q from turn_by_player[q] ([c][k] arrays), and every river safely re-solved for each
    player p from p's closing reach under (flop, turn_by_player[p]) with the opponent's T values
    river_cfv_by_player[p] ([river class] (D, L, H)); river node k plays its acting player's."""
    roots = {}
    for p in (0, 1):
        _, closing = forward(upper, [flop, turn_by_player[p]], closing=True)
        roots[p] = geom.roots(closing)
    D1 = upper.streets[-1].num_deals
    river, gaps, its = [], [0.0, 0.0], 0
    for c, lc in enumerate(geom.classes):
        if lc.template.num_nodes == 0:
            river.append([])
            continue
        res = {}
        for p in (0, 1):
            o = 1 - p
            res[p] = resolve_batch(geom.street, lc.template, lc.contrib, geom.group, D1, roots[p][c][p], roots[p][c][o],
                                   river_cfv_by_player[p][c], p, variant, iters, tol, safe=True)
            gaps[p] += float(np.sum(res[p].gap))
            its = max(its, int(np.max(res[p].iterations)))
        river.append([res[q].strategy[k] for k, q in enumerate(lc.template.players)])
    turn = [[turn_by_player[q][c][k] for k, q in enumerate(lc.template.players)] for c, lc in enumerate(upper.classes[1])]
    out = RangeCFR(full).exploitability([flop, turn, river])
    out.update(river_gap_sum=gaps, river_iterations=its)
    return out


def with_player(turn, new, player: int, upper) -> list:
    """turn[c][k] with player's nodes replaced by new[c][k]."""
    return [[new[c][k] if q == player else turn[c][k] for k, q in enumerate(lc.template.players)]
            for c, lc in enumerate(upper.classes[1])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", default="pot", choices=["pot", "limit"])
    ap.add_argument("--trunk-iters", type=int, default=100)
    ap.add_argument("--resolve-iters", default="100,300", help="turn re-solve iterations at which to grade")
    ap.add_argument("--sources", default="avg,br,unsafe")
    ap.add_argument("--leaf-tol", type=float, default=1e-2, help="ExactRiverLeaf tolerance (chips, summed)")
    ap.add_argument("--resolve-tol", type=float, default=1e-3, help="river re-solve tolerance (chips per root)")
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--inner", default="dcfr", choices=["cfr+", "dcfr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trunk-file", help="load the trunk from this pickle if it exists, else save it there")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    game = dg.make_game(FLOPS, 5, 3, a.game)
    full = holdem_public_tree(game)
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    leaf = ExactRiverLeaf(upper, river, a.leaf_tol, variant=a.inner, iters=a.sub_iters, values="br")
    result = {"args": vars(a), "git": dg.git_hash(), "flops": "2c3c4d,5c6d4c", "turn_lines": upper.classes[1][0].num_lines,
              "turn_deals": upper.streets[1].num_deals}

    def save():
        out.write_text(json.dumps(result, indent=1, default=float))

    tf = Path(a.trunk_file) if a.trunk_file else None
    if tf is not None and tf.exists():
        with open(tf, "rb") as f:
            saved = pickle.load(f)
        if saved["trunk_iters"] != a.trunk_iters or saved["game"] != a.game:
            raise SystemExit(f"{tf} holds a {saved['game']} trunk of {saved['trunk_iters']} iterations")
        print(f"trunk loaded from {tf}", flush=True)
    else:
        trunk = LeafCFR(upper, leaf, "dcfr", turn_cfv=True)
        while trunk.t < a.trunk_iters:
            trunk.iteration()
        avg = trunk.hand_sigma(trunk.average_strategy())
        saved = {"game": a.game, "trunk_iters": a.trunk_iters, "avg": avg, "seconds": trunk.solve_seconds,
                 "flop_turn": trunk.exploitability(),
                 "tcfv": [trunk.turn_cfv_average(q) for q in (0, 1)],
                 "leaf_avg": [[x / trunk.weight_sum[q] for x in trunk.leaf_cum[q]] for q in (0, 1)]}
    avg, ft = saved["avg"], saved["flop_turn"]
    reach = forward(upper, avg)
    roots = [r[0] for r in reach[1]]  # [c] (reach0, reach1) at the turn roots
    result["trunk"] = {"iterations": a.trunk_iters, "seconds": saved["seconds"], "flop_turn": ft}
    print(f"trunk {a.trunk_iters} its {saved['seconds']:.0f}s, flop+turn vs leaf {ft['exploitability']:.5f}", flush=True)
    river_trunk = [to_river_classes(upper, saved["leaf_avg"][1 - p]) for p in (0, 1)]
    if "grade" not in saved:
        t1 = time.perf_counter()
        saved["grade"] = composite(full, upper, leaf, avg[0], [avg[1], avg[1]], river_trunk, a.resolve_tol,
                                   a.sub_iters, a.inner)
        saved["grade_seconds"] = time.perf_counter() - t1
        if tf is not None:
            tf.parent.mkdir(parents=True, exist_ok=True)
            with open(tf.with_suffix(".tmp"), "wb") as f:
                pickle.dump(saved, f)
            tf.with_suffix(".tmp").replace(tf)
    base = saved["grade"]
    result["trunk"]["grade"] = base
    result["trunk"]["grade_seconds"] = saved["grade_seconds"]
    print(f"trunk grade (river re-solved) {base['exploitability']:.5f}", flush=True)
    save()

    bp = blueprint_values(upper.streets[1], upper.classes[1], avg[1], roots, leaf)
    tcfv = saved["tcfv"]
    bk = backup_values(upper.streets[1], upper.classes[1], avg[1], roots, saved["leaf_avg"])
    tsrc = {"avg": tcfv, "backup": bk, "onpolicy": bp["values"], "br": bp["br"]}
    result["t_values_vs_avg"] = {}
    for name, x in tsrc.items():
        row = {}
        for q in (0, 1):
            w = [r[q] for r in roots]
            size = sum(float((wc * np.abs(t)).sum()) for wc, t in zip(w, tcfv[q]))
            row[q] = {"l1": sum(float((wc * np.abs(y - t)).sum()) for wc, y, t in zip(w, x[q], tcfv[q])) / size,
                      "signed": sum(float((wc * (y - t)).sum()) for wc, y, t in zip(w, x[q], tcfv[q])) / size}
        result["t_values_vs_avg"][name] = row
        print(f"T values {name:8s} vs avg: L1 {row[0]['l1']:.4f} / {row[1]['l1']:.4f}, signed "
              f"{row[0]['signed']:+.4f} / {row[1]['signed']:+.4f} (p0 / p1, reach-weighted, over |avg|)", flush=True)
    save()
    marks = sorted(int(x) for x in a.resolve_iters.split(","))
    result["resolve"] = {}
    for source in [x for x in a.sources.split(",") if x]:  # none: the trunk only (--trunk-file)
        gadgets = {}
        for p in (0, 1):
            o = 1 - p
            cfv = None if source == "unsafe" else tsrc[source][o]
            gadgets[p] = LeafGadget(upper.streets[1], upper.classes[1], [r[p] for r in roots], cfv, p, leaf,
                                    reach_o=[r[o] for r in roots] if source == "unsafe" else None)
        for m in marks:
            row = {"players": {}}
            for p in (0, 1):
                o = 1 - p
                g = gadgets[p]
                ev = g.solve(m)
                ro, rp = roots[0][o], roots[0][p]
                tv = tcfv[o][0] if source == "unsafe" else tsrc[source][o][0]
                row["players"][p] = {
                    "iterations": g.t, "seconds": g.seconds, "leaf_calls": g.leaf_calls, "leaf_seconds": g.leaf_seconds,
                    "gap_sum": float(sum(x.sum() for x in ev["gap"])),
                    "max_gap_over_mass": ev["exploitability"],
                    "opp_br_minus_T_weighted": float((ro * (ev["br"][o][0] - tv)).sum()),
                    "opp_br_minus_T_positive_sum": float(np.maximum(ev["br"][o][0] - tv, 0.0).sum()),
                    "opp_T_abs_weighted": float((ro * np.abs(tv)).sum()),
                    "own_value_minus_trunk_avg": float((rp * (ev["values"][p][0] - tcfv[p][0])).sum()),
                    "own_value_minus_trunk_onpolicy": float((rp * (ev["values"][p][0] - bp["values"][p][0])).sum()),
                    "own_trunk_abs": float((rp * np.abs(tcfv[p][0])).sum())}
            t1 = time.perf_counter()
            turn_by = {p: with_player(avg[1], gadgets[p].average_strategy(), p, upper) for p in (0, 1)}
            rcfv = {p: to_river_classes(upper, gadgets[p].leaf_average(1 - p)) for p in (0, 1)}
            row["grade"] = composite(full, upper, leaf, avg[0], [turn_by[0], turn_by[1]], [rcfv[0], rcfv[1]],
                                     a.resolve_tol, a.sub_iters, a.inner)
            row["grade_seconds"] = time.perf_counter() - t1
            gs = sum(row["players"][p]["gap_sum"] for p in (0, 1))
            row["bound_half_gap_sum"] = gs / 2
            result["resolve"].setdefault(source, {})[m] = row
            print(f"{source:6s} {m:4d} its: grade {row['grade']['exploitability']:.5f} (trunk {base['exploitability']:.5f}), "
                  f"gaps p0 {row['players'][0]['gap_sum']:.2e} p1 {row['players'][1]['gap_sum']:.2e}, "
                  f"max gap/mass {max(row['players'][p]['max_gap_over_mass'] for p in (0, 1)):.2e}", flush=True)
            save()
    result["seconds"] = time.perf_counter() - t0
    save()


if __name__ == "__main__":
    main()
