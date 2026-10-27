"""Export an M1 checkpoint's average strategy for the strategy viewer (viewer/index.html).

    python -m bombpot.experiments.viewer_export --checkpoint P [--out DIR] [--fixtures]

The checkpoint's run folder (results/m1/<config>/) supplies the config and river buckets. The
default output is results/viewer/<method>_t<t>/:

  meta.json                 cards, hands, turn deals, betting templates, contributions, bet sizes,
                            the checkpoint, and the encoding below
  flop.bin                  street 0, shaped (1, 1, H, W)
  turn/<turn A>.bin         street 1 for the 45 turn deals with that board-A turn card, (45, 5, H, W)
  river/<turn A>.bin        street 2 for the same turn deals, (45, 25, ids, W)
  centers/<turn A>.bin      those turn deals' bucket centers, float64, (45, k, dim)
  check.json                the export's own checks (decoding error, row sums, the flop report)
  fixtures/                 with --fixtures: references for viewer/test_core.mjs

Encoding: every (deal, line, hand or bucket id) row holds W = sum over nodes of (actions - 1)
uint8 values, node k's at bytes off[k] .. off[k] + A_k - 2. They are the average strategy
times 255, rounded by largest remainder so that each node's values sum to 255; the last action's
value is dropped and is 255 minus the others. Turn deals are in HoldemRiver.turn_cards order,
which groups them by board A's turn card, 45 at a time.

Refresh (checkpoints arrive every 2 h; the 2 h milestones are hard-linked in keep/):

    cd /n/fs/ugrad/ug27/rb5709/Thesis
    ls results/m1/m1_kmeans2d/solve/                     # the newest ckpt_t*.npz
    NUMBA_NUM_THREADS=8 nice .venv/bin/python -m bombpot.experiments.viewer_export \\
        --checkpoint results/m1/m1_kmeans2d/solve/ckpt_t<T>.npz --fixtures
    node viewer/test_core.mjs results/viewer/kmeans_2d_t<T>

then republish every data file of results/viewer/kmeans_2d_t<T>/ to the same artifact URL, in
batches under 64 MB (viewer/README.md).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from bombpot.core.solvers.range_cfr import forward, next_root, street_reach
from bombpot.experiments.m1 import build
from bombpot.experiments.run import RESULTS, git_info
from bombpot.holdem.evaluator import hand_values

SCALE = 255
TURN_GROUP = 45  # turn deals per board-A turn card


def quantize(p: np.ndarray) -> np.ndarray:
    """Rows of probabilities (..., A) as uint8 values summing to 255, by largest remainder."""
    x = p * SCALE
    f = np.floor(x)
    short = np.clip(SCALE - f.sum(-1, keepdims=True), 0, p.shape[-1]).astype(np.int64)
    order = np.argsort(-(x - f), axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(p.shape[-1]), axis=-1)
    q = f.astype(np.int64) + (rank < short)
    if (q.sum(-1) != SCALE).any() or q.min() < 0 or q.max() > SCALE:
        raise ValueError("quantization failed")
    return q.astype(np.uint8)


def dequantize(q: np.ndarray) -> np.ndarray:
    """The stored values (..., A - 1) as probabilities (..., A), as the page decodes them."""
    q = q.astype(np.int64)
    return np.concatenate([q, SCALE - q.sum(-1, keepdims=True)], axis=-1) / SCALE


def average(cum: np.ndarray) -> np.ndarray:
    """SampledRangeCFR.average_strategy for one array."""
    total = cum.sum(-1, keepdims=True)
    return np.where(total > 0, cum / np.where(total > 0, total, 1.0), 1.0 / cum.shape[-1])


def template_json(tp) -> dict:
    return {"players": tp.players, "actions": [list(a) for a in tp.actions], "hist": tp.hist,
            "children": [[list(c) for c in row] for row in tp.children],
            "closes": [[h, e] for h, e in tp.closes]}


def write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(data).tobytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out")
    ap.add_argument("--fixtures", action="store_true", help="also write test references for the page's core")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    start = time.perf_counter()

    ckpt = Path(args.checkpoint).resolve()
    run_dir = ckpt.parent.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    game, pt, river, buckets = build(cfg, run_dir)
    info = json.loads(ckpt.with_suffix(".json").read_text()) if ckpt.with_suffix(".json").exists() else {}
    with np.load(ckpt) as z:
        t_ckpt, solve_seconds = int(z["t"]), float(z["solve_seconds"])
    method = buckets.method
    out = Path(args.out) if args.out else RESULTS / "viewer" / f"{method}_t{t_ckpt}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{ckpt.name}: {method}, t {t_ckpt}, {solve_seconds / 3600:.3f} h of solving -> {out}", flush=True)

    templates = pt.templates
    A = [[len(a) for a in tp.actions] for tp in templates]
    if any(row != A[0] for row in A):
        raise SystemExit("the page assumes one betting template shape on every street")
    off = np.concatenate([[0], np.cumsum([a - 1 for a in A[0]])]).astype(int)
    W = int(off[-1])
    D1 = pt.streets[1].num_deals
    tc = river.turn_cards
    shards = [int(tc[s * TURN_GROUP, 0]) for s in range(D1 // TURN_GROUP)]
    if D1 % TURN_GROUP or any((tc[s * TURN_GROUP:(s + 1) * TURN_GROUP, 0] != c).any() for s, c in enumerate(shards)):
        raise SystemExit("turn deals aren't grouped by board A's turn card")
    names = [game.deck.card_str(c) for c in shards]

    # Strategy, street by street: quantize, check against the float average, keep both for the fixtures.
    rng = np.random.default_rng(args.seed)
    enc, exact, checks = [], [], {"max_abs_error": {}, "min_last": {}}
    shapes = [pt.shape(0), pt.shape(1), (buckets.num_rows, len(pt.contrib[2]), buckets.num_ids)]
    points = []  # (t, d, line, i, k) to reread from the files later, with the float average there
    for _ in range(200):
        t = int(rng.integers(3))
        points.append((t,) + tuple(int(rng.integers(n)) for n in shapes[t]) + (int(rng.integers(len(A[t]))),))
    expected = {}
    with np.load(ckpt) as z:
        for t in range(3):
            shape = z[f"cum_t{t}_k0"].shape[:3]
            if shape != shapes[t]:
                raise SystemExit(f"street {t}: checkpoint arrays are {shape}, not {shapes[t]}")
            U = np.empty(shape + (W,), dtype=np.uint8)
            ex = []
            for k in range(len(A[t])):
                p = average(z[f"cum_t{t}_k{k}"])
                for pt_ in points:
                    if pt_[0] == t and pt_[4] == k:
                        expected[pt_] = p[pt_[1:4]].copy()
                q = quantize(p)
                U[..., off[k]:off[k + 1]] = q[..., :-1]
                err = float(np.abs(dequantize(q[..., :-1]) - p).max())
                checks["max_abs_error"][f"t{t}_k{k}"] = err
                if err >= 1 / SCALE:
                    raise SystemExit(f"street {t} node {k}: decoding error {err} is not below 1/255")
                ex.append(p if t < 2 else None)
            enc.append(U)
            exact.append(ex)
            print(f"street {t}: {shape} x {W} bytes, max decoding error "
                  f"{max(v for key, v in checks['max_abs_error'].items() if key.startswith(f't{t}_')):.5f}", flush=True)

    write(out / "flop.bin", enc[0])
    for s, name in enumerate(names):
        rows = slice(s * TURN_GROUP, (s + 1) * TURN_GROUP)
        write(out / "turn" / f"{name}.bin", enc[1][rows])
        write(out / "river" / f"{name}.bin", enc[2][rows])
        write(out / "centers" / f"{name}.bin", buckets.centers[rows].astype("<f8"))

    # Reread the files at random points and compare with the float average.
    def stored(t, d, line, i, k):
        if t == 0:
            path, pos, shape = out / "flop.bin", 0, enc[0].shape
        else:
            path = out / ("turn" if t == 1 else "river") / f"{names[d // TURN_GROUP]}.bin"
            pos, shape = d % TURN_GROUP, enc[t].shape
        row = ((pos * shape[1] + line) * shape[2] + i) * W
        with open(path, "rb") as f:
            f.seek(row + off[k])
            return np.frombuffer(f.read(A[t][k] - 1), dtype=np.uint8)

    errors = [float(np.abs(dequantize(stored(*key)) - expected[key]).max()) for key in points]
    checks["reread_points"] = len(errors)
    checks["reread_max_abs_error"] = max(errors)
    if max(errors) >= 1 / SCALE:
        raise SystemExit(f"reread error {max(errors)} is not below 1/255")
    for t in range(3):
        for k in range(len(A[t])):
            last = SCALE - enc[t][..., off[k]:off[k + 1]].astype(np.int64).sum(-1)
            checks["min_last"][f"t{t}_k{k}"] = int(last.min())
            if last.min() < 0:
                raise SystemExit(f"street {t} node {k}: stored values sum past 255")

    # The page must reproduce the solve log's flop report.
    hidx = {h: i for i, h in enumerate(pt.hands)}
    flop_hands = cfg.get("run", {}).get("flop_hands", [])
    bet = exact[0][0][0, 0, :, 1]
    checks["flop_p0_bet"] = {"overall": round(float(bet.mean()), 4),
                            **{h: round(float(bet[hidx[tuple(sorted(game.deck.parse(h)))]]), 4) for h in flop_hands}}
    if info.get("flop_p0_bet") and info["flop_p0_bet"] != checks["flop_p0_bet"]:
        raise SystemExit(f"flop report {checks['flop_p0_bet']} differs from the checkpoint's {info['flop_p0_bet']}")

    git = git_info()
    meta = {
        "format": 1,
        "method": method, "k": buckets.k, "num_ids": buckets.num_ids, "dim": int(buckets.centers.shape[2]),
        "checkpoint": {"path": str(ckpt.relative_to(RESULTS.resolve())) if ckpt.is_relative_to(RESULTS.resolve())
                       else str(ckpt), "t": t_ckpt, "solve_seconds": solve_seconds,
                       "saved": info.get("saved"), "git_hash": info.get("git_hash"), "slurm_job": info.get("slurm_job"),
                       "value": info.get("value")},
        "export": {"git_hash": git["git_hash"], "git_dirty": git["git_dirty"],
                   "exported": time.strftime("%Y-%m-%d %H:%M:%S")},
        "cards": {"ranks": "23456789TJQKA", "suits": "cdhs", "num_suits": game.num_suits},
        "flops": [list(f) for f in game.flops],
        "hands": [list(h) for h in pt.hands],
        "turn_cards": tc.tolist(),
        "turn_kappa": pt.streets[1].kappa.tolist(),
        "river_scale": river.scale,
        "ante": game.ante, "bet_sizes": list(game.bet_sizes),
        "templates": [template_json(tp) for tp in templates],
        "contrib": [c.tolist() for c in pt.contrib],
        "encoding": {"dtype": "uint8", "scale": SCALE, "width": W, "offsets": off.tolist(),
                     "turn_group": TURN_GROUP,
                     "shapes": {"flop": list(enc[0].shape), "turn": [TURN_GROUP] + list(enc[1].shape[1:]),
                                "river": [TURN_GROUP] + list(enc[2].shape[1:]),
                                "centers": [TURN_GROUP] + list(buckets.centers.shape[1:])}},
        "shards": names,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "check.json").write_text(json.dumps(checks, indent=1))
    print(f"wrote {out}: max decoding error {max(checks['max_abs_error'].values()):.5f}, reread "
          f"{checks['reread_max_abs_error']:.5f}, flop report {checks['flop_p0_bet']}", flush=True)

    if args.fixtures:
        write_fixtures(out / "fixtures", game, pt, river, buckets, enc, exact, off, rng)
    print(f"done in {time.perf_counter() - start:.0f} s", flush=True)
    return 0


def write_fixtures(folder: Path, game, pt, river, buckets, enc, exact, off, rng) -> None:
    """References for viewer/test_core.mjs:
      evaluator.json   hand_values of random 7-card sets, every category included
      buckets.json     equity and bucket ids at 100 random river deals, both players
      reach.json       both players' reach (forward() on the decoded strategy) at 20 turn and 10 river points
    """
    folder.mkdir(parents=True, exist_ok=True)
    ns = game.num_suits

    # (a) Evaluator: 500 random 7-card sets, plus up to 40 of each category from a larger draw.
    draws = np.array([rng.choice(52, size=7, replace=False) for _ in range(100_000)], dtype=np.int64)
    values = hand_values(draws, ns)
    pick = list(range(500))
    for cat in range(9):
        idx = np.flatnonzero(values // 16**5 == cat)
        pick += list(idx[idx >= 500][:40])
    (folder / "evaluator.json").write_text(json.dumps(
        {"cards": draws[pick].tolist(), "values": values[pick].tolist(),
         "categories": np.bincount(values[pick] // 16**5, minlength=9).tolist()}))

    # (b) Equity and bucket ids at random river deals (scale 1, as the exact evaluation builds them),
    # and how often the solver's sampling scale changes an id.
    scale = (pt.streets[1].num_deals / 64) * (river.num_children / 4)
    deals, solver_mismatch = [], 0
    for _ in range(100):
        d, c = int(rng.integers(pt.streets[1].num_deals)), int(rng.integers(river.num_children))
        st = river.street([d], [c])
        eq = st.equity()[0]
        ids = [buckets.ids(st, [d], [c], p)[0] for p in (0, 1)]
        st2 = river.street([d], [c], scale)
        solver_mismatch += int((buckets.ids(st2, [d], [c], 0)[0] != ids[0]).sum())
        deals.append({"turn_deal": d, "child": c, "river": river.river_cards(np.array([d]), np.array([c]))[0].tolist(),
                      "kappa": float(st.kappa[0]), "possible": st.valid[0].astype(int).tolist(),
                      "eq": eq.tolist(), "ids": [i.tolist() for i in ids]})
    (folder / "buckets.json").write_text(json.dumps({"deals": deals, "solver_scale_id_mismatches": solver_mismatch}))
    print(f"fixtures: bucket ids differ between scale 1 and the solver's scale at {solver_mismatch} of "
          f"{100 * len(pt.hands)} hand-deals", flush=True)

    # (c) Reach on the decoded strategy, the page's own arithmetic; the float strategy's for comparison.
    W = off[-1]
    tps = pt.templates
    A = [len(a) for a in tps[0].actions]

    def decoded(U):
        return [dequantize(U[..., off[k]:off[k + 1]]) for k in range(len(A))]

    dec = [decoded(enc[0]), decoded(enc[1])]
    reach = forward(pt, dec)
    reach_exact = forward(pt, [exact[0], exact[1]])
    turn_points = []
    for _ in range(20):
        d, line, k = int(rng.integers(pt.streets[1].num_deals)), int(rng.integers(5)), int(rng.integers(len(A)))
        r = [reach[1][k][p][d, line] for p in (0, 1)]
        re = [reach_exact[1][k][p][d, line] for p in (0, 1)]
        turn_points.append({"turn_deal": d, "line": line, "node": k, "reach": [x.tolist() for x in r],
                            "quantization_effect": max(float(np.abs(x - y).max()) for x, y in zip(r, re))})
    _, closing = street_reach(tps[1], dec[1], reach[1][0])
    river_points = []
    for _ in range(10):
        d, c = int(rng.integers(pt.streets[1].num_deals)), int(rng.integers(river.num_children))
        st = river.street([d], [c])
        root = next_root(closing, np.array([d]))
        ids = [buckets.ids(st, [d], [c], p)[0] for p in (0, 1)]
        U = enc[2][d]  # (25, ids, W)
        hsig = [dequantize(U[:, ids[q], off[k]:off[k + 1]])[None] for k, q in enumerate(tps[2].players)]
        nodes, _ = street_reach(tps[2], hsig, root)
        line, k = int(rng.integers(25)), int(rng.integers(len(A)))
        river_points.append({"turn_deal": d, "child": c,
                             "river": river.river_cards(np.array([d]), np.array([c]))[0].tolist(),
                             "line": line, "node": k, "reach": [nodes[k][p][0, line].tolist() for p in (0, 1)]})
    (folder / "reach.json").write_text(json.dumps({"turn": turn_points, "river": river_points}))
    print(f"fixtures: reach at {len(turn_points)} turn and {len(river_points)} river points; largest effect of "
          f"quantization on a turn reach {max(p['quantization_effect'] for p in turn_points):.5f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
