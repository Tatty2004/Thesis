"""Time the torch river-subgame solver (valuenet/gpu_subgame.py) against the compiled one
(core/solvers/fused.py) on datagen's own samples, and have the compiled evaluator grade the
torch solver's strategies.

    python -m bombpot.experiments.valuenet.bench_gpu_subgame --n 6 --seed 0 --tol 1e-2 \
        --flops Ah7c2d,KsKd9h --out results/valuenet/datagen/gpu_bench.json

Per sample: the compiled solve (float64, the job's threads), the torch solve (float32 on the
device; --double adds float64), the torch strategy's exploitability by the compiled
evaluator, and how far the two solves' values are apart (range-weighted, in pots).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bombpot.core.solvers.fused import FusedSubgames
from bombpot.experiments.valuenet.datagen import FlopPair, random_range, sample_spot, turn_equity
from bombpot.valuenet.gpu_subgame import GpuSubgame, StreetSums, sweep_index


def cpu_grade(st, tp, contrib, r0, r1, gpu) -> float:
    ref = FusedSubgames(st, [tp], [contrib], [(r0, r1)], gpu.variant)
    for k in range(gpu.K):
        A = ref.nact[0, k]
        idx = ref.base[0] + np.arange(ref.D)[:, None] * ref.size[0] + ref.off[0, k] + np.arange(ref.H * A)
        ref.cum[idx] = gpu.cum[k].double().cpu().numpy().reshape(ref.D, -1)
    return ref.evaluate()["exploitability"][0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--flops", default="Ah7c2d,KsKd9h")
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--double", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    nr, ns = (int(x) for x in a.deck.split(","))
    ranks, suits = "23456789TJQKA"[:nr], "cdhs"
    fixed = tuple(tuple(sorted(ranks.index(f[i]) * ns + suits.index(f[i + 1]) for i in (0, 2, 4)))
                  for f in a.flops.split(","))
    rng = np.random.default_rng(a.seed)
    fp = FlopPair(fixed, nr, ns, "pot", "loguniform")
    dev = torch.device(a.device)
    print(f"device {dev} {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''} | torch {torch.__version__} "
          f"| threads {torch.get_num_threads()}", flush=True)
    rows = []
    for n in range(a.n):
        sample_spot(rng, nr, ns)  # datagen's draw order
        rest = rng.permutation([c for c in range(nr * ns) if c not in {x for f in fixed for x in f}])
        turn = (int(rest[0]), int(rest[1]))
        bet = fp.sample_betting(rng, 2.0)
        d = fp.turn_deal(turn)
        valid = fp.upper.streets[1].valid[d]
        st = fp.river_street(d)
        strength = turn_equity(st).mean(axis=1)
        r0, r1 = random_range(rng, valid, strength), random_range(rng, valid, strength)
        pot, tp = bet["pot"], bet["template"]
        tol = a.tol * pot
        row = {"sample": n, "pot": pot, "nodes": tp.num_nodes, "deals": int(st.num_deals), "hands": len(r0)}
        t0 = time.perf_counter()
        ix = sweep_index(st)
        row["index_seconds"] = time.perf_counter() - t0
        ref = None
        if not a.skip_cpu:
            f = FusedSubgames(st, [tp], [pot / 2], [(r0, r1)], "dcfr")
            ref = f.solve(5000, tol)
            row.update(cpu_seconds=f.seconds, cpu_iters=ref["iterations"][0], cpu_expl=ref["exploitability"][0] / pot)
        for name, dt in (("gpu32", torch.float32),) + ((("gpu64", torch.float64),) if a.double else ()):
            t0 = time.perf_counter()
            sums = StreetSums(st, dev, dt, ix)
            g = GpuSubgame(sums, tp, pot / 2, r0, r1, "dcfr", dtype=dt)
            ev = g.solve(5000, tol)
            wall = time.perf_counter() - t0
            row.update({f"{name}_seconds": g.seconds, f"{name}_wall": wall, f"{name}_iters": ev["iterations"],
                        f"{name}_expl": ev["exploitability"] / pot,
                        f"{name}_expl_by_cpu": cpu_grade(st, tp, pot / 2, r0, r1, g) / pot})
            if ref is not None:
                row[f"{name}_value_gap"] = max(abs(((ev["values"][p] - ref["values"][0][p]) * r).sum()) / ev["mass"] / pot
                                               for p, r in ((0, r0), (1, r1)))
            if dev.type == "cuda":
                row[f"{name}_gpu_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
            del g, sums
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    tot = lambda k: sum(r[k] for r in rows[1:]) if len(rows) > 1 and k in rows[0] else float("nan")  # noqa: E731
    print(f"RESULT after the first sample (compile / warm-up): cpu {tot('cpu_seconds'):.1f}s  gpu32 solve "
          f"{tot('gpu32_seconds'):.1f}s wall {tot('gpu32_wall'):.1f}s  index {tot('index_seconds'):.1f}s  over {len(rows) - 1} samples")


if __name__ == "__main__":
    main()
