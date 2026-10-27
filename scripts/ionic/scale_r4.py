"""Scaling test (PLAN_full_solver.md, S0): time R4's DCFR iterations at the job's thread count.

Builds the game and solver exactly as `bombpot.experiments.run` does for
configs/holdem/flop_turn.yaml, runs `--iterations` iterations without logging, then
grades the average strategy once. Prints one JSON line (also written to `--out`).

    .venv/bin/python scripts/ionic/scale_r4.py --iterations 25 --out logs/scaling/<jobid>.json
"""
import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import numba
import yaml

from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.experiments.games import public_tree
from bombpot.experiments.run import build_game, expand, game_cfg

CONFIG = Path(__file__).resolve().parents[2] / "bombpot/experiments/configs/holdem/flop_turn.yaml"


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    (combo,) = expand(cfg["sweep"])
    s = cfg["solver"]
    start = time.perf_counter()
    pt = public_tree(build_game(game_cfg(cfg["game"], combo)))
    build_seconds = time.perf_counter() - start
    solver = RangeCFR(pt, s["variant"], s["alpha"], s["beta"], s["gamma"])

    times = []
    for _ in range(args.iterations):
        before = solver.solve_seconds
        solver.iteration()
        times.append(solver.solve_seconds - before)

    start = time.perf_counter()
    ex = solver.exploitability()
    eval_seconds = time.perf_counter() - start

    row = {"host": platform.node().split(".")[0], "cpu_model": cpu_model(),
           "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
           "affinity_cpus": len(os.sched_getaffinity(0)), "numba_threads": numba.get_num_threads(),
           "threading_layer": numba.threading_layer(),
           "iterations": solver.t, "build_seconds": build_seconds,
           "first_iteration": times[0],  # includes compiling the kernels
           "mean_after_first": statistics.mean(times[1:]), "median_after_first": statistics.median(times[1:]),
           "min": min(times[1:]), "max": max(times[1:]), "eval_seconds": eval_seconds,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, **ex}
    line = json.dumps(row)
    print(line, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(line + "\n")


if __name__ == "__main__":
    main()
