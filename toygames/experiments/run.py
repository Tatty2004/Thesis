"""Run an experiment from a YAML config.

    python -m toygames.experiments.run toygames/experiments/configs/<name>.yaml [--workers 4] [--force]

A config expands into runs. Each run writes its own config (seed included), the
git hash and its outputs to results/<run config hash>/. The config's summary goes
to results/<config hash>/. A run whose result.json already exists at the current
clean git hash is reused unless --force is given.

Experiments:
  solve  full-game DCFR (plus the LP for small games) for every game in the sweep
  grid   the bucketing experiment (see grid.py)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from toygames.games import make_game
from toygames.solvers.cfr import CFR, LOG_FIELDS
from toygames.tree import load_or_compile

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"


# Bookkeeping ------------------------------------------------------------------------


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_info() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", True
    return {"git_hash": head, "git_dirty": dirty}


def start_run(cfg: dict) -> Path:
    """Create results/<hash>/ holding the run's config, seed and git hash."""
    run_dir = RESULTS / config_hash(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    meta = {"config_hash": config_hash(cfg), "seed": cfg.get("seed"), **git_info(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "numpy": np.__version__, "host": platform.node()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log = run_dir / "log.csv"
    if log.exists():
        log.unlink()
    return run_dir


def cached_result(cfg: dict, force: bool) -> dict | None:
    """The stored result of this run if it was produced by the current clean commit."""
    run_dir = RESULTS / config_hash(cfg)
    if force or not (run_dir / "result.json").exists():
        return None
    meta = json.loads((run_dir / "meta.json").read_text())
    now = git_info()
    if now["git_dirty"] or meta.get("git_dirty") or meta.get("git_hash") != now["git_hash"]:
        return None
    return json.loads((run_dir / "result.json").read_text())


def finish_run(run_dir: Path, result: dict) -> dict:
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def expand(sweep: dict) -> list[dict]:
    keys = sorted(sweep)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(sweep[k] for k in keys))]


def game_cfg(base: dict, overrides: dict) -> dict:
    return {**base, **overrides}


def build_game(gcfg: dict):
    params = {k: v for k, v in gcfg.items() if k != "name"}
    return make_game(gcfg["name"], **params)


def tree_for(gcfg: dict):
    return load_or_compile(build_game(gcfg), CACHE / "trees")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def run_parallel(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def markdown_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if r.get(c) is not None else ""
                                     for c in cols) + " |")
    return "\n".join(out)


# Experiment: solve --------------------------------------------------------------------


def solve_one(job) -> dict:
    cfg, force = job
    cached = cached_result(cfg, force)
    if cached is not None:
        return cached
    seed_everything(cfg["seed"])
    run_dir = start_run(cfg)
    tree = tree_for(cfg["game"])
    s = cfg["solver"]
    solver = CFR(tree, s["variant"], s.get("alpha", 1.5), s.get("beta", 0.0), s.get("gamma", 2.0))
    rows = solver.run(iterations=s.get("iterations"), tol=s.get("tol"), max_iterations=s.get("max_iterations", 100_000),
                      log_every=s["log_every"], log_path=run_dir / "log.csv")
    last = rows[-1]
    result = {"game": tree.name, "infosets": sum(tree.n_infosets), "terminals": tree.n_terminals,
              "iterations": solver.t, "solve_seconds": solver.solve_seconds,
              "ms_per_iteration": 1000 * solver.solve_seconds / solver.t,
              "exploitability": last["exploitability"], "game_value": last["game_value"],
              "br_value_p0": last["br_value_p0"], "br_value_p1": last["br_value_p1"]}
    if cfg.get("lp"):
        from toygames.solvers.lp import solve_game

        start = time.perf_counter()
        (v0, v1), _ = solve_game(tree)
        result.update(lp_value=v0, lp_value_p1=v1, lp_seconds=time.perf_counter() - start,
                      lp_in_bracket=bool(-last["br_value_p1"] - 1e-9 <= v0 <= last["br_value_p0"] + 1e-9))
    return finish_run(run_dir, result)


def run_solve(cfg: dict, out: Path, workers: int, force: bool) -> None:
    jobs = []
    for combo in expand(cfg.get("sweep", {})):
        gcfg = game_cfg(cfg["game"], combo)
        run = {"experiment": "solve", "seed": cfg["seed"], "game": gcfg, "solver": cfg["solver"],
               "lp": gcfg.get("num_ranks", 0) <= cfg.get("lp_max_ranks", 0)}
        jobs.append((run, force))
    for run, _ in jobs:  # compile once, up front, so workers only load the cache
        tree_for(run["game"])
    results = run_parallel(solve_one, jobs, workers)
    rows = []
    for (run, _), res in zip(jobs, results):
        rows.append({**{k: run["game"].get(k) for k in ("num_ranks", "deck_mode")}, **res,
                     "run": config_hash(run)})
    cols = ["num_ranks", "deck_mode", "infosets", "terminals", "iterations", "ms_per_iteration", "solve_seconds",
            "exploitability", "game_value", "lp_value", "lp_seconds", "run"]
    fmt = {"infosets": "{:,}", "terminals": "{:,}", "ms_per_iteration": "{:.1f}", "solve_seconds": "{:.1f}",
           "exploitability": "{:.2e}", "game_value": "{:+.6f}", "lp_value": "{:+.6f}", "lp_seconds": "{:.1f}"}
    table = markdown_table(rows, cols, fmt)
    (out / "summary.md").write_text(f"# {cfg.get('title', 'solve')}\n\nLog columns: {', '.join(LOG_FIELDS)}\n\n"
                                    f"{table}\n")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(table)


# Entry point ----------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="rerun runs that already have results")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = start_run(cfg)
    print(f"results/{out.name}/  ({git_info()['git_hash'][:10]}{' dirty' if git_info()['git_dirty'] else ''})",
          flush=True)
    if cfg["experiment"] == "solve":
        run_solve(cfg, out, args.workers, args.force)
    elif cfg["experiment"] == "grid":
        from toygames.experiments.grid import run_grid

        run_grid(cfg, out, args.workers, args.force)
    else:
        raise ValueError(f"unknown experiment {cfg['experiment']!r}")
    (out / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    sys.exit(main())
