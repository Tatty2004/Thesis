"""M1: the first full solve of real double-board Hold'em on the 52-card deck (flop, turn and
river, both boards) for one pair of flops. The procedure and its checks are in
bombpot/holdem/M1_CLAUDE.md; the Slurm scripts are scripts/ionic/m1_*.sbatch.

    python -m bombpot.experiments.m1 fit --config C            fit the river buckets, then check them
    python -m bombpot.experiments.m1 check-fit --config C      the checks alone (Gate 3)
    python -m bombpot.experiments.m1 solve --config C [--resume] [--tag T] [overrides]
    python -m bombpot.experiments.m1 evaluate --checkpoint P [--max-minutes M]
    python -m bombpot.experiments.m1 compare A B               are two checkpoints bit-identical?

A config's run folder is results/m1/<config name>/:
  config.yaml, meta.json, fit.log     the config the folder was made with; the fit's bookkeeping
  buckets.npz, fit_states.json        river buckets per turn deal; the generator's state before each
  fit_checks.json                     the fit's checks
  solve[-<tag>]/                      one solve: solve.log, runs.jsonl and its checkpoints:
    ckpt_t<t>.npz                       SampledRangeCFR.save
    ckpt_t<t>.json                      what the solve knew then: health check, timings, flop strategy
    ckpt_t<t>.eval.json                 the exact check of its average strategy (evaluate)

The solve runs until a time budget, counted from the start of the process, that must end at
least 30 minutes before the Slurm job's walltime. It checkpoints every `checkpoint_hours` and
at the end, each time after a health check (every array finite, every average-strategy row
summing to 1, t increased); a failed check stops it without saving. Retention keeps the newest
2 checkpoints and the first at or after each of `milestone_hours` hours of solving, and deletes
the rest (evaluations stay).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from bombpot.core.abstraction.river import EquityBuckets, fit_equity_buckets
from bombpot.core.solvers.sampled_cfr import Lossless, SampledRangeCFR, exact_exploitability
from bombpot.experiments.run import RESULTS, git_info
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

M1 = RESULTS / "m1"
MARGIN_MINUTES = 30  # the time budget must end at least this long before the job's walltime
RUN_DEFAULTS = {"log_minutes": 5.0, "milestone_hours": [2, 4, 8, 16, 32, 64], "eval_chunk": 256,
                "eval_in_process": False, "flop_hands": []}


# Bookkeeping ----------------------------------------------------------------------------


class Log:
    """Timestamped lines to stdout (the Slurm log) and to a file."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def peak_gb() -> float:
    """This process's peak resident memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux


def environment() -> dict:
    import numba
    import scipy

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"host": platform.node(), "cpu_model": cpu, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "numba_threads": numba.get_num_threads(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__}


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def job_end_time() -> float | None:
    """When the Slurm job running this process reaches its walltime (Unix time), or None
    outside Slurm."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    for attempt in range(3):
        try:
            out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True,
                                 timeout=60).stdout.strip()
            return time.mktime(time.strptime(out, "%Y-%m-%dT%H:%M:%S"))
        except (OSError, ValueError, subprocess.SubprocessError):
            time.sleep(30 * (attempt + 1))
    raise SystemExit(f"can't read job {job}'s end time from squeue, so the time budget can't be checked")


# The game, its trees and the river abstraction --------------------------------------------


def load_config(path) -> tuple[str, dict]:
    path = Path(path)
    return path.stem, yaml.safe_load(path.read_text())


def make_game(cfg: dict) -> Holdem:
    return Holdem(**{**cfg["game"], "flops": tuple(cfg["game"]["flops"])})


def keep_config(run_dir: Path, cfg: dict) -> None:
    """The run folder keeps the config it was made with. A config whose game or buckets
    differ from it is refused, so buckets and checkpoints can't be mixed up."""
    path = run_dir / "config.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    old = yaml.safe_load(path.read_text())
    for key in ("game", "buckets"):
        if old.get(key) != cfg.get(key):
            raise SystemExit(f"{path} has {key} {old.get(key)}, not {cfg.get(key)}: a different run")


def build(cfg: dict, run_dir: Path):
    """The game, its flop-and-turn public tree, the river, and the river abstraction (the
    run folder's fitted buckets, or the lossless river)."""
    game = make_game(cfg)
    pt = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, pt)
    b = cfg["buckets"]
    D1 = pt.streets[-1].num_deals
    if b["method"] == "lossless":
        return game, pt, river, Lossless(D1, river.num_children, len(pt.hands))
    buckets = EquityBuckets.load(run_dir / "buckets.npz")
    fit = {"samples": b["samples"], "seed": b["seed"], "restarts": b["restarts"]}
    if (buckets.method, buckets.k, buckets.num_rows, buckets.fit) != (b["method"], b["k"], D1, fit):
        raise SystemExit(f"{run_dir / 'buckets.npz'} doesn't match the config's buckets {b}")
    return game, pt, river, buckets


def make_solver(cfg: dict, pt, river, abstraction) -> SampledRangeCFR:
    s = cfg["solver"]
    return SampledRangeCFR(pt, river, abstraction, batch=s["batch"], samples=s["samples"], seed=s["seed"])


# Fitting the buckets ------------------------------------------------------------------------


class OneParent:
    """The last street as fit_equity_buckets sees it when it fits a single parent: its
    parent 0 is `parent`."""

    def __init__(self, last, parent: int):
        self.last, self.parent, self.num_children = last, parent, last.num_children

    def street(self, parents, children, scale: float = 1.0):
        return self.last.street(np.asarray(parents) + self.parent, children, scale)


def fit_by_parent(last, num_parents: int, method: str, k: int, samples: int, seed: int, restarts: int,
                  log=None, log_every: int = 50) -> tuple[EquityBuckets, list]:
    """fit_equity_buckets one parent at a time on one shared generator (default_rng returns a
    Generator unaltered), so the centers are bit-identical to a single call. It also records
    the generator's state before each parent, from which any one parent's fit can be repeated
    alone (refit_one), and logs progress."""
    rng = np.random.default_rng(seed)
    centers, states = [], []
    start = time.perf_counter()
    for p in range(num_parents):
        states.append(rng.bit_generator.state)
        centers.append(fit_equity_buckets(OneParent(last, p), 1, method, k, samples, seed=rng,
                                          restarts=restarts).centers[0])
        if log is not None and ((p + 1) % log_every == 0 or p + 1 == num_parents):
            each = (time.perf_counter() - start) / (p + 1)
            log(f"fitted {p + 1}/{num_parents} turn deals, {each:.2f} s each, "
                f"{each * (num_parents - p - 1) / 60:.1f} min to go")
    return EquityBuckets(np.stack(centers), method, {"samples": samples, "seed": seed, "restarts": restarts}), states


def refit_one(last, parent: int, method: str, k: int, samples: int, state: dict, restarts: int) -> np.ndarray:
    """One parent's centers, fitted again from the generator state recorded before it."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return fit_equity_buckets(OneParent(last, parent), 1, method, k, samples, seed=rng, restarts=restarts).centers[0]


def same_bits(x, y) -> bool:
    x, y = np.ascontiguousarray(np.atleast_1d(x)), np.ascontiguousarray(np.atleast_1d(y))
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x.view(np.uint8), y.view(np.uint8))


def cmd_fit(args) -> int:
    name, cfg = load_config(args.config)
    run_dir = M1 / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "fit.log")
    b = cfg["buckets"]
    if b["method"] != "lossless" and (run_dir / "buckets.npz").exists() and not args.force:
        raise SystemExit(f"{run_dir / 'buckets.npz'} exists (--force refits)")
    keep_config(run_dir, cfg)
    git = git_info()
    meta = {"config": cfg, **git, **environment(), "started": now()}
    start = time.perf_counter()
    game = make_game(cfg)
    pt = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, pt)
    D1 = pt.streets[-1].num_deals
    meta["build_seconds"] = time.perf_counter() - start
    log(f"fit {name}: {b}, {D1} turn deals, git {git['git_hash'][:10]}{' dirty' if git['git_dirty'] else ''}")
    if b["method"] == "lossless":
        meta["note"] = "lossless river: nothing to fit"
    else:
        start = time.perf_counter()
        buckets, states = fit_by_parent(river, D1, b["method"], b["k"], b["samples"], b["seed"], b["restarts"], log)
        meta.update(fit_seconds=time.perf_counter() - start, centers_shape=list(buckets.centers.shape))
        meta["seconds_per_turn_deal"] = meta["fit_seconds"] / D1
        buckets.save(run_dir / "buckets.npz")
        write_json(run_dir / "fit_states.json", states)
    meta.update(finished=now(), peak_gb=peak_gb())
    write_json(run_dir / "meta.json", meta)
    log(f"done in {meta.get('fit_seconds', 0.0):.0f} s, peak {meta['peak_gb']:.2f} GB")
    return 0


def cmd_check_fit(args) -> int:
    """Gate 3's checks of a fit: shape and range, determinism, well-formed ids, occupancy, and
    a clean git hash. With --full, also a single fit_equity_buckets call over every parent
    (as long as the fit itself) must give the same centers."""
    name, cfg = load_config(args.config)
    run_dir = M1 / name
    log = Log(run_dir / "fit.log")
    b = cfg["buckets"]
    if b["method"] == "lossless":
        log("lossless river: no buckets to check")
        return 0
    game, pt, river, buckets = build(cfg, run_dir)
    D1, k = pt.streets[-1].num_deals, b["k"]
    states = json.loads((run_dir / "fit_states.json").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    c = buckets.centers
    dim = 1 if b["method"] == "avg_1d" else 2
    checks = {"shape": list(c.shape) == [D1, k, dim], "finite": bool(np.isfinite(c).all()),
              "in_0_1": bool(((c >= 0) & (c <= 1)).all()),
              "weakest_first": bool((np.diff(c.mean(axis=2), axis=1) >= 0).all())}
    refits = {str(p): same_bits(refit_one(river, p, b["method"], k, b["samples"], states[p], b["restarts"]), c[p])
              for p in (0, D1 // 2, D1 - 1)}
    first = fit_equity_buckets(OneParent(river, 0), 1, b["method"], k, b["samples"], seed=b["seed"],
                               restarts=b["restarts"]).centers[0]
    refits["0 from the seed"] = same_bits(first, c[0])
    checks["deterministic"] = all(refits.values())
    if args.full:
        whole = fit_equity_buckets(river, D1, b["method"], k, b["samples"], seed=b["seed"], restarts=b["restarts"])
        checks["one_call_identical"] = same_bits(whole.centers, c)
    # Ids at 5 random turn deals x 5 random rivers each: 0..k-1 if the hand can be held, else k.
    rng = np.random.default_rng(args.seed)
    parents = np.repeat(rng.choice(D1, size=5, replace=False), 5)
    children = np.concatenate([rng.choice(river.num_children, size=5, replace=False) for _ in range(5)])
    st = river.street(parents, children)
    well_formed, counts = True, []
    for p in (0, 1):
        ids, can = buckets.ids(st, parents, children, p), st.possible(p)
        well_formed &= bool((ids[can] < k).all() and (ids[can] >= 0).all() and (ids[~can] == k).all())
        counts += [np.bincount(ids[d][can[d]], minlength=k) for d in range(len(parents))] if p == 0 else []
    checks["ids_well_formed"] = well_formed
    checks["clean_git"] = not meta["git_dirty"] and meta["git_hash"] != "unknown"
    occ = np.concatenate(counts)
    occupancy = {"deals": len(parents), "hands_per_deal": int(st.possible(0)[0].sum()), "min": int(occ.min()),
                 "median": float(np.median(occ)), "max": int(occ.max()), "empty_share": float((occ == 0).mean())}
    out = {"checks": checks, "refits": refits, "occupancy": occupancy, "passed": all(checks.values()),
           "fit_git_hash": meta["git_hash"], "checked": now(), **environment()}
    write_json(run_dir / "fit_checks.json", out)
    log(f"checks {'PASS' if out['passed'] else 'FAIL'}: {checks}, refits {refits}")
    log(f"occupancy (hands per bucket over {len(parents)} river deals): {occupancy}")
    return 0 if out["passed"] else 1


# Solving -------------------------------------------------------------------------------------


def checkpoints(folder: Path) -> list[tuple[int, Path]]:
    return sorted((int(p.name[len("ckpt_t"):-len(".npz")]), p) for p in folder.glob("ckpt_t*.npz"))


def keep_set(entries, milestones) -> set:
    """Which checkpoints to keep, from (t, hours of solving) of each: the newest 2, and the
    first at or after each milestone."""
    entries = sorted(entries)
    keep = {t for t, _ in entries[-2:]}
    for m in milestones:
        first = next((t for t, h in entries if h >= m), None)
        if first is not None:
            keep.add(first)
    return keep


def retain(folder: Path, milestones, log) -> None:
    entries, paths = [], {}
    for t, p in checkpoints(folder):
        with np.load(p) as z:
            entries.append((t, float(z["solve_seconds"]) / 3600))
        paths[t] = p
    keep = keep_set(entries, milestones)
    for t, p in paths.items():
        if t not in keep:
            p.unlink()
            p.with_suffix(".json").unlink(missing_ok=True)
            log(f"retention: deleted {p.name}")


def health_check(solver: SampledRangeCFR, avg: list, previous_t: int) -> dict:
    finite = all(bool(np.isfinite(a).all()) for name in ("regret", "cum", "sigma")
                 for row in getattr(solver, name) for a in row)
    err = max(float(np.abs(s.sum(-1) - 1.0).max()) for row in avg for s in row)
    return {"passed": bool(finite and err <= 1e-9 and solver.t > previous_t), "finite": finite,
            "max_row_sum_error": err, "t": solver.t, "previous_t": previous_t}


def flop_report(game: Holdem, pt, avg: list, hands) -> dict:
    """Player 0's average strategy at the flop's first decision: the bet frequency of each
    listed hand, and over all hands (all equally likely there)."""
    tp = pt.templates[0]
    if tp.players[0] != 0 or tuple(tp.actions[0]) != ("c", "r"):
        raise ValueError(f"the flop's first node isn't player 0's check or bet: {tp.players[0]} {tp.actions[0]}")
    bet = avg[0][0][0, 0, :, 1]
    index = {h: i for i, h in enumerate(pt.hands)}
    out = {h: round(float(bet[index[tuple(sorted(game.deck.parse(h)))]]), 4) for h in hands}
    return {"overall": round(float(bet.mean()), 4), **out}


def cmd_solve(args) -> int:
    started = time.time()
    name, cfg = load_config(args.config)
    run_dir = M1 / name
    run = {**RUN_DEFAULTS, **cfg.get("run", {})}
    for key in ("budget_hours", "checkpoint_hours", "log_minutes"):
        if getattr(args, key) is not None:
            run[key] = getattr(args, key)
    if args.batch is not None:
        cfg["solver"] = {**cfg["solver"], "batch": args.batch}
    folder = run_dir / ("solve" if args.tag is None else f"solve-{args.tag}")
    folder.mkdir(parents=True, exist_ok=True)
    log = Log(folder / "solve.log")
    git, env = git_info(), environment()
    deadline = started + run["budget_hours"] * 3600
    end = job_end_time()
    log(f"solve {name} into {folder.name}: {cfg['solver']}, run {run}, git {git['git_hash'][:10]}"
        f"{' dirty' if git['git_dirty'] else ''}, {env['host']} ({env['cpu_model']}), job {env['slurm_job']}, "
        f"{env['slurm_cpus']} CPUs, {env['numba_threads']} threads")
    if end is not None:
        log(f"budget ends {time.strftime('%H:%M:%S', time.localtime(deadline))}, the job's walltime "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}: {(end - deadline) / 60:.0f} min apart")
        if deadline > end - MARGIN_MINUTES * 60:
            raise SystemExit(f"the time budget must end at least {MARGIN_MINUTES} min before the walltime")
    else:
        log("not in a Slurm job: the walltime margin isn't checked")
    if not (run_dir / "config.yaml").exists() and cfg["buckets"]["method"] != "lossless":
        raise SystemExit(f"{run_dir} has no fit: run fit first")
    keep_config(run_dir, cfg)
    found = checkpoints(folder)
    if args.resume and not found:
        raise SystemExit(f"--resume: no checkpoint in {folder}")
    if not args.resume and found:
        raise SystemExit(f"{folder} already has checkpoints: --resume continues them, --tag starts another run")
    game, pt, river, abstraction = build(cfg, run_dir)
    solver = make_solver(cfg, pt, river, abstraction)
    if args.resume:
        solver.load(found[-1][1])
        s = cfg["solver"]
        if (solver.batch, solver.samples, solver.seed) != (s["batch"], s["samples"], s["seed"]):
            raise SystemExit(f"{found[-1][1].name} has batch, samples, seed {solver.batch, solver.samples, solver.seed}, "
                             f"not the config's {s['batch'], s['samples'], s['seed']}")
        log(f"resumed from {found[-1][1].name}: t {solver.t}, {solver.solve_seconds / 3600:.3f} h of solving")
    record = {"event": "start", "time": now(), "resume": args.resume, "t": solver.t, "iterations": args.iterations,
              "solver": cfg["solver"], "run": run, **git, **env, "setup_seconds": time.time() - started}
    with open(folder / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    log(f"setup took {record['setup_seconds']:.0f} s, peak {peak_gb():.2f} GB")

    t0, s0 = solver.t, solver.solve_seconds
    saved, last_saved_t = [], solver.t
    window = (solver.t, solver.solve_seconds)
    next_log = time.time() + run["log_minutes"] * 60
    next_checkpoint = time.time() + run["checkpoint_hours"] * 3600
    first = last = 0.0

    def checkpoint() -> None:
        nonlocal last_saved_t
        avg = solver.average_strategy()
        health = health_check(solver, avg, last_saved_t)
        log(f"health check at t {solver.t}: {'PASS' if health['passed'] else 'FAIL'} {health}")
        if not health["passed"]:
            raise SystemExit("health check failed: stopping without saving")
        path = folder / f"ckpt_t{solver.t:09d}.npz"
        start = time.perf_counter()
        solver.save(path)
        info = {"checkpoint": path.name, "t": solver.t, "solve_seconds": solver.solve_seconds,
                "save_seconds": time.perf_counter() - start, "bytes": path.stat().st_size, "saved": now(),
                "health": health, "value": solver.value, "batch": solver.batch, "samples": solver.samples,
                "seed": solver.seed, **git, "host": env["host"], "slurm_job": env["slurm_job"]}
        if run["eval_in_process"]:
            info["in_process"] = exact_exploitability(pt, river, abstraction, avg, chunk=run["eval_chunk"])
            log(f"in-process exact check: {info['in_process']}")
        info["flop_p0_bet"] = flop_report(game, pt, avg, run["flop_hands"])
        del avg
        write_json(path.with_suffix(".json"), info)
        log(f"saved {path.name}: {info['bytes'] / 2**30:.2f} GiB in {info['save_seconds']:.1f} s; "
            f"player 0's flop bet frequency {info['flop_p0_bet']}")
        saved.append(info)
        last_saved_t = solver.t
        retain(folder, run["milestone_hours"], log)

    while (args.iterations is None or solver.t < args.iterations) and time.time() + 1.5 * last < deadline:
        before = solver.solve_seconds
        solver.iteration()
        last = solver.solve_seconds - before
        if solver.t == t0 + 1:
            first = last
            log(f"first iteration: {first:.2f} s (includes compiling), peak {peak_gb():.2f} GB")
        if time.time() >= next_log:
            n = solver.t - window[0]
            log(f"t {solver.t} | {(solver.solve_seconds - window[1]) / n:.3f} s/it | elapsed "
                f"{(time.time() - started) / 3600:.2f} h | solving {solver.solve_seconds / 3600:.3f} h | "
                f"peak {peak_gb():.2f} GB")
            window, next_log = (solver.t, solver.solve_seconds), time.time() + run["log_minutes"] * 60
        if time.time() >= next_checkpoint:
            checkpoint()
            next_checkpoint = time.time() + run["checkpoint_hours"] * 3600
    if solver.t > last_saved_t or not saved:
        checkpoint()
    peak = peak_gb()
    n = solver.t - t0
    per_it = (solver.solve_seconds - s0 - first) / (n - 1) if n > 1 else float("nan")
    summary = {"event": "end", "time": now(), "t": solver.t, "iterations": n, "first_iteration_seconds": first,
               "seconds_per_iteration": per_it, "river_deals_per_second": solver.batch * solver.samples / per_it,
               "solve_hours": solver.solve_seconds / 3600, "peak_gb": peak,
               "checkpoints": [{k: c[k] for k in ("checkpoint", "t", "bytes", "save_seconds")} for c in saved]}
    log(f"stopped at t {solver.t}: {n} iterations here, {per_it:.3f} s/it after the first, "
        f"{summary['river_deals_per_second']:.1f} river deals/s, peak {peak:.2f} GB")
    if args.resume_check:
        summary["resume_check"] = resume_check(cfg, pt, river, abstraction, solver, folder, args.resume_check, log)
    with open(folder / "runs.jsonl", "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    return 0 if summary.get("resume_check", {}).get("passed", True) else 1


def resume_check(cfg, pt, river, abstraction, solver, folder: Path, n: int, log) -> dict:
    """From the newest checkpoint, which holds the live solver's state: n more iterations
    uninterrupted (the live solver) and n more resumed (a second solver loaded from the
    file). Their regrets, averages, strategies and generators must be bit-identical."""
    t, path = checkpoints(folder)[-1]
    if t != solver.t:
        raise SystemExit(f"resume check: the newest checkpoint is t {t}, the solver t {solver.t}")
    twin = make_solver(cfg, pt, river, abstraction)
    twin.load(path)
    for _ in range(n):
        solver.iteration()
        twin.iteration()
    same = {name: all(same_bits(a, b) for ra, rb in zip(getattr(solver, name), getattr(twin, name))
                      for a, b in zip(ra, rb)) for name in ("regret", "cum", "sigma")}
    same["t"] = solver.t == twin.t
    same["generator"] = solver.rng.bit_generator.state == twin.rng.bit_generator.state
    out = {"from": path.name, "iterations": n, "same": same, "passed": all(same.values()), "peak_gb": peak_gb()}
    log(f"resume check from {path.name}, {n} iterations each: {'PASS' if out['passed'] else 'FAIL'} {same}, "
        f"peak with both solvers {out['peak_gb']:.2f} GB")
    info = json.loads(path.with_suffix(".json").read_text())
    write_json(path.with_suffix(".json"), {**info, "resume_check": out})
    return out


# Evaluating ----------------------------------------------------------------------------------


class Stop(Exception):
    pass


class Progress:
    """exact_exploitability's progress callback: logs chunks done and the ETA every 2% or
    every 2 minutes, and raises Stop after max_minutes (a timing run)."""

    def __init__(self, log, max_minutes: float | None = None):
        self.log, self.max_minutes = log, max_minutes
        self.start = time.perf_counter()
        self.first = None  # (chunks, time) after the first chunk, which includes compiling
        self.next_percent, self.next_time = 2.0, self.start + 120
        self.done = self.total = 0

    def seconds_per_chunk(self) -> float | None:
        if self.first is None or self.done <= self.first[0]:
            return None
        return (time.perf_counter() - self.first[1]) / (self.done - self.first[0])

    def projected_seconds(self) -> float | None:
        rate = self.seconds_per_chunk()
        return None if rate is None else (self.first[1] - self.start) + rate * (self.total - self.first[0])

    def __call__(self, done: int, total: int) -> None:
        self.done, self.total = done, total
        if self.first is None:
            self.first = (done, time.perf_counter())
        now_ = time.perf_counter()
        if 100 * done / total >= self.next_percent or now_ >= self.next_time or done == total:
            total_s = self.projected_seconds()
            eta = "" if total_s is None else (f", {self.seconds_per_chunk():.2f} s per chunk, projected "
                                              f"{total_s / 3600:.2f} h in all, ETA {(total_s - (now_ - self.start)) / 3600:.2f} h")
            self.log(f"chunks {done}/{total} ({100 * done / total:.2f}%), {(now_ - self.start) / 60:.1f} min{eta}")
            while self.next_percent <= 100 * done / total:
                self.next_percent += 2.0
            self.next_time = now_ + 120
        if self.max_minutes is not None and now_ - self.start >= 60 * self.max_minutes and done < total:
            raise Stop


def cmd_evaluate(args) -> int:
    ckpt = Path(args.checkpoint).resolve()
    folder = ckpt.parent
    run_dir = folder.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    run = {**RUN_DEFAULTS, **cfg.get("run", {})}
    chunk = args.chunk or run["eval_chunk"]
    log = Log(folder / f"{ckpt.stem}.eval.log")
    git, env = git_info(), environment()
    start = time.perf_counter()
    game, pt, river, abstraction = build(cfg, run_dir)
    solver = make_solver(cfg, pt, river, abstraction)
    solver.load(ckpt)
    t, solve_seconds = solver.t, solver.solve_seconds
    avg = solver.average_strategy()
    del solver
    setup = time.perf_counter() - start
    total_deals = pt.streets[-1].num_deals * river.num_children
    log(f"evaluate {ckpt.name} (t {t}, {solve_seconds / 3600:.3f} h of solving): {total_deals:,} river deals in "
        f"chunks of {chunk}, git {git['git_hash'][:10]}{' dirty' if git['git_dirty'] else ''}, {env['host']} "
        f"({env['cpu_model']}), {env['slurm_cpus']} CPUs, {env['numba_threads']} threads; setup {setup:.0f} s, "
        f"peak {peak_gb():.2f} GB")
    progress = Progress(log, args.max_minutes)
    base = {"checkpoint": ckpt.name, "t": t, "solve_seconds": solve_seconds, "chunk": chunk, "river_deals": total_deals,
            "setup_seconds": setup, **git, **env}
    try:
        result = exact_exploitability(pt, river, abstraction, avg, chunk=chunk, progress=progress)
    except Stop:
        projected = progress.projected_seconds()
        timing = {**base, "chunks_done": progress.done, "chunks": progress.total,
                  "elapsed_seconds": time.perf_counter() - progress.start,
                  "seconds_per_chunk": progress.seconds_per_chunk(),
                  "projected_hours": None if projected is None else projected / 3600, "peak_gb": peak_gb(),
                  "finished": now()}
        write_json(folder / f"{ckpt.stem}.timing.json", timing)
        log(f"stopped after {args.max_minutes} min, as asked: {progress.done}/{progress.total} chunks, projected "
            f"{timing['projected_hours']} h for the whole check, peak {timing['peak_gb']:.2f} GB")
        return 0
    out = {**result, "seconds": time.perf_counter() - progress.start, **base, "peak_gb": peak_gb(), "finished": now()}
    write_json(folder / f"{ckpt.stem}.eval.json", out)
    log(f"exploitability {result['exploitability']!r}, BR0 {result['br_value_p0']!r}, BR1 {result['br_value_p1']!r}, "
        f"value {result['game_value']!r}, {out['seconds'] / 3600:.3f} h, peak {out['peak_gb']:.2f} GB")
    return 0


def cmd_compare(args) -> int:
    """Whether two checkpoints hold bit-identical solver states: every array, t, the
    generator, batch, samples and seed (solve time aside)."""
    with np.load(args.a) as za, np.load(args.b) as zb:
        if set(za.files) != set(zb.files):
            print(f"different contents: {sorted(set(za.files) ^ set(zb.files))}")
            return 1
        differ = [key for key in sorted(za.files) if key != "solve_seconds" and not same_bits(za[key], zb[key])]
        t = (int(za["t"]), int(zb["t"]))
    print(f"{Path(args.a).name} vs {Path(args.b).name} (t {t[0]} and {t[1]}): "
          f"{'bit-identical' if not differ else f'{len(differ)} entries differ: {differ[:10]}'}")
    return 0 if not differ else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("fit", "check-fit"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--force", action="store_true", help="fit: refit when buckets.npz exists")
        p.add_argument("--full", action="store_true", help="check-fit: also compare with one call over every parent")
        p.add_argument("--seed", type=int, default=1, help="check-fit: which river deals the id checks use")
    p = sub.add_parser("solve")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    p.add_argument("--tag", help="solve into solve-<tag>/ instead of solve/")
    p.add_argument("--budget-hours", type=float, help="override run.budget_hours")
    p.add_argument("--checkpoint-hours", type=float, help="override run.checkpoint_hours")
    p.add_argument("--log-minutes", type=float, help="override run.log_minutes")
    p.add_argument("--batch", type=int, help="override solver.batch")
    p.add_argument("--iterations", type=int, help="also stop when t reaches this")
    p.add_argument("--resume-check", type=int, default=0, metavar="N",
                   help="at the end, N more iterations uninterrupted and resumed from the last checkpoint")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chunk", type=int, help="river deals per chunk (default: the config's run.eval_chunk)")
    p.add_argument("--max-minutes", type=float, help="stop after this long and record the projected time")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args(argv)
    commands = {"fit": lambda a: cmd_fit(a) or cmd_check_fit(a), "check-fit": cmd_check_fit, "solve": cmd_solve,
                "evaluate": cmd_evaluate, "compare": cmd_compare}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
