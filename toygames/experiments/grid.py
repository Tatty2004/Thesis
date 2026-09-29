"""The bucketing experiment.

For every game (N, deck mode) and every (method, k, seed):
  1. bucket each public state, build the bucketed game and solve it with DCFR until
     its exploitability inside the bucketed game is <= tol. That is the solver error.
  2. map the strategy back to the full game and compute its exact exploitability.
     That is the total error.
The unabstracted game is solved the same way ("full"), and `lossless` must match it
exactly. Outputs in results/<config hash>/: runs.csv, table.md, table.csv and
total_error_vs_k.{png,pdf}. Error is not assumed to fall as k grows (Waugh et al.
2009); table.md logs every seed where it doesn't.
"""
from __future__ import annotations

import csv
import hashlib
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from toygames.abstraction.abstract_game import build_abstract_game
from toygames.abstraction.bucketing import make_bucketing
from toygames.abstraction.features import compute_features
from toygames.eval.exploitability import exploitability
from toygames.experiments.run import (CACHE, build_game, cached_result, config_hash, expand, finish_run,
                                      game_cfg, markdown_table, run_parallel, seed_everything, start_run, tree_for)
from toygames.solvers.cfr import CFR


_LAST_FEATURES: dict = {}


def features_for(gcfg: dict):
    """Equity features of the game, cached on disk and (the last one) in memory."""
    game = build_game(gcfg)
    if _LAST_FEATURES.get("name") == game.name:
        return _LAST_FEATURES["features"]
    path = CACHE / "features" / f"{hashlib.sha1(game.name.encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            feats = pickle.load(f)
    else:
        feats = compute_features(game)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    _LAST_FEATURES.update(name=game.name, features=feats)
    return feats


def grid_one(job) -> dict:
    run, force, git = job
    cached = cached_result(run, force, git)
    if cached is not None:
        return cached
    seed_everything(run["seed"])
    run_dir = start_run(run, git)
    full = tree_for(run["game"])
    start = time.perf_counter()
    bucket_stats = {}
    if run["method"] == "full":
        tree, ag = full, None
    else:
        feats = features_for(run["game"])
        bucketing = make_bucketing(feats, run["method"], run["k"], run["seed"], run.get("restarts", 1))
        ag = build_abstract_game(full, bucketing)
        tree = ag.tree
        for t in range(feats.num_streets):
            counts = bucketing.bucket_counts(t)
            bucket_stats[f"street{t + 1}_mean_buckets"] = float(counts.mean())
    build_seconds = time.perf_counter() - start
    s = run["solver"]
    solver = CFR(tree, s["variant"], s["alpha"], s["beta"], s["gamma"])
    rows = solver.run(tol=s["tol"], max_iterations=s["max_iterations"], log_every=s["log_every"],
                      log_path=run_dir / "log.csv")
    inside = total = rows[-1]
    if ag is not None:
        rep = exploitability(full, ag.lift(solver.average_strategy()))
        total = {"exploitability": rep.exploitability, "game_value": rep.game_value,
                 "br_value_p0": rep.br_value[0], "br_value_p1": rep.br_value[1]}
    result = {
        "game": full.name, "num_ranks": run["game"]["num_ranks"], "deck_mode": run["game"]["deck_mode"],
        "method": run["method"], "k": run["k"], "seed": run["seed"],
        "full_infosets": sum(full.n_infosets), "abstract_infosets": sum(tree.n_infosets),
        "abstract_terminals": tree.n_terminals, **bucket_stats,
        "iterations": solver.t, "converged": inside["exploitability"] <= s["tol"],
        "build_seconds": build_seconds, "solve_seconds": solver.solve_seconds,
        "solver_error": inside["exploitability"], "abstract_value": inside["game_value"],
        "total_error": total["exploitability"], "full_value": total["game_value"],
        "full_br_value_p0": total["br_value_p0"], "full_br_value_p1": total["br_value_p1"],
    }
    return finish_run(run_dir, result)


def grid_jobs(cfg: dict, force: bool, git: dict) -> list:
    jobs = []
    combos = sorted(expand(cfg["games"]), key=lambda c: (c["num_ranks"], c["deck_mode"]))
    for combo in combos:
        gcfg = game_cfg(cfg["game"], combo)
        base = {"experiment": "grid", "game": gcfg, "solver": cfg["solver"]}
        jobs.append(({**base, "method": "full", "k": None, "seed": cfg["seed"]}, force, git))
        jobs.append(({**base, "method": "lossless", "k": None, "seed": cfg["seed"]}, force, git))
        for method in cfg["methods"]:
            for k in cfg["ks"]:
                if method == "product" and int(np.sqrt(k)) ** 2 != k:
                    continue
                for seed in cfg["seeds"]:
                    run = {**base, "method": method, "k": k, "seed": seed}
                    if "bucketing" in cfg:
                        run["restarts"] = cfg["bucketing"]["restarts"]
                    jobs.append((run, force, git))
    return jobs


def run_grid(cfg: dict, out: Path, workers: int, force: bool, git: dict) -> None:
    jobs = grid_jobs(cfg, force, git)
    for combo in expand(cfg["games"]):  # compile and featurize once, up front
        gcfg = game_cfg(cfg["game"], combo)
        tree_for(gcfg)
        features_for(gcfg)
    print(f"{len(jobs)} runs", flush=True)
    results = run_parallel(grid_one, jobs, workers)
    for (run, _, _), res in zip(jobs, results):
        res["run"] = config_hash(run)
    write_outputs(cfg, results, out)


# Summaries ------------------------------------------------------------------------------

METHOD_ORDER = ["avg_1d", "product", "kmeans_2d", "emd_2d"]


def _stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0, "min": v.min(), "max": v.max(), "n": len(v)}


def write_outputs(cfg: dict, results: list[dict], out: Path) -> None:
    fields = list(results[0].keys())
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}, key=lambda k: (
            fields.index(k) if k in fields else len(fields), k)))
        w.writeheader()
        w.writerows(results)

    games = sorted({(r["num_ranks"], r["deck_mode"]) for r in results}, key=lambda g: (g[0], g[1] != "shared"))
    by = defaultdict(list)
    for r in results:
        by[(r["num_ranks"], r["deck_mode"], r["method"], r["k"])].append(r)

    # Main table: one row per (N, deck, method, k), statistics over seeds.
    rows = []
    for n, deck in games:
        for method in ["full", "lossless"] + METHOD_ORDER:
            for k in sorted({r["k"] for r in results if r["method"] == method and r["k"] is not None}) or [None]:
                runs = by.get((n, deck, method, k))
                if not runs:
                    continue
                te, se = _stats([r["total_error"] for r in runs]), _stats([r["solver_error"] for r in runs])
                rows.append({"N": n, "deck": deck, "method": method, "k": "" if k is None else k, "seeds": te["n"],
                             "total_mean": te["mean"], "total_sd": te["sd"], "total_min": te["min"],
                             "total_max": te["max"], "solver_mean": se["mean"], "solver_max": se["max"],
                             "abstract_infosets": np.mean([r["abstract_infosets"] for r in runs]),
                             "full_infosets": runs[0]["full_infosets"],
                             "iterations": np.mean([r["iterations"] for r in runs]),
                             "all_converged": all(r["converged"] for r in runs),
                             "seconds": np.mean([r["solve_seconds"] for r in runs])})
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fmt = {"total_mean": "{:.4f}", "total_sd": "{:.4f}", "total_min": "{:.4f}", "total_max": "{:.4f}",
           "solver_mean": "{:.1e}", "solver_max": "{:.1e}", "abstract_infosets": "{:,.0f}",
           "full_infosets": "{:,}", "iterations": "{:.0f}", "seconds": "{:.1f}"}
    cols = ["N", "deck", "method", "k", "seeds", "total_mean", "total_sd", "total_min", "total_max",
            "solver_mean", "solver_max", "abstract_infosets", "full_infosets", "iterations", "all_converged",
            "seconds"]
    lines = [f"# {cfg.get('title', 'Bucketing experiment')}", "",
             f"Total error = exact full-game exploitability of the mapped-back strategy; solver error = "
             f"exploitability inside the bucketed game (DCFR stopped at tol = {cfg['solver']['tol']:g}). "
             f"Chips per hand; statistics over k-means seeds {cfg['seeds']}.", "",
             markdown_table(rows, cols, fmt), ""]

    # Lossless must match the unabstracted game exactly.
    lines += ["## Lossless check", ""]
    for n, deck in games:
        full, loss = by[(n, deck, "full", None)][0], by[(n, deck, "lossless", None)][0]
        same = all(full[x] == loss[x] for x in ("iterations", "solver_error", "total_error", "full_value",
                                                  "abstract_infosets"))
        lines.append(f"- N={n} {deck}: full {full['total_error']:.6e} after {full['iterations']} its, lossless "
                     f"{loss['total_error']:.6e} after {loss['iterations']} its -> "
                     f"{'identical' if same else 'MISMATCH'}")
    lines.append("")

    # Invariants every run must satisfy, whatever the bucketing.
    lines += ["## Sanity checks", ""]
    for n, deck in games:
        rs = [r for r in results if (r["num_ranks"], r["deck_mode"]) == (n, deck)]
        lo = max(-r["full_br_value_p1"] for r in rs)  # every profile brackets the game value
        hi = min(r["full_br_value_p0"] for r in rs)
        lines.append(
            f"- N={n} {deck}: {sum(r['converged'] for r in rs)}/{len(rs)} reached tol; "
            f"total >= solver error in {sum(r['total_error'] >= r['solver_error'] - 1e-12 for r in rs)}/{len(rs)}; "
            f"max |full-game value - bucketed value| {max(abs(r['full_value'] - r['abstract_value']) for r in rs):.1e}; "
            f"game value in [{lo:+.6f}, {hi:+.6f}] (consistent: {lo <= hi + 1e-12})")
    lines.append("")

    # The questions: kmeans_2d vs avg_1d, emd_2d vs kmeans_2d, and shared vs independent.
    lines += ["## Comparisons (mean total error over seeds)", ""]
    comp = []
    for n, deck in games:
        for k in cfg["ks"]:
            m = {meth: np.mean([r["total_error"] for r in by[(n, deck, meth, k)]]) for meth in METHOD_ORDER
                 if by.get((n, deck, meth, k))}
            comp.append({"N": n, "deck": deck, "k": k, **{meth: m.get(meth) for meth in METHOD_ORDER},
                         "kmeans_2d - avg_1d": m["kmeans_2d"] - m["avg_1d"],
                         "emd_2d - kmeans_2d": m["emd_2d"] - m["kmeans_2d"]})
    cfmt = {meth: "{:.4f}" for meth in METHOD_ORDER}
    cfmt.update({"kmeans_2d - avg_1d": "{:+.4f}", "emd_2d - kmeans_2d": "{:+.4f}"})
    lines += [markdown_table(comp, ["N", "deck", "k"] + METHOD_ORDER + ["kmeans_2d - avg_1d", "emd_2d - kmeans_2d"],
                             cfmt), ""]
    lines += ["Negative differences favour the 2D method. Per-seed wins (seed i of one method vs seed i of the "
              "other; seeds only set k-means initialisation, so this is a rough count):", ""]
    for a, b in (("kmeans_2d", "avg_1d"), ("emd_2d", "kmeans_2d")):
        tally = {"lower": 0, "tied": 0, "higher": 0}
        for n, deck in games:
            for k in cfg["ks"]:
                ra, rb = by.get((n, deck, a, k), []), by.get((n, deck, b, k), [])
                for x, y in zip(sorted(ra, key=lambda r: r["seed"]), sorted(rb, key=lambda r: r["seed"])):
                    d = x["total_error"] - y["total_error"]
                    tally["lower" if d < -1e-9 else "higher" if d > 1e-9 else "tied"] += 1
        lines.append(f"- {a} vs {b}: lower total error in {tally['lower']}, tied in {tally['tied']}, "
                     f"higher in {tally['higher']} of {sum(tally.values())} (N, deck, k, seed) cases")
    lines.append("")

    # Monotonicity in k (logged, not asserted).
    lines += ["## Does total error fall as k grows? (logged, not asserted)", ""]
    violations = []
    for n, deck in games:
        for method in ("avg_1d", "kmeans_2d", "emd_2d"):
            for seed in cfg["seeds"]:
                errs = [next((r["total_error"] for r in by.get((n, deck, method, k), []) if r["seed"] == seed), None)
                        for k in cfg["ks"]]
                for k1, k2, e1, e2 in zip(cfg["ks"], cfg["ks"][1:], errs, errs[1:]):
                    if e1 is not None and e2 is not None and e2 > e1 + 1e-9:
                        violations.append(f"- N={n} {deck} {method} seed {seed}: k={k1} {e1:.4f} -> k={k2} {e2:.4f}")
    lines += violations or ["- none: every method and seed has non-increasing total error in k"]
    lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    plot(cfg, results, games, by, out)
    print("\n".join(lines))


def plot(cfg, results, games, by, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"avg_1d": "#2a78d6", "product": "#eb6834", "kmeans_2d": "#1baf7a", "emd_2d": "#eda100"}
    markers = {"avg_1d": "o", "product": "s", "kmeans_2d": "^", "emd_2d": "D"}
    labels = {"avg_1d": "avg_1d (baseline)", "product": "product", "kmeans_2d": "kmeans_2d", "emd_2d": "emd_2d"}
    ns = sorted({n for n, _ in games})
    decks = [d for d in ("shared", "independent") if any(g[1] == d for g in games)]
    fig, axes = plt.subplots(len(decks), len(ns), figsize=(max(3.6 * len(ns) + 0.8, 10.0), 3.3 * len(decks) + 1.4),
                             sharex=True, sharey=True, squeeze=False, facecolor=surface)
    ks = cfg["ks"]
    offsets = {"avg_1d": -0.09, "product": 0.0, "kmeans_2d": 0.0, "emd_2d": 0.09}
    for i, deck in enumerate(decks):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            ax.set_facecolor(surface)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.grid(axis="y", which="major", color=grid, linewidth=0.8)
            ax.tick_params(which="both", colors=ink2, labelsize=9, length=0)
            floor = by.get((n, deck, "lossless", None))
            if floor:
                ax.axhline(floor[0]["total_error"], color=ink2, linewidth=1.0, zorder=1)
            for method in METHOD_ORDER:
                xs, mean, lo, hi = [], [], [], []
                for k in ks:
                    runs = by.get((n, deck, method, k))
                    if runs:
                        e = [r["total_error"] for r in runs]
                        xs.append(k + offsets[method])
                        mean.append(np.mean(e))
                        lo.append(min(e))
                        hi.append(max(e))
                if not xs:
                    continue
                c = colors[method]
                if len(xs) > 1:
                    ax.fill_between(xs, lo, hi, color=c, alpha=0.12, linewidth=0, zorder=2)
                    ax.plot(xs, mean, color=c, linewidth=2, solid_capstyle="round", zorder=3)
                else:
                    ax.vlines(xs, lo, hi, color=c, linewidth=2, zorder=3)
                ax.plot(xs, mean, linestyle="none", marker=markers[method], markersize=7, color=c,
                        markeredgecolor=surface, markeredgewidth=1.5, zorder=4, label=labels[method])
            ax.set_yscale("log")
            ax.set_xticks(ks)
            if i == 0:
                ax.set_title(f"N = {n}", color=ink, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{deck} deck\ntotal error (chips/hand)", color=ink2, fontsize=9)
            if i == len(decks) - 1:
                ax.set_xlabel("buckets per public state (k)", color=ink2, fontsize=9)
    handles, labs = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=ink2, linewidth=1.0))
    labs.append("lossless (solver floor)")
    top = 1 - 0.95 / fig.get_figheight()
    fig.suptitle("Total error vs bucket count k", color=ink, fontsize=12.5, x=0.5, y=0.995)
    fig.text(0.5, 1 - 0.36 / fig.get_figheight(), "Exact full-game exploitability of the mapped-back strategy, "
             "chips per hand, log scale", ha="center", va="top", color=ink2, fontsize=9.5)
    fig.legend(handles, labs, loc="upper center", ncol=len(labs), frameon=False, fontsize=9,
               labelcolor=ink, bbox_to_anchor=(0.5, 1 - 0.6 / fig.get_figheight()))
    fig.text(0.5, 0.004, f"Line: mean over {len(cfg['seeds'])} k-means seeds. Band or bar: min to max over seeds. "
             f"Bucketed games solved with DCFR to exploitability {cfg['solver']['tol']:g}.",
             ha="center", va="bottom", color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, top))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"total_error_vs_k.{ext}", dpi=200, facecolor=surface)
    plt.close(fig)
