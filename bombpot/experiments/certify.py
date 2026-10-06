"""Check B: re-certify the full-game equilibria with methods that don't share our solver.

For every game in the sweep:
  1. Solve with DCFR to `tol` and save the average strategy (strategy_dcfr.npz, which
     loads back as {infoset key: action probabilities}); run CFR+ and vanilla CFR for a
     fixed budget; solve the sequence-form LP where num_ranks <= lp_max_ranks and save
     that strategy too (strategy_lp.npz).
  2. Best response as a linear program (HiGHS) against the DCFR strategy must match our
     backward-induction best response (num_ranks <= lp_br_max_ranks).
  3. Every profile brackets the game value: -BR1(sigma0) <= v* <= BR0(sigma1). The
     brackets of all solvers must overlap, contain the LP value, and contain each
     profile's own value.
  4. Dominated-action audit (eval/audit.py): folding hands that can't lose, calling with
     hands that can't win. Its reach-weighted bound must stay within each player's
     best-response gain.
  5. Swapping boards A and B throughout a strategy must leave its exploitability, best
     responses and value unchanged (the DCFR strategy and random ones).
  6. The saved strategy loads back bit for bit.
OpenSpiel's grading of the same games is in tests/test_equilibrium.py, since OpenSpiel
is only used in tests/. Outputs: certify.md and certify.json in the sweep's results
folder, and each game's strategies, logs and result.json in its own run folder.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.eval.audit import audit
from bombpot.core.eval.exploitability import exploitability
from bombpot.experiments.run import (cached_result, config_hash, expand, finish_run, game_cfg, markdown_table,
                                      run_parallel, seed_everything, start_run, tree_for)
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import best_response_lp, solve_game
from bombpot.core.tree import load_strategy, normalize, save_strategy

LP_BR_TOL = 1e-7
SWAP_TOL = 1e-12


def swap_key(key: str) -> str:
    """The same infoset with boards A and B exchanged. Keys are "player:hand:A|B:betting"
    (LimitPoker.infoset_key), with each board's ranks in deal order."""
    player, hand, boards, betting = key.split(":")
    return ":".join((player, hand, "|".join(reversed(boards.split("|"))), betting))


def swap_boards(tree, sigma) -> list[np.ndarray]:
    """The strategy that plays at each infoset what `sigma` plays at its board-swapped twin."""
    out = []
    for p in (0, 1):
        index = tree.infoset_index(p)
        twin = np.array([index[swap_key(k)] for k in tree.keys[p]], dtype=np.int64)
        na = tree.num_actions[p]
        assert all(tree.actions[p][i] == tree.actions[p][j] for i, j in enumerate(twin))
        offsets = np.arange(tree.n_seqs[p] - 1) - np.repeat(np.cumsum(na) - na, na)
        new = np.ones(tree.n_seqs[p])
        new[np.repeat(tree.first_seq[p], na) + offsets] = sigma[p][np.repeat(tree.first_seq[p][twin], na) + offsets]
        out.append(new)
    return out


def random_profile(tree, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [normalize(tree, p, np.concatenate(([1.0], rng.random(tree.n_seqs[p] - 1)))) for p in (0, 1)]


def report(rep) -> dict:
    return {"exploitability": rep.exploitability, "br_value_p0": rep.br_value[0], "br_value_p1": rep.br_value[1],
            "game_value": rep.game_value, "lower": -rep.br_value[1], "upper": rep.br_value[0]}


def certify_one(job) -> dict:
    run, force, git = job
    cached = cached_result(run, force, git)
    if cached is not None:
        return cached
    seed_everything(run["seed"])
    run_dir = start_run(run, git)
    tree = tree_for(run["game"])
    result = {"game": tree.name, "infosets": sum(tree.n_infosets), "terminals": tree.n_terminals}
    profiles = {}

    # 1. Solvers.
    for name, s in run["solvers"].items():
        solver = CFR(tree, s["variant"], s.get("alpha", 1.5), s.get("beta", 0.0), s.get("gamma", 2.0))
        log = run_dir / ("log.csv" if name == "dcfr" else f"log_{name}.csv")
        rows = solver.run(iterations=s.get("iterations"), tol=s.get("tol"), max_iterations=s.get("max_iterations", 100_000),
                          log_every=s["log_every"], log_path=log)
        profiles[name] = solver.average_strategy()
        result[name] = {"iterations": solver.t, "solve_seconds": solver.solve_seconds,
                        **{k: rows[-1][k] for k in ("exploitability", "br_value_p0", "br_value_p1", "game_value")},
                        "lower": -rows[-1]["br_value_p1"], "upper": rows[-1]["br_value_p0"]}
        if name == "dcfr":
            save_strategy(run_dir / "strategy_dcfr.npz", tree, profiles[name], solver="dcfr", iterations=solver.t,
                          exploitability=rows[-1]["exploitability"])
    if run["lp"]:
        start = time.perf_counter()
        (v0, v1), sigma = solve_game(tree)
        profiles["lp"] = sigma
        result["lp"] = {"value_p0_lp": v0, "value_p1_lp": v1, "seconds": time.perf_counter() - start,
                        **report(exploitability(tree, sigma))}
        save_strategy(run_dir / "strategy_lp.npz", tree, sigma, solver="lp", value=v0)

    # 2. Best response by LP against the DCFR strategy.
    dcfr = profiles["dcfr"]
    if run["lp_br"]:
        start = time.perf_counter()
        lp_br = [best_response_lp(tree, p, dcfr[1 - p])[0] for p in (0, 1)]
        ours = (result["dcfr"]["br_value_p0"], result["dcfr"]["br_value_p1"])
        diff = max(abs(a - b) for a, b in zip(lp_br, ours))
        result["lp_br"] = {"value_p0": lp_br[0], "value_p1": lp_br[1], "max_diff": diff,
                           "seconds": time.perf_counter() - start, "ok": diff <= LP_BR_TOL}

    # 3. Value brackets.
    names = [n for n in ("dcfr", "cfr+", "cfr", "lp") if n in result]
    lo = max(result[n]["lower"] for n in names)
    hi = min(result[n]["upper"] for n in names)
    own = all(result[n]["lower"] - 1e-12 <= result[n]["game_value"] <= result[n]["upper"] + 1e-12 for n in names)
    brackets = {"lower": lo, "upper": hi, "width": hi - lo, "overlap": lo <= hi + 1e-9, "own_value_inside": own}
    if "lp" in result:
        v = result["lp"]["value_p0_lp"]
        brackets["lp_value_inside_all"] = all(result[n]["lower"] - 1e-9 <= v <= result[n]["upper"] + 1e-9
                                              for n in names)
        brackets["lp_values_agree"] = abs(result["lp"]["value_p0_lp"] - result["lp"]["value_p1_lp"]) <= 1e-6
    result["brackets"] = {**brackets, "ok": all(v for k, v in brackets.items() if isinstance(v, bool))}

    # 4. Dominated actions.
    result["audit"] = {}
    for name in ("dcfr", "lp"):
        if name in profiles:
            r = result[name]
            result["audit"][name] = audit(tree, profiles[name], (r["br_value_p0"], r["br_value_p1"]), r["game_value"])
    result["audit"]["ok"] = all(a["ok"] for a in result["audit"].values())

    # 5. Board swap.
    diffs = {}
    tests = {"dcfr": dcfr, **{f"random{i}": random_profile(tree, run["seed"] + i) for i in range(run["random_profiles"])}}
    for name, sigma in tests.items():
        a, b = exploitability(tree, sigma), exploitability(tree, swap_boards(tree, sigma))
        diffs[name] = max(abs(a.exploitability - b.exploitability), abs(a.game_value - b.game_value),
                          *(abs(x - y) for x, y in zip(a.br_value, b.br_value)))
    result["swap"] = {"max_diff": max(diffs.values()), "diffs": diffs, "ok": max(diffs.values()) <= SWAP_TOL}

    # 6. Saved strategy round trip.
    saved = load_strategy(run_dir / "strategy_dcfr.npz").sigma(tree)
    result["roundtrip"] = {"ok": all(np.array_equal(x, y) for x, y in zip(saved, dcfr))}

    checks = ["brackets", "audit", "swap", "roundtrip"] + (["lp_br"] if "lp_br" in result else [])
    result["ok"] = all(result[c]["ok"] for c in checks)
    return finish_run(run_dir, result)


def run_certify(cfg: dict, out: Path, workers: int, force: bool, git: dict) -> None:
    jobs = []
    for combo in expand(cfg.get("sweep", {})):
        gcfg = game_cfg(cfg["game"], combo)
        n = gcfg.get("num_ranks", 0)
        run = {"experiment": "certify", "seed": cfg["seed"], "game": gcfg, "solvers": cfg["solvers"],
               "lp": n <= cfg.get("lp_max_ranks", 0), "lp_br": n <= cfg.get("lp_br_max_ranks", 0),
               "random_profiles": cfg.get("random_profiles", 2)}
        jobs.append((run, force, git))
    for run, _, _ in jobs:  # compile once, up front, so workers only load the cache
        tree_for(run["game"])
    results = run_parallel(certify_one, jobs, workers)
    rows = []
    for (run, _, _), r in zip(jobs, results):
        a = r["audit"]["dcfr"]
        rows.append({
            "N": run["game"]["num_ranks"], "deck": run["game"]["deck_mode"], "infosets": r["infosets"],
            "dcfr_its": r["dcfr"]["iterations"], "dcfr_expl": r["dcfr"]["exploitability"],
            "cfr+_expl": r["cfr+"]["exploitability"], "cfr_expl": r["cfr"]["exploitability"],
            "lower": r["brackets"]["lower"], "width": r["brackets"]["width"],
            "lp_value": r.get("lp", {}).get("value_p0_lp"),
            "lp_br_diff": r.get("lp_br", {}).get("max_diff"),
            "dominated": a["p0"]["dominated"] + a["p1"]["dominated"],
            "mistake_rate": max(a["p0"]["mistake_rate"], a["p1"]["mistake_rate"]),
            "audit_ratio": max(a[q]["bound_sum"] / a[q]["br_gain"] if a[q]["br_gain"] > 0 else 0.0
                               for q in ("p0", "p1")),
            "swap_diff": r["swap"]["max_diff"], "ok": r["ok"], "run": config_hash(run)})
    cols = ["N", "deck", "infosets", "dcfr_its", "dcfr_expl", "cfr+_expl", "cfr_expl", "lower", "width", "lp_value",
            "lp_br_diff", "dominated", "mistake_rate", "audit_ratio", "swap_diff", "ok", "run"]
    fmt = {"infosets": "{:,}", "dcfr_expl": "{:.2e}", "cfr+_expl": "{:.2e}", "cfr_expl": "{:.2e}",
           "lower": "{:+.8f}", "width": "{:.1e}", "lp_value": "{:+.8f}", "lp_br_diff": "{:.1e}",
           "dominated": "{:,}", "mistake_rate": "{:.1e}", "audit_ratio": "{:.1e}", "swap_diff": "{:.1e}"}
    table = markdown_table(rows, cols, fmt)
    lp_inside = [r["brackets"]["lp_value_inside_all"] for r in results if "lp_value_inside_all" in r["brackets"]]
    notes = [
        "- `lower`, `width`: the tightest bracket on player 0's game value over all solvers (DCFR, CFR+, "
        "CFR and the LP where it runs), from max of -BR1(sigma0) to min of BR0(sigma1). Brackets overlap "
        f"in every game: {all(r['brackets']['overlap'] for r in results)}. The LP value lies inside every "
        f"solver's bracket in all {len(lp_inside)} games where the LP runs: {all(lp_inside)}.",
        f"- `lp_br_diff`: largest gap between the LP best response and ours against the DCFR strategy "
        f"(must be <= {LP_BR_TOL:g}).",
        "- `dominated`: (infoset, action) pairs dominated by another action that also ends the player's "
        "part in the hand (here always: folding a hand that can't lose). `mistake_rate`: how often the DCFR "
        "strategy plays such an action at these spots, weighted by how often each spot comes up (larger "
        "player). `audit_ratio`: the larger of the two players' reach-weighted bound divided by that "
        "player's best-response gain (must be <= 1).",
        f"- `swap_diff`: largest change in exploitability, best responses or value after swapping boards A "
        f"and B, over the DCFR strategy and {cfg.get('random_profiles', 2)} random ones (must be <= {SWAP_TOL:g}).",
        f"- All {len(rows)} games pass every check: {all(r['ok'] for r in rows)}.",
    ]
    text = f"# {cfg.get('title', 'certify')}\n\n{table}\n\n" + "\n".join(notes) + "\n"
    (out / "certify.md").write_text(text)
    (out / "certify.json").write_text(json.dumps(results, indent=2))
    print(text)
