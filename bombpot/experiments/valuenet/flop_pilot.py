"""Analysis of the flop-end sample cost pilot (flop_datagen.py): label error of cheap variants
against the full solve, and the cost of each.

    python -m bombpot.experiments.valuenet.flop_pilot run --model M --flops F --equity E --seed S \
        --out results/valuenet/flopnet_pilot/runs      (one GPU job per state: everything below)
    python -m bombpot.experiments.valuenet.flop_pilot summarize --json runs/*.json --out pilot.json
    python -m bombpot.experiments.valuenet.flop_pilot files --ref ...npz [--variants ...npz] --out X.json

`run` keeps the per-deal arrays in memory (scratch was full on 2026-11-22: about 100 MB per state
as files) and writes only <out>/<state>.json (the analysis) and a small <state>.npz (ranges and
the reference labels). In the same job, on the same state: the reference (every deal, alternating
DCFR, --iters), the check-down proxy, a simultaneous-update run on a fixed sample of --sim-deals
deals, and timing runs on K deals (--time-K, --time-iters iterations each).

Reference: every turn deal, the last check (e.g. 300 alternating DCFR iterations). Error metric =
train.py's hand_mae: sum over both players and hands of range x |label - reference| / (sum of
both ranges = 2), in pots.

Variants (computed from the reference's per-deal arrays; no extra solve, since the turn deals
are independent subgames):
  iters      every deal at an earlier check
  plain_K    K uniform random deals without replacement, (D / K) x their sum
  ratio_K    per hand: (deals the hand can be held at / of those in the sample) x the sample's sum
  cv_K       control variate: the check-down network value f_d at every deal, plus (D / K) x the
             sample's sum of (v_d - f_d)
each at every check and averaged over `--draws` random deal subsets (mean MAE, its RMS, and the
MAE of the subsets' mean label: ~0 for an unbiased estimator).

Variant files (runs with --deals K and/or --simultaneous on the same states) are compared on their
own deals with the reference's per-deal values there (plain estimator on both), which isolates
their iteration error from the sampling error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path) -> dict:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    out["info"] = json.loads(str(out["info"]))
    out["path"] = str(path)
    return out


def mae(a: np.ndarray, b: np.ndarray, r0: np.ndarray, r1: np.ndarray) -> float:
    """(2, H) labels: range-weighted mean |a - b| over both players (train.py hand_mae)."""
    d = np.abs(a.astype(np.float64) - b)
    return float((d[0] @ r0 + d[1] @ r1) / (r0.sum() + r1.sum()))


def valid_of(s: dict) -> np.ndarray:
    """(D, H) possible hands at every turn deal of the sample's flop pair."""
    from bombpot.experiments.valuenet.flop_datagen import FlopEnd

    flops = tuple(tuple(int(c) for c in f) for f in s["flops"])
    nr, ns = (int(x) for x in s["info"]["args"]["deck"].split(","))
    return FlopEnd(flops, nr, ns).turn.valid


def analyse_ref(s: dict, Ks, draws: int, seed: int) -> dict:
    r0, r1 = s["range0"].astype(np.float64), s["range1"].astype(np.float64)
    checks = [int(c) for c in s["checks"]]
    per_pair, D = s["info"]["per_pair"], s["info"]["num_deals"]
    ref = s["label_cfv"][-1].astype(np.float64)
    deal = s["deal_cfv"].astype(np.float64)  # (checks, 2, D, H), each deal on the label's scale x D... / per_pair
    out = {"path": s["path"], "flops": s["info"]["flops"], "line": int(s["line"]), "pot": float(s["pot"]),
           "checks": checks, "exploitability": [float(x) for x in s["exploitability"]],
           "seconds": [float(x) for x in s["seconds"]], "leaf_seconds": [float(x) for x in s["leaf_seconds"]],
           "leaf_rows": [int(x) for x in s["leaf_rows"]], "gpu": s["info"]["gpu"],
           "ref_mean_abs": mae(ref, np.zeros_like(ref), r0, r1),
           "br_minus_cfv": mae(s["label_br"][-1], ref, r0, r1)}
    out["iters"] = {c: mae(s["label_cfv"][i], ref, r0, r1) for i, c in enumerate(checks)}
    valid = valid_of(s)
    n_all = valid.sum(0)  # (H,)
    f = s["proxy"].astype(np.float64) if "proxy" in s else None
    if f is not None:
        out["proxy_mae"] = mae(f.sum(1) / per_pair, ref, r0, r1)  # the check-down values alone as the label
    rng = np.random.default_rng(seed)
    out["sampled"] = {}
    for K in Ks:
        subsets = [np.sort(rng.choice(D, K, replace=False)) for _ in range(draws)]
        for i, c in enumerate(checks):
            v = deal[i]
            ests = {"plain": [], "ratio": [], "cv": []}
            for S in subsets:
                part = v[:, S].sum(1)
                ests["plain"].append(part * (D / K) / per_pair)
                n_s = valid[S].sum(0)
                ests["ratio"].append(part * np.divide(n_all, n_s, out=np.zeros(len(n_all)), where=n_s > 0) / per_pair)
                if f is not None:
                    ests["cv"].append((f.sum(1) + (v[:, S] - f[:, S]).sum(1) * (D / K)) / per_pair)
            for name, xs in ests.items():
                if not xs:
                    continue
                errs = np.array([mae(x, ref, r0, r1) for x in xs])
                out["sampled"][f"{name}_K{K}_it{c}"] = {
                    "mae": float(errs.mean()), "mae_p90": float(np.quantile(errs, 0.9)),
                    "mae_of_mean": mae(np.mean(xs, axis=0), ref, r0, r1)}
    return out


def analyse_variant(v: dict, refs: dict) -> dict | None:
    key = (v["info"]["flops"], int(v["info"]["args"]["seed"]))
    s = refs.get(key)
    if s is None:
        return None
    r0, r1 = s["range0"].astype(np.float64), s["range1"].astype(np.float64)
    S = v["deals"]
    per_pair, D = s["info"]["per_pair"], s["info"]["num_deals"]
    assert np.array_equal(v["range0"], s["range0"]) and int(v["line"]) == int(s["line"]), "not the same state"
    ref_S = s["deal_cfv"][-1][:, S].astype(np.float64).sum(1) * (D / len(S)) / per_pair
    rec = {"path": v["path"], "deals": len(S), "simultaneous": bool(v["info"]["args"]["simultaneous"]),
           "gpu": v["info"]["gpu"], "checks": [int(c) for c in v["checks"]],
           "seconds": [float(x) for x in v["seconds"]], "leaf_seconds": [float(x) for x in v["leaf_seconds"]],
           "leaf_rows": [int(x) for x in v["leaf_rows"]], "setup_seconds": v["info"]["setup_seconds"],
           "equity_seconds": v["info"]["equity_seconds"], "exploitability": [float(x) for x in v["exploitability"]]}
    rec["vs_ref_same_deals"] = {int(c): mae(v["label_cfv"][i], ref_S, r0, r1) for i, c in enumerate(v["checks"])}
    if "deal_cfv" in v:  # per-deal agreement with the full solve at the same check (same algorithm: rounding only)
        sc = {int(c): i for i, c in enumerate(s["checks"])}
        rec["deal_max_diff_same_check"] = {int(c): float(np.abs(v["deal_cfv"][i].astype(np.float64)
                                                                - s["deal_cfv"][sc[int(c)]][:, S]).max())
                                           for i, c in enumerate(v["checks"]) if int(c) in sc}
    return rec


def as_record(rec: dict, state: dict, deals, info: dict, path: str) -> dict:
    """flop_datagen.solve_state's output in the layout of a saved sample (load())."""
    out = {k: np.asarray(v) for k, v in rec.items() if k != "setup_seconds" and len(v)}
    out.update(range0=state["range0"], range1=state["range1"], line=state["line"], pot=state["pot"],
               deals=np.asarray(deals), path=path, info={**info, "setup_seconds": rec["setup_seconds"]})
    return out


def run(argv=None):
    import time

    import torch

    from bombpot.experiments.valuenet import datagen as dg
    from bombpot.experiments.valuenet import flop_datagen as fd
    from bombpot.valuenet.model import load_model

    ap = argparse.ArgumentParser(description="the pilot on one state, in one job")
    ap.add_argument("--model", required=True)
    ap.add_argument("--flops", required=True)
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--equity", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--line", type=int, default=-1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--checks", default="10,25,50,100,200,300")
    ap.add_argument("--sim-deals", type=int, default=200)
    ap.add_argument("--time-K", default="25,50,100,200")
    ap.add_argument("--time-iters", type=int, default=50)
    ap.add_argument("--K", default="10,25,50,100,200,400")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = a.device
    nr, ns = (int(x) for x in a.deck.split(","))
    amp = not a.no_amp
    fe = fd.FlopEnd(fd.parse_flops(a.flops, nr, ns), nr, ns, np.load(a.equity))
    model = load_model(a.model, dev)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    print(f"{a.flops} seed {a.seed} on {gpu}", flush=True)
    rng = np.random.default_rng(a.seed)
    state = fd.sample_state(fe, rng, fe.strength(), a.line)
    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    name = f"{a.flops.replace(',', '_')}_s{a.seed}"
    base = {"flops": a.flops, "num_deals": fe.D, "per_pair": fe.per_pair, "gpu": gpu, "git": dg.git_hash(),
            "equity_seconds": 0.0}
    log = lambda m: print(m, flush=True)  # noqa: E731
    print(f"reference: line {state['line']} pot {state['pot']:g}, all {fe.D} deals, {a.iters} iterations", flush=True)
    deals = np.arange(fe.D)
    rec = fd.solve_state(fe, model, state, deals, a.iters, checks, device=dev, amp=amp, log=log)
    ref = as_record(rec, state, deals, {**base, "args": {"seed": a.seed, "simultaneous": False, "deck": a.deck}},
                    name + "_ref")
    ref["flops"] = np.array(fe.game.flops)
    ref["proxy"], proxy_seconds = fd.proxy_values(fe, model, state, deals, dev, amp=amp)
    print(f"proxy: {proxy_seconds:.1f}s", flush=True)
    variants = []
    S = np.sort(np.random.default_rng([7, a.seed]).choice(fe.D, a.sim_deals, replace=False))
    print(f"simultaneous on {len(S)} deals", flush=True)
    rec = fd.solve_state(fe, model, state, S, a.iters, checks, device=dev, amp=amp, simultaneous=True, log=log)
    variants.append(as_record(rec, state, S, {**base, "args": {"seed": a.seed, "simultaneous": True, "deck": a.deck}},
                              name + f"_sim{len(S)}"))
    for K in (int(x) for x in a.time_K.split(",")):
        T = np.sort(np.random.default_rng([11, a.seed, K]).choice(fe.D, K, replace=False))
        t0 = time.perf_counter()
        eq = fe.equity_all
        fe.equity_all, fe._eq = None, {}  # time the equities of a sample's deals too (no cache in production)
        fe.equity(T)
        eq_s = time.perf_counter() - t0
        fe.equity_all = eq
        print(f"alternating on {K} deals, {a.time_iters} iterations (equities {eq_s:.1f}s CPU)", flush=True)
        rec = fd.solve_state(fe, model, state, T, a.time_iters, [c for c in checks if c <= a.time_iters],
                             device=dev, amp=amp, log=log)
        variants.append(as_record(rec, state, T, {**base, "equity_seconds": eq_s,
                                                  "args": {"seed": a.seed, "simultaneous": False, "deck": a.deck}},
                                  name + f"_alt{K}"))
    refs = {(a.flops, a.seed): ref}
    result = {"ref": analyse_ref(ref, [int(x) for x in a.K.split(",")], a.draws, a.seed),
              "variants": [analyse_variant(v, refs) for v in variants], "proxy_seconds": proxy_seconds}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
    np.savez_compressed(out / f"{name}.npz", range0=state["range0"].astype(np.float32),
                        range1=state["range1"].astype(np.float32), line=state["line"], pot=state["pot"],
                        checks=np.array(checks), label_cfv=ref["label_cfv"], label_br=ref["label_br"])
    print(f"wrote {out / name}.json", flush=True)
    print("DONE", flush=True)


def summarize(argv=None):
    ap = argparse.ArgumentParser(description="combine run JSONs")
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = [json.loads(Path(p).read_text()) for p in a.json]
    rows = [r["ref"] for r in runs]
    var = [v for r in runs for v in r["variants"] if v is not None]
    write_summary(rows, var, a.out)


def write_summary(rows, var, path) -> None:
    summary = {"samples": len(rows), "iters": {}, "sampled": {}}
    for c in rows[0]["iters"]:
        summary["iters"][c] = float(np.mean([r["iters"][c] for r in rows if c in r["iters"]]))
    for k in rows[0]["sampled"]:
        summary["sampled"][k] = {m: float(np.mean([r["sampled"][k][m] for r in rows if k in r["sampled"]]))
                                 for m in ("mae", "mae_p90", "mae_of_mean")}
    summary["ref_mean_abs"] = float(np.mean([r["ref_mean_abs"] for r in rows]))
    summary["br_minus_cfv"] = float(np.mean([r["br_minus_cfv"] for r in rows]))
    if all("proxy_mae" in r for r in rows):
        summary["proxy_mae"] = float(np.mean([r["proxy_mae"] for r in rows]))
    leaf_ms = [1e3 * r["leaf_seconds"][-1] / r["leaf_rows"][-1] for r in rows]
    summary["ref"] = {"ms_per_network_row": float(np.mean(leaf_ms)), "by_sample": leaf_ms,
                      "gpus": [r["gpu"] for r in rows],
                      "seconds": [r["seconds"][-1] for r in rows],
                      "network_share": float(np.mean([r["leaf_seconds"][-1] / r["seconds"][-1] for r in rows])),
                      "exploitability": {c: float(np.mean([r["exploitability"][i] for r in rows]))
                                         for i, c in enumerate(rows[0]["checks"])}}
    groups: dict = {}
    for v in var:
        groups.setdefault((v["deals"], v["simultaneous"]), []).append(v)
    summary["variants"] = {}
    for (K, sim), vs in sorted(groups.items()):
        cks = vs[0]["checks"]
        summary["variants"][f"{'sim' if sim else 'alt'}_K{K}"] = {
            "seconds_per_iteration": float(np.mean([v["seconds"][-1] / v["checks"][-1] for v in vs])),
            "network_ms_per_row": float(np.mean([1e3 * v["leaf_seconds"][-1] / v["leaf_rows"][-1] for v in vs])),
            "equity_seconds": float(np.mean([v["equity_seconds"] for v in vs])),
            "vs_ref_same_deals": {c: float(np.mean([v["vs_ref_same_deals"][str(c)] if str(c) in v["vs_ref_same_deals"]
                                                    else v["vs_ref_same_deals"][c] for v in vs])) for c in cks},
            "exploitability": {c: float(np.mean([v["exploitability"][i] for v in vs])) for i, c in enumerate(cks)}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"summary": summary, "refs": rows, "variants": var}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "run":
        return run(argv[1:])
    if argv and argv[0] == "summarize":
        return summarize(argv[1:])
    if argv and argv[0] == "files":
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--K", default="25,50,100,200,400")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Ks = [int(x) for x in a.K.split(",")]
    refs, rows = {}, []
    for p in a.ref:
        s = load(p)
        refs[(s["info"]["flops"], int(s["info"]["args"]["seed"]))] = s
        rows.append(analyse_ref(s, Ks, a.draws, a.seed))
        print(f"{p}: iters {rows[-1]['iters']}", flush=True)
    var = [r for r in (analyse_variant(load(p), refs) for p in a.variants) if r is not None]
    write_summary(rows, var, a.out)


if __name__ == "__main__":
    main()
