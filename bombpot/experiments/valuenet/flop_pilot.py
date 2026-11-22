"""Analysis of the flop-end sample cost pilot (flop_datagen.py): label error of cheap variants
against the full solve, and the cost of each.

    python -m bombpot.experiments.valuenet.flop_pilot --ref DIR/sample_*_all.npz \
        [--variants DIR/sample_*_<K>*.npz] --out results/valuenet/flopnet_pilot/pilot.json

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


def main(argv=None):
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
    # summary: mean over reference samples of every variant's error
    summary = {"iters": {}, "sampled": {}}
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
    summary["ms_per_network_row"] = {"mean": float(np.mean(leaf_ms)), "by_sample": leaf_ms,
                                     "gpus": [r["gpu"] for r in rows]}
    summary["network_share"] = float(np.mean([r["leaf_seconds"][-1] / r["seconds"][-1] for r in rows]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "refs": rows, "variants": var}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
