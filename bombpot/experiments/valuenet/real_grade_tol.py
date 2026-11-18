"""How much of a real-deck exact grade is the river re-solve tolerance (PLAN_valuenet.md, G.3).

A full regrade at a tighter tolerance costs hundreds of GPU-hours, so a sample of turn deals is
re-solved at the tighter tolerance (real_grade --part d/2070 into a directory that symlinks the
same trunk's inputs). Here the base grade's best-response leaf values on those deals are replaced
by the tight ones and both best responses are recomputed on the flop+turn tree. The change, scaled
by D / |sample|, estimates the change of a full tight regrade (to first order the best response's
flop and turn choices stay fixed, so the change is linear in the per-deal leaf changes).

    python -m bombpot.experiments.valuenet.real_grade_tol --base results/valuenet/real/attn_op1_r2 \
        --tight results/valuenet/real/attn_op1_r2_tol1e-3 [--tight ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bombpot.experiments.valuenet import real_grade as rg


def load_parts(grade_dir: Path, D: int, L: int, H: int, dense: bool):
    """All part files of a grade directory: (br (2, D, L, H) or {deal: (2, L, H)}, stats per deal)."""
    br = np.zeros((2, D, L, H)) if dense else {}
    stats = {}
    for path in sorted(grade_dir.glob("part_*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            for j, d in enumerate(f["deals"]):
                if dense:
                    br[:, int(d)] = f["br"][j]
                else:
                    br[int(d)] = f["br"][j]
                stats[int(d)] = f["stats"][j]
    return br, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--tight", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = Path(a.base)
    game, upper, river = rg.build(base)
    D, H, L = upper.streets[1].num_deals, len(upper.hands), len(rg.river_lines(upper))
    br, base_stats = load_parts(base / "grade", D, L, H, dense=True)
    if len(base_stats) != D:
        raise SystemExit(f"base grade has {len(base_stats)} of {D} deals")
    avg = rg.load_strategy(base / "strategy.npz", upper)
    ref = rg.finish(upper, avg, br)
    print("base", json.dumps(ref), flush=True)
    out = {"base": ref, "tight": {}}
    for t in a.tight:
        tdir = Path(t)
        tight, tstats = load_parts(tdir / "grade", D, L, H, dense=False)
        deals = np.array(sorted(tight))
        if len(deals) == 0:
            continue
        mixed = br.copy()
        for d in deals:
            mixed[:, d] = tight[d]
        res = rg.finish(upper, avg, mixed)
        scale = D / len(deals)
        tol = float(np.load(next((tdir / "grade").glob("part_*.npz")))["tol"])
        delta = {k: res[k] - ref[k] for k in ("exploitability", "br_value_p0", "br_value_p1")}
        est = {k: ref[k] + scale * v for k, v in delta.items()}
        it_b = np.array([base_stats[d][..., 0].sum() for d in deals])
        it_t = np.array([tstats[d][..., 0].sum() for d in deals])
        sec_t = np.array([tstats[d][..., 2].sum() for d in deals])
        capped = int(sum((tstats[d][..., 0] >= 5000).sum() for d in deals))
        out["tight"][str(tdir)] = {"tol": tol, "deals": len(deals), "sample_result": res, "sample_delta": delta,
                                   "estimated_full": est, "iterations_ratio": float(it_t.sum() / it_b.sum()),
                                   "mean_gpu_seconds_per_deal": float(sec_t.mean()),
                                   "resolves_at_iteration_cap": capped,
                                   "largest_gap_over_mass": float(max(tstats[d][..., 1].max() for d in deals))}
        print(tdir.name, json.dumps(out["tight"][str(tdir)]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
