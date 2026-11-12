"""Train the counterfactual value network on HDF5 shards (bombpot/valuenet/data.py).

    python -m bombpot.experiments.valuenet.train --config bombpot/experiments/configs/valuenet/<name>.yaml
        [--run NAME] [--data DIR] [--device cuda|cpu|auto] [--epochs N] [--max-minutes M] [--synthetic]

Loss: Huber (delta `train.huber_delta`, in pots) on the per-bin values of both players,
weighted by each bin's range mass (data.py; or, with data.weight "hands", by its share of the
possible hands), normalized by the total weight of the batch: a weighted mean over bins, in
pots. data.target picks the per-hand values behind the bin targets: "cfv" (the solved
profile's) or "br" (best-response values). Adam, cosine or step learning rate. Held-out samples
are whole flop pairs (data.holdout_fraction).

Each epoch logs to results/valuenet/net/<run>/log.csv:
  epoch, seconds, lr, train_loss            mean training loss over the epoch
  heldout_loss, heldout_zero_loss           held-out loss, and that of predicting 0 (baseline)
  heldout_hand_mae                          per-hand error after mapping bins back to hands:
                                            sum_i r_i |pred(bin_i) - cfv_i| / sum_i r_i, both
                                            players, over held-out samples, in pots
  heldout_hand_mae_floor                    the same for the exact bin means: the within-bin
                                            spread, which per-bin outputs can't resolve
The best held-out model goes to best.pt (valuenet.model.load_model), with config.yaml and
meta.json (git, device, GPU) beside it; summary.json at the end.

`--synthetic` first writes synthetic shards (config `synthetic`) into the data directory if
it has none, for tests and for checking the GPU path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from bombpot.experiments.run import RESULTS, ROOT, git_info

NET = RESULTS / "valuenet" / "net"
DEFAULTS = {
    "data": {"dir": None, "holdout_fraction": 0.2, "block": 512, "salt": "valuenet-split", "target": "cfv", "weight": "range"},
    "features": {"method": "joint_2d", "n_bins": 20, "board": "onehot", "num_ranks": 13, "num_suits": 4},
    "model": {"depth": 7, "width": 500, "zero_sum": True},
    "train": {"epochs": 50, "batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0, "schedule": "cosine",
              "step_size": 20, "gamma": 0.3, "huber_delta": 1.0, "seed": 0, "threads": 8, "device": "auto",
              "max_minutes": None, "buffer_blocks": 8},
    "synthetic": {"shards": 4, "samples": 2000, "flop_pairs": 25, "noise": 0.02},
}


def merged(cfg: dict) -> dict:
    out = {k: dict(v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


def weighted_huber(pred, target, weight, delta: float):
    """sum w huber(pred - target) and sum w, over the batch."""
    import torch

    loss = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=delta)
    return (weight * loss).sum(), weight.sum()


def to_torch(batch: dict, device, board_ids: bool):
    import torch

    t = {k: torch.as_tensor(batch[k], device=device) for k in ("scalars", "h0", "h1", "t0", "t1", "w0", "w1")}
    t["board"] = torch.as_tensor(batch["board"], device=device)
    if not board_ids:
        t["board"] = t["board"].float()
    return t


def evaluate(model, shards, split: str, device, delta: float, batch_size: int) -> dict:
    """Held-out loss, the zero baseline, and the per-hand error (and its floor)."""
    import torch

    from bombpot.valuenet.features import to_hands

    model.eval()
    sums = dict.fromkeys(("loss", "zero", "w", "mae", "floor", "mass"), 0.0)
    with torch.no_grad():
        for batch in shards.batches(split, batch_size):
            t = to_torch(batch, device, model.features.board == "ids")
            v = model(t["board"], t["scalars"], t["h0"], t["h1"])
            for p in (0, 1):
                l, w = weighted_huber(v[p], t[f"t{p}"], t[f"w{p}"], delta)
                z, _ = weighted_huber(torch.zeros_like(v[p]), t[f"t{p}"], t[f"w{p}"], delta)
                sums["loss"] += l.item()
                sums["zero"] += z.item()
                sums["w"] += w.item()
                r, cfv = batch[f"range{p}"], batch[f"cfv{p}"]
                pred = to_hands(v[p].double().cpu().numpy(), batch["bins"])
                best = to_hands(batch[f"t{p}"].astype(np.float64), batch["bins"])
                sums["mae"] += float((r * np.abs(pred - cfv)).sum())
                sums["floor"] += float((r * np.abs(best - cfv)).sum())
                sums["mass"] += float(r.sum())
    model.train()
    w, m = max(sums["w"], 1e-300), max(sums["mass"], 1e-300)
    return {"loss": sums["loss"] / w, "zero_loss": sums["zero"] / w, "hand_mae": sums["mae"] / m,
            "hand_mae_floor": sums["floor"] / m}


def pick_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def train(cfg: dict, run_dir: Path, log=print) -> dict:
    import torch

    from bombpot.valuenet.data import ShardSet
    from bombpot.valuenet.features import FeatureConfig
    from bombpot.valuenet.model import ModelConfig, ValueNet, save_model

    cfg = merged(cfg)
    tc = cfg["train"]
    torch.set_num_threads(int(tc["threads"]))
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    device = pick_device(tc["device"])
    fcfg = FeatureConfig(**cfg["features"])
    shards = ShardSet(cfg["data"]["dir"], fcfg, cfg["data"]["holdout_fraction"], cfg["data"]["block"],
                      cfg["data"]["salt"], cfg["data"]["target"], cfg["data"]["weight"])
    n_train, n_held = shards.count("train"), shards.count("heldout")
    if n_held == 0 or n_train == 0:
        raise ValueError(f"split has {n_train} training and {n_held} held-out samples")
    model = ValueNet(fcfg, ModelConfig(**cfg["model"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    if tc["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    elif tc["schedule"] == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tc["step_size"], gamma=tc["gamma"])
    else:
        raise ValueError(f"unknown schedule {tc['schedule']!r}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**git_info(), "device": str(device), "torch": torch.__version__, "host": platform.node(),
            "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "train_samples": n_train, "heldout_samples": n_held,
            "shards": [str(p) for p in shards.paths],
            "parameters": sum(p.numel() for p in model.parameters())}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"device {device} gpu {meta['gpu']} | train {n_train} held-out {n_held} | params {meta['parameters']}")

    fields = ["epoch", "seconds", "lr", "train_loss", "heldout_loss", "heldout_zero_loss", "heldout_hand_mae",
              "heldout_hand_mae_floor"]
    start, best, rows = time.perf_counter(), math.inf, []
    delta, bs = tc["huber_delta"], tc["batch_size"]
    with open(run_dir / "log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, tc["epochs"] + 1):
            tot, wsum = 0.0, 0.0
            for batch in shards.batches("train", bs, rng, tc["buffer_blocks"]):
                t = to_torch(batch, device, fcfg.board == "ids")
                v = model(t["board"], t["scalars"], t["h0"], t["h1"])
                l0, w0 = weighted_huber(v[0], t["t0"], t["w0"], delta)
                l1, w1 = weighted_huber(v[1], t["t1"], t["w1"], delta)
                loss = (l0 + l1) / (w0 + w1).clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += (l0 + l1).item()
                wsum += (w0 + w1).item()
            lr = opt.param_groups[0]["lr"]
            sched.step()
            ev = evaluate(model, shards, "heldout", device, delta, 4 * bs)
            row = {"epoch": epoch, "seconds": round(time.perf_counter() - start, 2), "lr": lr,
                   "train_loss": tot / max(wsum, 1e-300), "heldout_loss": ev["loss"],
                   "heldout_zero_loss": ev["zero_loss"], "heldout_hand_mae": ev["hand_mae"],
                   "heldout_hand_mae_floor": ev["hand_mae_floor"]}
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            log(" ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            if ev["loss"] < best:
                best = ev["loss"]
                save_model(run_dir / "best.pt", model, epoch=epoch, heldout_loss=ev["loss"],
                           hand_mae=ev["hand_mae"])
            if tc["max_minutes"] and time.perf_counter() - start > 60 * tc["max_minutes"]:
                log(f"stopping: max_minutes {tc['max_minutes']} reached after epoch {epoch}")
                break
    shards.close()
    best_row = min(rows, key=lambda r: r["heldout_loss"])
    summary = {"epochs": len(rows), "best": best_row, "first": rows[0], "seconds": rows[-1]["seconds"], **meta}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def make_synthetic(cfg: dict, log=print) -> None:
    from bombpot.valuenet.data import write_synthetic_shard

    cfg = merged(cfg)
    d = Path(cfg["data"]["dir"])
    if d.is_dir() and any(d.glob("*.h5")):
        return
    sc, fc = cfg["synthetic"], cfg["features"]
    for i in range(sc["shards"]):
        write_synthetic_shard(d / f"synthetic_{i:03d}.h5", sc["samples"], seed=1000 + i, num_ranks=fc["num_ranks"],
                              num_suits=fc["num_suits"], flop_pairs=sc["flop_pairs"], noise=sc["noise"])
    log(f"wrote {sc['shards']} synthetic shards to {d}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", help="run name (default: the config's file name)")
    ap.add_argument("--data", help="shard directory (overrides data.dir)")
    ap.add_argument("--device")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--synthetic", action="store_true", help="write synthetic shards first if the directory has none")
    a = ap.parse_args(argv)
    path = Path(a.config)
    cfg = merged(yaml.safe_load(path.read_text()))
    if a.data:
        cfg["data"]["dir"] = a.data
    for key, val in (("device", a.device), ("epochs", a.epochs), ("max_minutes", a.max_minutes)):
        if val is not None:
            cfg["train"][key] = val
    if cfg["data"]["dir"] is None:
        raise SystemExit("no data directory: set data.dir or --data")
    d = Path(cfg["data"]["dir"])
    cfg["data"]["dir"] = str(d if d.is_absolute() else ROOT / d)
    if a.synthetic:
        make_synthetic(cfg)
    run = a.run or path.stem
    summary = train(cfg, NET / run)
    b = summary["best"]
    print(f"best epoch {b['epoch']}: held-out loss {b['heldout_loss']:.6g} (zero {b['heldout_zero_loss']:.6g}), "
          f"hand MAE {b['heldout_hand_mae']:.6g} pots (floor {b['heldout_hand_mae_floor']:.6g}); "
          f"{summary['seconds']:.0f} s on {summary['device']} {summary['gpu'] or ''}")


if __name__ == "__main__":
    main()
