"""Train the counterfactual value network on HDF5 shards (bombpot/valuenet/data.py).

    python -m bombpot.experiments.valuenet.train --config bombpot/experiments/configs/valuenet/<name>.yaml
        [--run NAME] [--data DIR] [--device cuda|cpu|auto] [--epochs N] [--max-minutes M] [--synthetic]
        [--init best.pt]

Loss: Huber (delta `train.huber_delta`, in pots) on the per-bin values of both players,
weighted by each bin's range mass (data.py; or, with data.weight "hands", by its share of the
possible hands), normalized by the total weight of the batch: a weighted mean over bins, in
pots. train.schedule: "cosine" / "step" (per epoch), or "cosine_time": a cosine over wall time
that reaches 0 after train.time_budget_minutes (updated every step; training stops at the end
of the epoch that reaches it), for runs under a job time limit whose speed isn't known in
advance. train.init (--init): start from a saved model's weights (same features and model
config; a fresh optimizer and schedule), to continue a run in a second job. train.amp: mixed precision on CUDA (the loss stays in float32). data.target picks the per-hand values behind the bin targets: "cfv" (the solved
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
With data.eval_dir, every epoch also scores the model on all samples of that directory (an
exact test set kept apart from the training data, e.g. flop-end labels of held-out flop
pairs): eval_loss, eval_hand_mae, eval_hand_mae_floor. It plays no part in training or model
selection. data.limit uses only the first N samples of data.dir (both sides of its split).
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
    "data": {"dir": None, "holdout_fraction": 0.2, "block": 512, "salt": "valuenet-split", "target": "cfv", "weight": "range", "split": "flop_pair",
             "eval_dir": None, "limit": None},
    "features": {"method": "joint_2d", "n_bins": 20, "board": "onehot", "num_ranks": 13, "num_suits": 4},
    "model": {"depth": 7, "width": 500, "zero_sum": True},
    "train": {"epochs": 50, "batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0, "schedule": "cosine",
              "step_size": 20, "gamma": 0.3, "huber_delta": 1.0, "seed": 0, "threads": 8, "device": "auto",
              "max_minutes": None, "buffer_blocks": 8, "clip": None, "amp": False, "time_budget_minutes": None, "init": None},
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
    for k in ("hx", "hc"):  # per-hand inputs (features.method "hand_set")
        if k in batch:
            t[k] = torch.as_tensor(batch[k], device=device)
    return t


def forward(model, t: dict):
    """The model's (v0, v1) for a to_torch batch, with the per-hand inputs if the batch has
    them."""
    return model(t["board"], t["scalars"], t["h0"], t["h1"], **{k: t[k] for k in ("hx", "hc") if k in t})


def forward_chunked(model, t: dict):
    """`forward` in pieces of the model's `chunk` samples (models whose memory grows fast with
    the batch), for evaluation."""
    import torch

    chunk, n = getattr(model, "chunk", None), len(t["scalars"])
    if not chunk or n <= chunk:
        return forward(model, t)
    parts = [forward(model, {k: v[a:a + chunk] for k, v in t.items()}) for a in range(0, n, chunk)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def evaluate(model, shards, split: str, device, delta: float, batch_size: int) -> dict:
    """Held-out loss, the zero baseline, and the per-hand error (and its floor)."""
    import torch

    from bombpot.valuenet.features import to_hands

    model.eval()
    sums = dict.fromkeys(("loss", "zero", "w", "mae", "floor", "mass"), 0.0)
    with torch.no_grad():
        for batch in shards.batches(split, batch_size):
            t = to_torch(batch, device, model.features.board == "ids")
            v = forward_chunked(model, t)
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
    from bombpot.valuenet.model import ModelConfig, build_model, save_model

    cfg = merged(cfg)
    tc = cfg["train"]
    torch.set_num_threads(int(tc["threads"]))
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    device = pick_device(tc["device"])
    fcfg = FeatureConfig(**cfg["features"])
    dc = cfg["data"]
    shards = ShardSet(dc["dir"], fcfg, dc["holdout_fraction"], dc["block"], dc["salt"], dc["target"], dc["weight"],
                      dc["split"], limit=dc["limit"])
    test = ShardSet(dc["eval_dir"], fcfg, 0.0, dc["block"], target="cfv", weight=dc["weight"]) if dc["eval_dir"] else None
    n_train, n_held = shards.count("train"), shards.count("heldout")
    if n_held == 0 or n_train == 0:
        raise ValueError(f"split has {n_train} training and {n_held} held-out samples")
    model = build_model(fcfg, ModelConfig(**cfg["model"])).to(device)
    if tc["init"]:
        from bombpot.valuenet.model import load_model

        prev = load_model(tc["init"], device)
        if prev.features != model.features or prev.config != model.config:
            raise ValueError(f"train.init {tc['init']} has other features or another model config")
        model.load_state_dict(prev.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    if tc["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    elif tc["schedule"] == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tc["step_size"], gamma=tc["gamma"])
    elif tc["schedule"] == "cosine_time":
        if not tc["time_budget_minutes"]:
            raise ValueError("schedule cosine_time needs train.time_budget_minutes")
        sched = None
    else:
        raise ValueError(f"unknown schedule {tc['schedule']!r}")
    budget = 60.0 * tc["time_budget_minutes"] if sched is None else None
    amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**git_info(), "device": str(device), "torch": torch.__version__, "host": platform.node(),
            "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "train_samples": n_train, "heldout_samples": n_held,
            "shards": [str(p) for p in shards.paths], "eval_samples": test.count("all") if test else 0,
            "parameters": sum(p.numel() for p in model.parameters())}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"device {device} gpu {meta['gpu']} | train {n_train} held-out {n_held} | params {meta['parameters']}")

    fields = ["epoch", "seconds", "lr", "train_loss", "heldout_loss", "heldout_zero_loss", "heldout_hand_mae",
              "heldout_hand_mae_floor"] + (["eval_loss", "eval_hand_mae", "eval_hand_mae_floor"] if test else [])
    start, best, rows = time.perf_counter(), math.inf, []
    delta, bs = tc["huber_delta"], tc["batch_size"]
    with open(run_dir / "log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, tc["epochs"] + 1):
            tot, wsum = 0.0, 0.0
            for batch in shards.batches("train", bs, rng, tc["buffer_blocks"]):
                if budget is not None:
                    frac = min((time.perf_counter() - start) / budget, 1.0)
                    for group in opt.param_groups:
                        group["lr"] = 0.5 * tc["lr"] * (1.0 + math.cos(math.pi * frac))
                t = to_torch(batch, device, fcfg.board == "ids")
                with torch.autocast("cuda", enabled=amp):
                    v = forward(model, t)
                l0, w0 = weighted_huber(v[0].float(), t["t0"], t["w0"], delta)
                l1, w1 = weighted_huber(v[1].float(), t["t1"], t["w1"], delta)
                loss = (l0 + l1) / (w0 + w1).clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if tc["clip"]:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
                scaler.step(opt)
                scaler.update()
                tot += (l0 + l1).item()
                wsum += (w0 + w1).item()
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step()
            ev = evaluate(model, shards, "heldout", device, delta, 4 * bs)
            row = {"epoch": epoch, "seconds": round(time.perf_counter() - start, 2), "lr": lr,
                   "train_loss": tot / max(wsum, 1e-300), "heldout_loss": ev["loss"],
                   "heldout_zero_loss": ev["zero_loss"], "heldout_hand_mae": ev["hand_mae"],
                   "heldout_hand_mae_floor": ev["hand_mae_floor"]}
            if test is not None:
                et = evaluate(model, test, "all", device, delta, 4 * bs)
                row.update(eval_loss=et["loss"], eval_hand_mae=et["hand_mae"], eval_hand_mae_floor=et["hand_mae_floor"])
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
            if budget is not None and time.perf_counter() - start >= budget:
                log(f"time budget of {tc['time_budget_minutes']} minutes reached after epoch {epoch}")
                break
    shards.close()
    if test is not None:
        test.close()
    best_row = min(rows, key=lambda r: r["heldout_loss"])
    summary = {"epochs": len(rows), "best": best_row, "first": rows[0], "last": rows[-1],
               "seconds": rows[-1]["seconds"], **meta}
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
                              num_suits=fc["num_suits"], flop_pairs=sc["flop_pairs"], noise=sc["noise"],
                              card_hands=fc["method"] == "hand_set")
    log(f"wrote {sc['shards']} synthetic shards to {d}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", help="run name (default: the config's file name)")
    ap.add_argument("--data", help="shard directory (overrides data.dir)")
    ap.add_argument("--device")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--init", help="start from this saved model's weights")
    ap.add_argument("--limit", type=int, help="use only the first N samples (data.limit)")
    ap.add_argument("--eval-dir", help="exact test shards scored every epoch (data.eval_dir)")
    ap.add_argument("--target", help="label set (data.target), e.g. cfv or k50cfv")
    ap.add_argument("--seed", type=int, help="train.seed")
    ap.add_argument("--epochs-for", type=int, help="set train.epochs to max(min_epochs, min(max_epochs, "
                                                   "samples_seen / N)) for N samples (train.samples_seen etc.)")
    ap.add_argument("--synthetic", action="store_true", help="write synthetic shards first if the directory has none")
    a = ap.parse_args(argv)
    path = Path(a.config)
    cfg = merged(yaml.safe_load(path.read_text()))
    if a.data:
        cfg["data"]["dir"] = a.data
    for key, val in (("limit", a.limit), ("eval_dir", a.eval_dir), ("target", a.target)):
        if val is not None:
            cfg["data"][key] = val
    if a.seed is not None:
        cfg["train"]["seed"] = a.seed
    if a.epochs_for:  # a learning-curve point: about the same number of samples seen, within bounds
        tc = cfg["train"]
        tc["epochs"] = int(max(tc.get("min_epochs", 30), min(tc.get("max_epochs", 400),
                                                            round(tc.get("samples_seen", 1.5e6) / a.epochs_for))))
    for key, val in (("device", a.device), ("epochs", a.epochs), ("max_minutes", a.max_minutes), ("init", a.init)):
        if val is not None:
            cfg["train"][key] = val
    if cfg["data"]["dir"] is None:
        raise SystemExit("no data directory: set data.dir or --data")
    for key in ("dir", "eval_dir"):
        if cfg["data"][key] is not None:
            d = Path(cfg["data"][key])
            cfg["data"][key] = str(d if d.is_absolute() else ROOT / d)
    if a.synthetic:
        make_synthetic(cfg)
    run = a.run or path.stem
    summary = train(cfg, NET / run)
    b = summary["best"]
    print(f"best epoch {b['epoch']}: held-out loss {b['heldout_loss']:.6g} (zero {b['heldout_zero_loss']:.6g}), "
          f"hand MAE {b['heldout_hand_mae']:.6g} pots (floor {b['heldout_hand_mae_floor']:.6g}); "
          f"{summary['seconds']:.0f} s on {summary['device']} {summary['gpu'] or ''}")
    if "eval_hand_mae" in b:
        print(f"eval (data.eval_dir) hand MAE at that epoch {b['eval_hand_mae']:.6g} pots, "
              f"at the last epoch {summary['last']['eval_hand_mae']:.6g}")


if __name__ == "__main__":
    main()
