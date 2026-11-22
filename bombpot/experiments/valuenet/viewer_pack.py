"""Copy what viewer_server needs from a real_trunk output directory into one folder to take to
another machine: trunk.json, strategy.npz, the averaged leaf values in float32 (leafcfv32.npy,
half of leafcfv.npy), grade.json when there is one, and any rivers already solved; for the
live mode also the value network (net.pt: --model, default trunk.json's), the turn deals'
equities (equity_<flops>.npy, else recomputed per deal) and, when the trunk ran with
--turn-cfv, its averaged turn-root values in float32 (turncfv.npy; without it the live turn
re-solve backs its T values up from leafcfv32.npy).

    python -m bombpot.experiments.valuenet.viewer_pack --trunk results/valuenet/real/<name> --out <dir> [--model net.pt]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def to_float32(src: Path, dst: Path, rows: int = 64) -> None:
    """(2, D, ...) float64 .npy -> float32, a block of deals at a time."""
    x = np.load(src, mmap_mode="r")
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=x.shape)
    for q in range(x.shape[0]):
        for d in range(0, x.shape[1], rows):
            out[q, d:d + rows] = x[q, d:d + rows]
    out.flush()
    del out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="the network for live re-solves (default: trunk.json's)")
    a = ap.parse_args(argv)
    src, out = Path(a.trunk), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("trunk.json", "grade.json", "strategy.npz"):
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    to_float32(src / "leafcfv.npy", out / "leafcfv32.npy")
    if (src / "turncfv.npy").exists():
        to_float32(src / "turncfv.npy", out / "turncfv.npy")
    args = json.loads((src / "trunk.json").read_text())["args"]
    shutil.copyfile(a.model or args["model"], out / "net.pt")
    nr, ns = args["deck"].split(",")
    eq = f"equity_{args['flops'].replace(',', '_')}_{nr}x{ns}.npy"
    for d in (src, src.parent):
        if (d / eq).exists():
            shutil.copyfile(d / eq, out / eq)
            break
    if (src / "river_cache").is_dir():
        shutil.copytree(src / "river_cache", out / "river_cache", dirs_exist_ok=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"packed {src} into {out}: {size / 2 ** 30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
