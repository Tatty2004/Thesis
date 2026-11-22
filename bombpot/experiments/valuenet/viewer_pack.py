"""Copy what viewer_server needs from a real_trunk output directory into one folder to take to
another machine: trunk.json, strategy.npz, the averaged leaf values in float32 (leafcfv32.npy,
half of leafcfv.npy), grade.json when there is one, and any rivers already solved.

    python -m bombpot.experiments.valuenet.viewer_pack --trunk results/valuenet/real/<name> --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.trunk), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("trunk.json", "grade.json", "strategy.npz"):
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    cfv = np.load(src / "leafcfv.npy", mmap_mode="r")
    dst = np.lib.format.open_memmap(out / "leafcfv32.npy", mode="w+", dtype=np.float32, shape=cfv.shape)
    for q in range(cfv.shape[0]):
        for d in range(0, cfv.shape[1], 64):
            dst[q, d:d + 64] = cfv[q, d:d + 64]
    dst.flush()
    if (src / "river_cache").is_dir():
        shutil.copytree(src / "river_cache", out / "river_cache", dirs_exist_ok=True)
    print(f"packed {src} into {out}")


if __name__ == "__main__":
    main()
