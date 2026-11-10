"""Synthetic training (experiments/valuenet/train.py): shards from a known smooth zero-sum
function of (equities, ranges, pot, stack, board) plus noise, small deck; half a minute of
CPU training per bin method must bring the held-out loss well below predicting zeros."""
import csv

import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")

from bombpot.experiments.valuenet.train import make_synthetic, train  # noqa: E402
from bombpot.valuenet.model import load_model  # noqa: E402


@pytest.mark.parametrize("method,n", [("avg_1d", 20), ("joint_2d", 6)])
def test_synthetic_training_beats_zero(tmp_path_factory, method, n):
    d = tmp_path_factory.getbasetemp() / "synthetic_small"
    cfg = {"data": {"dir": str(d), "holdout_fraction": 0.25, "block": 256},
           "features": {"method": method, "n_bins": n, "num_ranks": 5, "num_suits": 3},
           "model": {"depth": 3, "width": 128},
           "train": {"epochs": 500, "batch_size": 128, "lr": 1e-3, "max_minutes": 0.5, "device": "cpu", "threads": 8},
           "synthetic": {"shards": 3, "samples": 1500, "flop_pairs": 30}}
    make_synthetic(cfg, log=lambda m: None)
    run = tmp_path_factory.mktemp(method)
    s = train(cfg, run, log=lambda m: None)
    b, first = s["best"], s["first"]
    print(f"\n{method}: {s['epochs']} epochs, held-out loss {first['heldout_loss']:.4g} -> {b['heldout_loss']:.4g} "
          f"(zero {b['heldout_zero_loss']:.4g}); hand MAE {first['heldout_hand_mae']:.4g} -> {b['heldout_hand_mae']:.4g} "
          f"(floor {b['heldout_hand_mae_floor']:.4g})")
    assert b["heldout_loss"] < 0.1 * b["heldout_zero_loss"]
    assert b["heldout_hand_mae"] < 0.8 * first["heldout_hand_mae"]
    rows = list(csv.DictReader(open(run / "log.csv")))
    assert len(rows) == s["epochs"]
    model = load_model(run / "best.pt")
    assert model.info["epoch"] == b["epoch"] and model.features.method == method
