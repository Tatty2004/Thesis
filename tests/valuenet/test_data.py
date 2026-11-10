"""valuenet/data.py: shards in the spec's layout, featurized targets, the flop-pair split."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bombpot.valuenet.data import FIELDS, ShardSet, featurize, heldout_mask, write_synthetic_shard  # noqa: E402
from bombpot.valuenet.features import FeatureConfig  # noqa: E402

CFG = FeatureConfig("joint_2d", 5, num_ranks=5, num_suits=3)


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    for i in range(3):
        write_synthetic_shard(d / f"s{i}.h5", 300, seed=i, num_ranks=5, num_suits=3, flop_pairs=12)
    return d


def test_layout(shard_dir):
    with h5py.File(shard_dir / "s0.h5", "r") as f:
        N, H = f["range0"].shape
        assert H == 36 and f["hands"].shape == (36, 2) and f["equity"].shape == (N, H, 2)
        assert f["flops"].shape == (N, 2, 3) and f["turn"].shape == (N, 2)
        r0, r1, eq = f["range0"][:], f["range1"][:], f["equity"][:]
        assert np.allclose(r0.sum(1), 1, atol=1e-5) and np.allclose(r1.sum(1), 1, atol=1e-5)
        assert ((r0 > 0).sum(1) == 21).all()  # C(7, 2) hands possible after the turn cards
        zs = (r0 * f["cfv0"][:]).sum(1) + (r1 * f["cfv1"][:]).sum(1)
        assert np.abs(zs).max() < 1e-5  # synthetic values are zero sum
        assert np.all(eq[r0 == 0] == 0)


def test_featurized_targets(shard_dir):
    with h5py.File(shard_dir / "s1.h5", "r") as f:
        raw = {k: f[k][:50] for k in FIELDS}
    x = featurize(raw, CFG)
    B = CFG.num_bins
    assert x["h0"].shape == x["t0"].shape == (50, B) and x["board"].shape == (50, CFG.board_dim)
    assert np.allclose(x["h0"].sum(1), 1, atol=1e-5)
    for p in (0, 1):  # mass x bin mean = the range-weighted sum of hand values, so zero sum survives
        assert np.allclose((x[f"h{p}"] * x[f"t{p}"]).sum(1), (raw[f"range{p}"] * raw[f"cfv{p}"]).sum(1), atol=1e-5)
        assert np.all(x[f"t{p}"][x[f"h{p}"] == 0] == 0)


def test_split_by_flop_pair(shard_dir):
    s = ShardSet(shard_dir, CFG, holdout_fraction=0.3, block=64)
    flops = np.concatenate([f["flops"][:] for f in s.files])
    held = np.concatenate(s.heldout)
    keys = [tuple(sorted(tuple(sorted(b)) for b in fp)) for fp in flops]
    side = {}
    for k, h in zip(keys, held):
        assert side.setdefault(k, h) == h  # one side per flop pair
    assert 0 < held.mean() < 1
    swapped = flops[:, ::-1, ::-1]  # board order and card order don't matter
    assert np.array_equal(heldout_mask(swapped, 0.3), held)
    n = {sp: sum(len(b["scalars"]) for b in s.batches(sp, 100, np.random.default_rng(0), buffer=2))
         for sp in ("train", "heldout", "all")}
    assert n["train"] == s.count("train") and n["heldout"] == s.count("heldout") and n["all"] == 900
    assert n["train"] + n["heldout"] == 900
    s.close()
