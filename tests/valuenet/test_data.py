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
    assert np.array_equal(heldout_mask(swapped, 0.3, num_suits=CFG.num_suits), held)
    n = {sp: sum(len(b["scalars"]) for b in s.batches(sp, 100, np.random.default_rng(0), buffer=2))
         for sp in ("train", "heldout", "all")}
    assert n["train"] == s.count("train") and n["heldout"] == s.count("heldout") and n["all"] == 900
    assert n["train"] + n["heldout"] == 900
    s.close()


def test_split_respects_suit_isomorphism():
    from bombpot.valuenet.data import flop_pair_key

    def cards(s):  # "Ah7c2d" -> ids, suits c d h s
        r, su = "23456789TJQKA", "cdhs"
        return [r.index(s[i]) * 4 + su.index(s[i + 1]) for i in range(0, len(s), 2)]

    a = (cards("Ah7c2d"), cards("KsKd9h"))
    b = (cards("Ad7h2c"), cards("KsKc9d"))  # the relabeling h->d, c->h, d->c, s->s
    c = (cards("Ah7c2d"), cards("KsKd9d"))  # a different class: 9 shares the K's suit
    assert flop_pair_key(*a) == flop_pair_key(*b) == flop_pair_key(*b[::-1])
    assert flop_pair_key(*a) != flop_pair_key(*c)
    flops = np.array([[a[0], a[1]], [b[1], b[0]]])
    assert heldout_mask(flops, 0.5)[0] == heldout_mask(flops, 0.5)[1]


def test_hands_weighting_trains_bins_the_range_leaves_empty():
    # weight="hands": a bin's target is the plain mean of its possible hands' values and its
    # weight their share of the possible hands, whatever the player's range; the inputs stay
    # the range histograms.
    from bombpot.valuenet.data import featurize
    from bombpot.valuenet.features import bin_index

    rng = np.random.default_rng(0)
    n, H = 6, 36
    cfg = FeatureConfig("avg_1d", 5, num_ranks=5, num_suits=3)
    possible = rng.random((n, H)) > 0.2
    equity = rng.random((n, H, 2)) * possible[..., None]
    raw = {"flops": np.tile(np.array([[0, 3, 7], [6, 9, 13]]), (n, 1, 1)), "turn": np.tile(np.array([1, 2]), (n, 1)),
           "equity": equity, "pot": np.full(n, 6.0), "stack": np.full(n, 98.0), "possible": possible}
    for p in (0, 1):
        r = rng.random((n, H)) * possible * (rng.random((n, H)) > 0.5)  # half the possible hands have no range
        raw[f"range{p}"] = r / r.sum(-1, keepdims=True)
        raw[f"br{p}"] = rng.standard_normal((n, H)) * possible
    a = featurize(raw, cfg, "br", "range")
    b = featurize(raw, cfg, "br", "hands")
    bins = bin_index(equity, cfg)
    for p in (0, 1):
        assert np.array_equal(a[f"h{p}"], b[f"h{p}"]) and np.array_equal(a[f"w{p}"], a[f"h{p}"])
        assert np.allclose(b[f"w{p}"].sum(-1), 1, atol=1e-6)
        for s in range(n):
            for k in range(cfg.num_bins):
                m = possible[s] & (bins[s] == k)
                assert b[f"w{p}"][s, k] == pytest.approx(m.sum() / possible[s].sum(), abs=1e-6)
                if m.any():
                    assert b[f"t{p}"][s, k] == pytest.approx(raw[f"br{p}"][s][m].mean(), abs=1e-5)
