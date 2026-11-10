"""valuenet/features.py: equity-bin histograms, the bin assignment, board and pot inputs.
Numpy only, so these run without torch."""
import numpy as np
import pytest

from bombpot.valuenet.features import (FeatureConfig, bin_index, bin_means, board_features, board_ids,
                                       encode_boards, histogram, normalize, pot_features, to_hands)

H = 36


def sample(seed, n=None):
    rng = np.random.default_rng(seed)
    shape = (H,) if n is None else (n, H)
    eq = rng.random(shape + (2,))
    eq[..., :3, :] = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]  # the edges of [0, 1]
    possible = rng.random(shape) > 0.2
    r = rng.random(shape) * possible
    return eq, r, possible


@pytest.mark.parametrize("cfg", [FeatureConfig("avg_1d", 50), FeatureConfig("joint_2d", 20), FeatureConfig("avg_1d", 7)])
def test_histogram_sums_to_range_mass_and_shapes(cfg):
    eq, r, possible = sample(0)
    bins = bin_index(eq, cfg)
    assert bins.shape == (H,) and bins.min() >= 0 and bins.max() < cfg.num_bins
    h = histogram(r, bins, cfg.num_bins)
    assert h.shape == (cfg.num_bins,)
    assert cfg.num_bins == (cfg.n_bins if cfg.method == "avg_1d" else cfg.n_bins ** 2)
    assert h.sum() == pytest.approx(r.sum(), rel=1e-12)
    # batched equals one at a time
    eqs, rs, _ = sample(1, n=5)
    hb = histogram(rs, bin_index(eqs, cfg), cfg.num_bins)
    for n in range(5):
        assert np.allclose(hb[n], histogram(rs[n], bin_index(eqs[n], cfg), cfg.num_bins), atol=1e-15)


def test_bins_are_fixed_on_unit_interval():
    one = FeatureConfig("avg_1d", 10)
    assert bin_index(np.array([[0.0, 0.0], [0.05, 0.1], [0.5, 0.5], [1.0, 1.0], [0.99, 1.0]]), one).tolist() \
        == [0, 0, 5, 9, 9]
    two = FeatureConfig("joint_2d", 4)
    assert bin_index(np.array([[0.0, 0.0], [0.3, 0.9], [1.0, 0.0], [1.0, 1.0]]), two).tolist() == [0, 7, 12, 15]


@pytest.mark.parametrize("method", ["avg_1d", "joint_2d"])
def test_bin_assignment_round_trips(method):
    cfg = FeatureConfig(method, 6)
    eq, r, possible = sample(2)
    bins = bin_index(eq, cfg)
    vals = np.random.default_rng(3).standard_normal(cfg.num_bins)
    hands = to_hands(vals, bins)
    assert np.array_equal(hands, vals[bins])
    means, mass = bin_means(hands, r + 1e-3, bins, cfg.num_bins)  # a per-bin function comes back exactly
    assert np.allclose(means[mass > 0], vals[mass > 0], atol=1e-12)
    assert np.all(means[mass == 0] == 0)
    # weighted mean identity: sum_b mass_b mean_b = sum_i r_i v_i (what the zero-sum check relies on)
    v = np.random.default_rng(4).standard_normal(H)
    m, w = bin_means(v, r, bins, cfg.num_bins)
    assert (w * m).sum() == pytest.approx((r * v).sum(), rel=1e-12)


def test_impossible_hands_are_ignored():
    cfg = FeatureConfig("joint_2d", 5)
    eq, r, possible = sample(5)
    bins = bin_index(eq, cfg)
    junk = r + 7.0 * ~possible  # mass on hands that can't be held
    assert np.allclose(histogram(junk, bins, cfg.num_bins, possible), histogram(r, bins, cfg.num_bins), atol=1e-14)
    n, mass = normalize(junk, possible)
    assert n[~possible].sum() == 0 and n.sum() == pytest.approx(1.0) and mass == pytest.approx(r.sum())
    u, mass = normalize(np.zeros(H), possible)  # no mass: uniform over the possible hands
    assert mass == 0 and np.allclose(u[possible], 1 / possible.sum()) and u[~possible].sum() == 0


def test_board_encoding():
    cfg = FeatureConfig(num_ranks=13, num_suits=4)
    c = lambda r, s: r * 4 + s  # noqa: E731
    boards = np.array([[c(12, 2), c(5, 0), c(0, 1), c(5, 3)],   # Ah 7c 2d 7s: paired by the turn
                       [c(11, 3), c(11, 1), c(7, 2), c(4, 2)]])  # Ks Kd 9h 6h
    f = board_features(boards, cfg)
    assert f.shape == (cfg.board_dim,) and set(np.unique(f)) <= {0.0, 1.0}
    # flop card order doesn't matter
    perm = boards[:, [2, 0, 1, 3]]
    assert np.array_equal(board_features(perm, cfg), f) and np.array_equal(board_ids(perm, cfg), board_ids(boards, cfg))
    k, R = 4, 13
    per = k * R + 6 + k
    assert f[:k * R].reshape(k, R).sum(0)[5] == 2  # two sevens on board A
    assert f[k * R:k * R + 6].sum() == 0 and f[per + k * R:per + k * R + 6].sum() == 1  # B: one same-suit pair (9h 6h)
    assert np.argmax(f[k * R + 6:per]) == 0 and np.argmax(f[per + k * R + 6:2 * per]) == 1  # max suit counts 1, 2
    assert f[2 * per:].sum() == 4  # same suit across boards: Ah-9h, Ah-6h, 2d-Kd, 7s-Ks
    assert encode_boards(boards, FeatureConfig(board="ids")).shape == (8,)
    assert encode_boards(boards[None], FeatureConfig(board="none")).shape == (1, 0)
    small = FeatureConfig(num_ranks=5, num_suits=3)
    assert board_features(np.array([[0, 1, 2, 14], [3, 4, 5, 13]]), small).shape == (small.board_dim,)


def test_pot_features_and_config_round_trip():
    x = pot_features(np.array([2.0, 20.0]), np.array([100.0, 91.0]))
    assert x.shape == (2, 2) and np.allclose(x[:, 0], np.log([2, 20])) and np.allclose(x[:, 1], np.log1p([50, 4.55]))
    cfg = FeatureConfig("avg_1d", 33, "ids", 5, 3)
    assert FeatureConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        FeatureConfig("kmeans", 3)
