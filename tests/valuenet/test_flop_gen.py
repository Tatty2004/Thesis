"""Flop-end shards for the flop-net test (experiments/valuenet/flop_gen.py): 3-card boards and the
extra per-hand features, a shard written by flop_gen.Shard read back by data.ShardSet into a
HandSetNet, data.limit, the control-variate label, and the flop-pair draw's split."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from bombpot.experiments.valuenet import flop_gen as fg  # noqa: E402
from bombpot.valuenet.data import ShardSet, heldout_mask, flop_pair_key  # noqa: E402
from bombpot.valuenet.features import FeatureConfig, canonical_boards, hand_features  # noqa: E402
from bombpot.valuenet.model import ModelConfig, build_model  # noqa: E402


def test_canonical_boards_flop_and_turn():
    b4 = np.array([[[0, 40, 13, 7], [5, 51, 22, 30]]])
    assert canonical_boards(b4, FeatureConfig(board_cards=4)).tolist() == [[[40, 13, 0, 7], [51, 22, 5, 30]]]
    b3 = np.array([[[0, 40, 13], [5, 51, 22]]])
    assert canonical_boards(b3, FeatureConfig(board_cards=3)).tolist() == [[[40, 13, 0], [51, 22, 5]]]


def test_hand_features_extra():
    cfg = FeatureConfig("hand_set", 3, board_cards=3, hand_extra=2)
    hands = np.array([[10, 11], [12, 20], [30, 31]])
    boards = np.array([[0, 1, 2], [3, 4, 5]])
    eq, extra = np.random.default_rng(0).random((3, 2)), np.arange(6.0).reshape(3, 2)
    x = hand_features(eq, hands, boards, cfg, extra)
    assert x.shape == (3, 13) and np.allclose(x[:, 11:], extra) and np.allclose(x[:, :2], eq)
    base = hand_features(eq, hands, boards, FeatureConfig("hand_set", 3, board_cards=3))
    assert np.allclose(x[:, :11], base)
    with pytest.raises(ValueError):
        hand_features(eq, hands, boards, cfg)


def _shard(path, n, seed):
    rng = np.random.default_rng(seed)
    sh = fg.Shard(path, {"test": 1})
    H = 1035
    for s in range(n):
        cards = rng.permutation(52)
        flops = (tuple(sorted(cards[:3].tolist())), tuple(sorted(cards[3:6].tolist())))
        rest = np.setdiff1d(np.arange(52), cards[:6])
        iu = np.triu_indices(46, 1)
        hands = np.stack([rest[iu[0]], rest[iu[1]]], -1).astype(np.int8)
        r = [rng.random(H) for _ in (0, 1)]
        r = [x / x.sum() for x in r]
        v = rng.normal(0, 0.3, (2, H))
        sh.add({"flops": np.array(flops, np.int8), "hand_set": sh.hand_set(flops, hands), "pot": 12.0,
                "stack": 95.0, "line": 3, "range0": r[0], "range1": r[1], "equity": rng.random((H, 2)),
                "equity_sd": rng.random((H, 2)) * 0.2, "possible": np.ones(H, bool),
                "cfv0": v[0], "cfv1": v[1], "k50cfv0": v[0] + 0.01, "k50cfv1": v[1] - 0.01, "deals": 50})
    sh.close()


def test_shard_roundtrip(tmp_path):
    _shard(tmp_path / "a.h5", 6, 0)
    _shard(tmp_path / "b.h5", 5, 1)
    cfg = FeatureConfig("hand_set", 1035, board_cards=3, hand_extra=2)
    ss = ShardSet(tmp_path, cfg, 0.3, block=4, salt="t")
    assert ss.count("train") + ss.count("heldout") == 11 == ss.count("all")
    lim = ShardSet(tmp_path, cfg, 0.3, block=4, salt="t", limit=8)
    assert lim.count("all") == 8
    assert sum(len(b["scalars"]) for b in lim.batches("all", 3)) == 8
    k = ShardSet(tmp_path, cfg, 0.0, block=4, target="k50cfv")
    b0 = next(ShardSet(tmp_path, cfg, 0.0, block=4).batches("all", 16))
    b1 = next(k.batches("all", 16))
    assert np.allclose(b1["cfv0"], b0["cfv0"] + 0.01, atol=1e-6)
    assert b0["hx"].shape == (11, 1035, 13) and b0["board"].shape[-1] == cfg.board_dim
    model = build_model(cfg, ModelConfig(kind="set", depth=1, width=16, heads=2)).eval()
    t = {key: torch.as_tensor(b0[key]) for key in ("board", "scalars", "h0", "h1", "hx", "hc")}
    v0, v1 = model(t["board"].float(), t["scalars"], t["h0"], t["h1"], hx=t["hx"], hc=t["hc"])
    assert v0.shape == (11, 1035)
    zs = (t["h0"] * v0).sum(-1) + (t["h1"] * v1).sum(-1)
    assert torch.allclose(zs, torch.zeros_like(zs), atol=1e-5)


def test_cv_label():
    rng = np.random.default_rng(0)
    D, K, H, per = 30, 7, 5, 4
    f, vall = rng.random((2, D, H)), rng.random((2, D, H))
    S = np.sort(rng.choice(D, K, replace=False))
    lab = fg.cv_label(f, vall[:, S], S, per)
    assert np.allclose(lab, (f.sum(1) + (vall[:, S] - f[:, S]).sum(1) * D / K) / per)
    assert np.allclose(fg.cv_label(f, vall, np.arange(D), per), vall.sum(1) / per)  # every deal: exact


def test_draw_pairs_sides():
    rng = np.random.default_rng(5)
    bad = fg.never_keys()
    for side in ("train", "heldout"):
        pairs = fg.draw_pairs(rng, 20, side)
        held = heldout_mask(np.array(pairs), fg.FRACTION, fg.SALT)
        assert held.all() if side == "heldout" else not held.any()
        assert not any(flop_pair_key(*[list(f) for f in p]) in bad for p in pairs)
    assert len(bad) == 6
