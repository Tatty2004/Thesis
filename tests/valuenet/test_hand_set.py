"""The network that shares weights across hands (valuenet/model.py HandSetNet, features.method
"hand_set"): per-hand features, shapes, the zero-sum identity, equivariance under permuting
the hands, save / load, the data path and a short synthetic training."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.valuenet.features import HAND_FEATURES, FeatureConfig, hand_cards, hand_features  # noqa: E402
from bombpot.valuenet.model import (HandSetNet, ModelConfig, ValueNet, build_model, load_model,  # noqa: E402
                                    save_model)

CFG = FeatureConfig("hand_set", 36, num_ranks=5, num_suits=3)  # 15 cards, 6 on the flops, C(9, 2) hands


def sample(cfg, n, seed=0):
    """Random samples on cfg's deck: one flop pair, hands = pairs of the other cards."""
    rng = np.random.default_rng(seed)
    C = cfg.num_cards
    flops = rng.permutation(C)[:6]
    rest = np.setdiff1d(np.arange(C), flops)
    iu = np.triu_indices(len(rest), 1)
    hands = np.stack([rest[iu[0]], rest[iu[1]]], -1)
    H = len(hands)
    turn = np.stack([rng.choice(rest, 2, replace=False) for _ in range(n)])
    boards = np.concatenate([np.broadcast_to(flops.reshape(2, 3), (n, 2, 3)), turn[..., None]], -1)
    hx = hand_features(rng.random((n, H, 2)), hands, boards, cfg)
    possible = hx[..., 2] > 0
    ranges = [possible * rng.random((n, H)) ** 3 for _ in (0, 1)]
    ranges = [r / r.sum(-1, keepdims=True) for r in ranges]
    from bombpot.valuenet.features import encode_boards
    t = {"board": torch.as_tensor(encode_boards(boards, cfg)), "scalars": torch.as_tensor(rng.random((n, 2))),
         "h0": torch.as_tensor(ranges[0]), "h1": torch.as_tensor(ranges[1]), "hx": torch.as_tensor(hx),
         "hc": torch.as_tensor(np.broadcast_to(hand_cards(hands), (n, H, 2)).copy())}
    return t, boards, hands


def run(model, t):
    return model(t["board"], t["scalars"], t["h0"], t["h1"], hx=t["hx"], hc=t["hc"])


def test_hand_features():
    cfg = FeatureConfig("hand_set", 3)
    # boards As Ks 2c | 7d / Ah Kh Qh | 2d ; hands: AdKd (possible), 7d7h (blocked by the turn), 2h3h
    card = lambda r, s: "23456789TJQKA".index(r) * 4 + "cdhs".index(s)  # noqa: E731
    boards = np.array([[card("A", "s"), card("K", "s"), card("2", "c"), card("7", "d")],
                       [card("A", "h"), card("K", "h"), card("Q", "h"), card("2", "d")]])
    hands = np.array([[card("K", "d"), card("A", "d")], [card("7", "d"), card("7", "h")], [card("2", "h"), card("3", "h")]])
    eq = np.array([[0.9, 0.5], [0.0, 0.0], [0.2, 0.7]])
    x = hand_features(eq, hands, boards, cfg)
    assert x.shape == (3, HAND_FEATURES) and x.dtype == np.float32
    assert np.array_equal(hand_cards(hands)[0], [card("A", "d"), card("K", "d")])  # higher card first
    assert np.allclose(x[:, :2], eq) and x[:, 2].tolist() == [1.0, 0.0, 1.0]
    # AdKd: A pairs one card on each board, K too; diamonds: one on A (7d), one on B (2d)
    assert np.allclose(x[0, 3:], [.25, .25, .25, .25, .25, .25, .25, .25])
    # 3h2h: the 3 pairs nothing, the 2 pairs one card on each board; hearts: none on A, three on B
    assert np.allclose(x[2, 3:], [0, 0, .25, .25, 0, .75, 0, .75])
    # a leading batch axis with a shared hand list
    xb = hand_features(np.stack([eq, eq]), hands, np.stack([boards, boards[::-1]]), cfg)
    assert np.array_equal(xb[0], x) and np.allclose(xb[1, 2, 3:], [0, 0, .25, .25, .75, 0, .75, 0])


@pytest.mark.parametrize("mc", [ModelConfig(kind="set", depth=2, width=32),
                                ModelConfig(kind="set", depth=2, width=32, heads=2),
                                ModelConfig(kind="set", depth=1, width=16, card_context=False)])
def test_shapes_zero_sum_and_impossible_hands(mc):
    torch.manual_seed(0)
    model = build_model(CFG, mc).double().eval()
    assert isinstance(model, HandSetNet)
    t, _, _ = sample(CFG, 7)
    v0, v1 = run(model, t)
    assert v0.shape == v1.shape == (7, 36)
    assert ((t["h0"] * v0).sum(-1) + (t["h1"] * v1).sum(-1)).abs().max().item() < 1e-10
    # what a hand that can't be held carries (its equity, its range entry) changes nothing
    # for the hands that can
    possible = t["hx"][..., 2] > 0
    assert not possible.all()
    u = dict(t)
    u["hx"] = t["hx"].clone()
    u["hx"][..., :2] = torch.where(possible.unsqueeze(-1), t["hx"][..., :2], torch.ones(()))
    u["h0"] = torch.where(possible, t["h0"], torch.full((), 0.3, dtype=torch.float64))
    w0, w1 = run(model, u)
    assert torch.allclose(w0[possible], v0[possible], atol=1e-10) and torch.allclose(w1[possible], v1[possible], atol=1e-10)


@pytest.mark.parametrize("heads", [0, 2])
def test_permutation_equivariance(heads):
    torch.manual_seed(1)
    model = HandSetNet(CFG, ModelConfig(kind="set", depth=3, width=24, heads=heads)).double().eval()
    t, _, _ = sample(CFG, 5, seed=3)
    v0, v1 = run(model, t)
    perm = torch.randperm(36, generator=torch.Generator().manual_seed(2))
    u = {k: (v[:, perm] if k in ("h0", "h1", "hx", "hc") else v) for k, v in t.items()}
    w0, w1 = run(model, u)
    assert torch.allclose(w0, v0[:, perm], atol=1e-9) and torch.allclose(w1, v1[:, perm], atol=1e-9)
    assert not torch.allclose(v0, w0, atol=1e-3)  # and the outputs do differ across hands


def test_save_load_round_trip_and_kinds(tmp_path):
    model = HandSetNet(CFG, ModelConfig(kind="set", depth=2, width=20, heads=2, embed_dim=4))
    save_model(tmp_path / "m.pt", model, epoch=5)
    back = load_model(tmp_path / "m.pt")
    assert isinstance(back, HandSetNet) and back.features == CFG and back.config == model.config
    assert back.info == {"epoch": 5}
    t, _, _ = sample(CFG, 4)
    t = {k: (v.float() if v.dtype == torch.float64 else v) for k, v in t.items()}
    with torch.no_grad():
        for a, b in zip(run(model.eval(), t), run(back, t)):
            assert torch.equal(a, b)
    # the per-hand MLP takes the same inputs and ignores the per-hand ones; "mlp" stays the default
    mlp = build_model(CFG, ModelConfig(depth=2, width=16))
    assert isinstance(mlp, ValueNet) and run(mlp, t)[0].shape == (4, 36)
    with pytest.raises(ValueError):
        ModelConfig(kind="nope")
    with pytest.raises(ValueError):
        HandSetNet(FeatureConfig("hand", 36, num_ranks=5, num_suits=3), ModelConfig(kind="set"))


def test_training_on_synthetic_shards(tmp_path):
    """The data path (hx, hc from the shard's hand lists, one per flop pair) and a short CPU
    training."""
    pytest.importorskip("h5py")
    from bombpot.experiments.valuenet.train import evaluate, make_synthetic, train
    from bombpot.valuenet.data import ShardSet

    cfg = {"data": {"dir": str(tmp_path / "data"), "holdout_fraction": 0.25, "block": 256},
           "features": {"method": "hand_set", "n_bins": 36, "num_ranks": 5, "num_suits": 3},
           "model": {"kind": "set", "depth": 2, "width": 48},
           "train": {"epochs": 40, "batch_size": 32, "lr": 2e-3, "max_minutes": 0.5, "device": "cpu", "threads": 4},
           "synthetic": {"shards": 2, "samples": 600, "flop_pairs": 30}}
    make_synthetic(cfg, log=lambda m: None)
    shards = ShardSet(cfg["data"]["dir"], FeatureConfig(**cfg["features"]), 0.25, 256)
    batch = shards.read(0, 0, "all")
    raw = {k: shards.files[0][k][:256] for k in ("range0", "range1", "hand_set", "flops", "turn")}
    shards.close()
    assert batch["hx"].shape == (256, 36, HAND_FEATURES) and batch["hc"].shape == (256, 36, 2)
    possible = batch["hx"][..., 2] > 0  # derived from the cards; the ranges mark the same hands
    assert np.array_equal(possible, (raw["range0"] > 0) | (raw["range1"] > 0))
    assert len(set(raw["hand_set"].tolist())) > 1 and not np.array_equal(batch["hc"][0], batch["hc"][1])
    s = train(cfg, tmp_path / "run", log=lambda m: None)
    b = s["best"]
    print(f"\nhand_set: {s['epochs']} epochs, held-out loss {b['heldout_loss']:.4g} (zero {b['heldout_zero_loss']:.4g})")
    assert b["heldout_loss"] < 0.2 * b["heldout_zero_loss"]
    assert b["heldout_hand_mae_floor"] < 1e-6  # every hand its own bin
    model = load_model(tmp_path / "run" / "best.pt")
    assert isinstance(model, HandSetNet)
    model.chunk = 50  # evaluation in pieces gives the same numbers
    shards = ShardSet(cfg["data"]["dir"], model.features, 0.25, 256)
    ev = evaluate(model, shards, "heldout", "cpu", 1.0, 400)
    shards.close()
    assert abs(ev["loss"] - b["heldout_loss"]) < 1e-6 * max(1.0, b["heldout_loss"]) + 1e-7
