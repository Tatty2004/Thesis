"""valuenet/leaf.py: NetLeaf's units (a constant network) and its batch path (one forward
pass over every (deal, line, close)) against a per-item loop through features.py."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.valuenet.features import (FeatureConfig, bin_index, encode_boards, histogram, normalize,  # noqa: E402
                                       pot_features, to_hands)
from bombpot.valuenet.leaf import NetLeaf  # noqa: E402
from bombpot.valuenet.model import ModelConfig, ValueNet  # noqa: E402

D, L, C, H = 3, 4, 2, 36


class Constant(torch.nn.Module):
    """Outputs c (in pots) in every bin for both players, no correction."""

    def __init__(self, features, c):
        super().__init__()
        self.features, self.c = features, c

    def forward(self, board, scalars, h0, h1):
        return torch.full_like(h0, self.c), torch.full_like(h1, self.c)


def setup(seed=0):
    rng = np.random.default_rng(seed)
    equity = rng.random((D, H, 2))
    possible = rng.random((D, H)) > 0.3
    equity[~possible] = 0
    reach = [rng.random((D, L, C, H)) ** 2 for _ in (0, 1)]
    reach[0][1, 2, 0] = 0.0   # player 0 never gets here
    reach[1][2, 3, 1] = 0.0   # player 1 never gets here
    pot = rng.uniform(2, 100, (L, C))
    stack = 101 - pot / 2
    kappa = rng.uniform(0.1, 1.0, D)
    boards = np.stack([rng.choice(15, 8, replace=False).reshape(2, 4) for _ in range(D)])
    return equity, possible, reach, pot, stack, kappa, boards


@pytest.mark.parametrize("p", [0, 1])
def test_constant_network_units(p):
    equity, possible, reach, pot, stack, kappa, boards = setup()
    cfg = FeatureConfig("joint_2d", 4, num_ranks=5, num_suits=3)
    c = 0.375  # exact in float32, the network's dtype here
    leaf = NetLeaf(Constant(cfg, c), equity, pot, stack, kappa, boards_by_deal=boards, possible_by_deal=possible,
                   batch_size=5)
    agg = leaf.values(p, 1, tuple(reach))
    opp = (reach[1 - p] * possible[:, None, None, :]).sum(-1)
    want = c * pot[None] * opp * kappa[:, None, None]
    assert agg.shape == (D, L, C, H)
    assert np.allclose(agg, np.where(possible[:, None, None, :], want[..., None], 0.0), rtol=1e-12, atol=0)
    # line classes: a list in, a list out
    leaf2 = NetLeaf(Constant(cfg, c), equity, [pot, pot[:2]], [stack, stack[:2]], kappa, boards_by_deal=boards,
                    possible_by_deal=possible)
    out = leaf2.values(p, 1, [tuple(reach), tuple(r[:, :2] for r in reach)])
    assert np.allclose(out[0], agg, rtol=1e-12) and np.allclose(out[1], agg[:, :2], rtol=1e-12)


@pytest.mark.parametrize("cfg", [FeatureConfig("avg_1d", 10, num_ranks=5, num_suits=3),
                                 FeatureConfig("joint_2d", 4, "ids", 5, 3)])
def test_batch_equals_per_item_loop(cfg):
    equity, possible, reach, pot, stack, kappa, boards = setup(1)
    torch.manual_seed(0)
    model = ValueNet(cfg, ModelConfig(depth=3, width=32)).double().eval()
    leaf = NetLeaf(model, equity, pot, stack, kappa, boards_by_deal=boards, possible_by_deal=possible, batch_size=7)
    for p in (0, 1):
        agg = leaf.values(p, 1, tuple(reach))
        want = np.zeros_like(agg)
        for d in range(D):
            bins = bin_index(equity[d], cfg)
            board = torch.as_tensor(encode_boards(boards[d], cfg))[None]
            board = board if cfg.board == "ids" else board.double()
            for l in range(L):
                for c in range(C):
                    n0, m0 = normalize(reach[0][d, l, c], possible[d])
                    n1, m1 = normalize(reach[1][d, l, c], possible[d])
                    h = [torch.as_tensor(histogram(x, bins, cfg.num_bins))[None] for x in (n0, n1)]
                    scal = torch.as_tensor(pot_features(pot[l, c], stack[l, c])).double()[None]
                    with torch.no_grad():
                        v = model(board, scal, h[0], h[1])[p][0].numpy()
                    m = (m0, m1)[1 - p]
                    want[d, l, c] = np.where(possible[d], to_hands(v, bins) * pot[l, c] * m * kappa[d], 0.0)
        assert np.allclose(agg, want, rtol=1e-10, atol=1e-12)
        assert np.all(agg[2, 3, 1] == 0) if p == 0 else np.all(agg[1, 2, 0] == 0)  # opponent never here
        assert np.abs(agg[1, 2, 0]).max() > 0 if p == 0 else np.abs(agg[2, 3, 1]).max() > 0  # own reach 0: defined


def test_per_hand_network_equals_per_item_loop():
    """HandSetNet through NetLeaf: per-hand features of every deal, batches of model.chunk."""
    from bombpot.valuenet.features import hand_cards, hand_features
    from bombpot.valuenet.model import HandSetNet

    cfg = FeatureConfig("hand_set", H, num_ranks=5, num_suits=3)
    rng = np.random.default_rng(5)
    flops = rng.permutation(15)[:6]
    rest = np.setdiff1d(np.arange(15), flops)
    iu = np.triu_indices(9, 1)
    hands = np.stack([rest[iu[0]], rest[iu[1]]], -1)
    turns = np.stack([rng.choice(rest, 2, replace=False) for _ in range(D)])
    boards = np.concatenate([np.broadcast_to(flops.reshape(2, 3), (D, 2, 3)), turns[..., None]], -1)
    possible = np.stack([~np.isin(hands, t).any(-1) for t in turns])
    equity = rng.random((D, H, 2)) * possible[..., None]
    reach = [rng.random((D, L, C, H)) ** 2 for _ in (0, 1)]
    reach[0][1, 2, 0] = 0.0
    pot = rng.uniform(2, 100, (L, C))
    stack, kappa = 101 - pot / 2, rng.uniform(0.1, 1.0, D)
    torch.manual_seed(0)
    model = HandSetNet(cfg, ModelConfig(kind="set", depth=2, width=24)).double().eval()
    model.chunk = 5
    with pytest.raises(ValueError):
        NetLeaf(model, equity, pot, stack, kappa, boards_by_deal=boards, possible_by_deal=possible)
    leaf = NetLeaf(model, equity, pot, stack, kappa, boards_by_deal=boards, possible_by_deal=possible, hands=hands)
    assert leaf.batch_size == 5
    hc = torch.as_tensor(hand_cards(hands))[None]
    for p in (0, 1):
        agg = leaf.values(p, 1, tuple(reach))
        want = np.zeros_like(agg)
        for d in range(D):
            hx = torch.as_tensor(hand_features(equity[d], hands, boards[d], cfg))[None]
            assert np.array_equal(hx[0, :, 2].numpy() > 0, possible[d])
            board = torch.as_tensor(encode_boards(boards[d], cfg))[None].double()
            for l in range(L):
                for c in range(C):
                    n0, m0 = normalize(reach[0][d, l, c], possible[d])
                    n1, m1 = normalize(reach[1][d, l, c], possible[d])
                    scal = torch.as_tensor(pot_features(pot[l, c], stack[l, c])).double()[None]
                    with torch.no_grad():
                        v = model(board, scal, torch.as_tensor(n0)[None], torch.as_tensor(n1)[None], hx=hx, hc=hc)[p][0]
                    m = (m0, m1)[1 - p]
                    want[d, l, c] = np.where(possible[d], v.numpy() * pot[l, c] * m * kappa[d], 0.0)
        assert np.allclose(agg, want, rtol=1e-9, atol=1e-12)


def test_devices_and_blocks_give_the_same_values():
    """Rows shared over several replicas, in small blocks, equal one replica's."""
    cfg = FeatureConfig("joint_2d", 4, "ids", 5, 3)
    equity, possible, reach, pot, stack, kappa, boards = setup(2)
    torch.manual_seed(0)
    model = ValueNet(cfg, ModelConfig(depth=3, width=32)).double().eval()
    kw = dict(boards_by_deal=boards, possible_by_deal=possible)
    one = NetLeaf(model, equity, pot, stack, kappa, **kw)
    many = NetLeaf(model, equity, pot, stack, kappa, devices=["cpu", "cpu", "cpu"], block=5, batch_size=3, **kw)
    assert len(many.replicas) == 3
    for p in (0, 1):
        np.testing.assert_allclose(many.values(p, 1, tuple(reach)), one.values(p, 1, tuple(reach)), rtol=1e-12, atol=1e-14)
