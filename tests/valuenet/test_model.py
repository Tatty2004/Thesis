"""valuenet/model.py: output shapes, the zero-sum correction, save / load."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.valuenet.features import FeatureConfig  # noqa: E402
from bombpot.valuenet.model import ModelConfig, ValueNet, load_model, save_model, zero_sum_correction  # noqa: E402


def inputs(cfg, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    if cfg.board == "ids":
        board = torch.randint(0, cfg.num_cards, (n, cfg.board_dim), generator=g)
    else:
        board = torch.rand(n, cfg.board_dim, generator=g).round()
    h = [torch.rand(n, cfg.num_bins, generator=g) ** 4 for _ in (0, 1)]
    h = [x / x.sum(-1, keepdim=True) for x in h]
    return board, torch.rand(n, 2, generator=g), h[0], h[1]


@pytest.mark.parametrize("cfg", [FeatureConfig("avg_1d", 50), FeatureConfig("joint_2d", 20),
                                 FeatureConfig("joint_2d", 5, "ids", 5, 3), FeatureConfig("avg_1d", 8, "none")])
def test_shapes_and_zero_sum(cfg):
    torch.manual_seed(0)
    model = ValueNet(cfg).double()
    assert sum(isinstance(m, torch.nn.PReLU) for m in model.modules()) == 7
    board, scal, h0, h1 = inputs(cfg, 64)
    v0, v1 = model(board, scal.double(), h0.double(), h1.double())
    assert v0.shape == v1.shape == (64, cfg.num_bins)
    resid = (h0 * v0).sum(-1) + (h1 * v1).sum(-1)
    assert resid.abs().max().item() < 1e-6
    # float32 too, at float32 rounding
    v0, v1 = ValueNet(cfg, ModelConfig(depth=2, width=32))(board, scal, h0, h1)
    assert ((h0 * v0).sum(-1) + (h1 * v1).sum(-1)).abs().max().item() < 1e-6


def test_correction_with_unnormalized_masses_and_off_switch():
    g = torch.Generator().manual_seed(1)
    v0, v1, h0, h1 = (torch.randn(10, 7, generator=g, dtype=torch.float64) for _ in range(4))
    h0, h1 = h0.abs() * 3, h1.abs()
    c0, c1 = zero_sum_correction(v0, v1, h0, h1)
    assert ((h0 * c0).sum(-1) + (h1 * c1).sum(-1)).abs().max().item() < 1e-12
    assert torch.allclose(c0 - v0, c1 - v1)  # one shift for both players
    cfg = FeatureConfig("avg_1d", 4)
    model = ValueNet(cfg, ModelConfig(depth=1, width=8, zero_sum=False))
    board, scal, a, b = inputs(cfg, 5)
    raw = model.mlp(torch.cat([board, scal, a, b], -1))
    assert torch.equal(torch.cat(model(board, scal, a, b), -1), raw)


def test_save_load_round_trip(tmp_path):
    cfg = FeatureConfig("joint_2d", 6, "ids", 5, 3)
    model = ValueNet(cfg, ModelConfig(depth=3, width=40, embed_dim=4))
    save_model(tmp_path / "m.pt", model, epoch=3)
    back = load_model(tmp_path / "m.pt")
    assert back.features == cfg and back.config == model.config and back.info == {"epoch": 3}
    x = inputs(cfg, 9)
    with torch.no_grad():
        for a, b in zip(model.eval()(*x), back(*x)):
            assert torch.equal(a, b)
