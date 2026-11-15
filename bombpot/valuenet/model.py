"""The counterfactual value network (DeepStack, Moravcik et al. 2017, supplementary):

    [board, pot / stack, range0 histogram, range1 histogram]
        -> MLP (default 7 hidden layers of 500 units, PReLU)
        -> per-bin counterfactual values of both players, in pots
        -> zero-sum correction

Values are in pots: a sample's root counterfactual values divided by its pot, under the
two ranges normalized to sum 1 (features.normalize). The game is zero sum, so the true
values satisfy  sum_b h0[b] v0[b] + sum_b h1[b] v1[b] = 0  (h = the range histograms, v the
range-weighted mean value per bin). The correction layer enforces it: with
err = sum h0 v0 + sum h1 v1 it subtracts err / (sum h0 + sum h1) from every bin of both
players (DeepStack's "outer" correction; for normalized ranges that is err / 2).

Hands take their bin's value (features.to_hands).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from bombpot.valuenet.features import NUM_SCALARS, FeatureConfig


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 7         # hidden layers
    width: int = 500       # units per hidden layer
    zero_sum: bool = True  # apply the zero-sum correction
    embed_dim: int = 8     # card embedding width, for FeatureConfig.board == "ids"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


def zero_sum_correction(v0, v1, h0, h1):
    """Shift both players' per-bin values by one constant so that
    sum h0 v0 + sum h1 v1 = 0 for each sample (rows of (N, B) tensors)."""
    err = (h0 * v0).sum(-1, keepdim=True) + (h1 * v1).sum(-1, keepdim=True)
    mass = h0.sum(-1, keepdim=True) + h1.sum(-1, keepdim=True)
    delta = err / torch.where(mass > 0, mass, torch.ones_like(mass))
    return v0 - delta, v1 - delta


class ValueNet(nn.Module):

    def __init__(self, features: FeatureConfig, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.features, self.config = features, config
        B = features.num_bins
        if features.board == "ids":
            self.embed = nn.Embedding(features.num_cards, config.embed_dim)
            board_width = features.board_dim * config.embed_dim
        else:
            self.embed = None
            board_width = features.board_dim
        layers, width = [], board_width + NUM_SCALARS + 2 * B
        for _ in range(config.depth):
            layers += [nn.Linear(width, config.width), nn.PReLU(config.width)]
            width = config.width
        layers.append(nn.Linear(width, 2 * B))
        self.mlp = nn.Sequential(*layers)

    def forward(self, board, scalars, h0, h1):
        """board (N, board_dim) floats or card ids, scalars (N, 2) from pot_features,
        h0 / h1 (N, B) range histograms -> (v0, v1), each (N, B), in pots."""
        if self.embed is not None:
            board = self.embed(board).flatten(1)
        x = torch.cat([board.to(h0.dtype), scalars.to(h0.dtype), h0, h1], dim=-1)
        v0, v1 = self.mlp(x).chunk(2, dim=-1)
        if self.config.zero_sum:
            v0, v1 = zero_sum_correction(v0, v1, h0, h1)
        return v0, v1


def save_model(path, model: ValueNet, **info) -> None:
    """The weights with both configs (and any `info`), so load_model needs nothing else."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": model.features.to_dict(), "model": model.config.to_dict(),
                "state_dict": model.state_dict(), "info": info}, path)


def load_model(path, device="cpu") -> ValueNet:
    blob = torch.load(path, map_location=device, weights_only=True)
    model = ValueNet(FeatureConfig.from_dict(blob["features"]), ModelConfig.from_dict(blob["model"]))
    model.load_state_dict(blob["state_dict"])
    model.info = blob.get("info", {})
    return model.to(device).eval()


class Ensemble(nn.Module):
    """The mean of several ValueNets' outputs (same features). The zero-sum constraint
    sum h0 v0 + sum h1 v1 = 0 is linear in (v0, v1) for given ranges, so the mean of corrected
    members satisfies it too; no further correction is needed."""

    def __init__(self, members):
        super().__init__()
        members = list(members)
        if not members:
            raise ValueError("an ensemble needs at least one member")
        f = members[0].features
        if any(m.features != f for m in members):
            raise ValueError("ensemble members must share their FeatureConfig")
        self.members = nn.ModuleList(members)
        self.features = f
        self.info = {"ensemble": [getattr(m, "info", {}) for m in members]}

    def forward(self, board, scalars, h0, h1):
        outs = [m(board, scalars, h0, h1) for m in self.members]
        n = len(outs)
        return sum(o[0] for o in outs) / n, sum(o[1] for o in outs) / n


def load_ensemble(paths, device="cpu") -> Ensemble:
    """An Ensemble of the saved ValueNets at `paths`."""
    return Ensemble([load_model(p, device) for p in paths]).to(device).eval()
