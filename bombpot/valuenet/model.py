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

HandSetNet (ModelConfig.kind "set", features.method "hand_set") is the other network: every
hand is its own input and output, and one set of weights serves all hands.

    per hand i:  x_i = [card embeddings and rank / suit one-hots of its two cards, pair,
                        suited, features.hand_features (equities, possible, card-vs-board
                        counts), both players' range mass at i]
    global:      g = MLP([board, pot / stack])
    h_i = encoder(x_i) + g
    `depth` blocks, each  h_i += MLP(norm(h)_i, pooled, card_i, g)  with
        pooled = [sum_j r0_j z_j, sum_j r1_j z_j, mean over possible j of z_j]   (z = norm(h))
        card_i = for each player, the range-weighted sum of z_j over the hands j that hold
                 one of hand i's cards (the hands i blocks; i itself counts twice)
        and, with `heads` > 0, self-attention over the possible hands
    (v0_i, v1_i) = decoder(h_i), in pots, then the same zero-sum correction over hands.

A counterfactual value is a sum over the opponent's hands that share no card with i of range
mass times a payoff, so range-weighted sums over hands (all of them, and those through each
card) are the statistics it is made of; several rounds let the payoff depend on both ranges,
as the solved strategies do. Nothing depends on the order of the hands: permuting them
permutes the outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from bombpot.valuenet.features import HAND_FEATURES, NUM_SCALARS, FeatureConfig

KINDS = ("mlp", "set")


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 7         # hidden layers (kind "set": blocks)
    width: int = 500       # units per hidden layer (kind "set": width of a hand's state)
    zero_sum: bool = True  # apply the zero-sum correction
    embed_dim: int = 8     # card embedding width, for FeatureConfig.board == "ids" and kind "set"
    kind: str = "mlp"      # "mlp" (ValueNet) or "set" (HandSetNet)
    heads: int = 0         # kind "set": attention heads per block (0: pooling only)
    card_context: bool = True  # kind "set": pool over the hands through each card too
    dropout: float = 0.0   # kind "set": dropout on each block's update

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}")

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

    def forward(self, board, scalars, h0, h1, **_):
        """board (N, board_dim) floats or card ids, scalars (N, 2) from pot_features,
        h0 / h1 (N, B) range histograms -> (v0, v1), each (N, B), in pots. Per-hand inputs
        (hx, hc) are ignored."""
        if self.embed is not None:
            board = self.embed(board).flatten(1)
        x = torch.cat([board.to(h0.dtype), scalars.to(h0.dtype), h0, h1], dim=-1)
        v0, v1 = self.mlp(x).chunk(2, dim=-1)
        if self.config.zero_sum:
            v0, v1 = zero_sum_correction(v0, v1, h0, h1)
        return v0, v1


class SetBlock(nn.Module):

    def __init__(self, d: int, num_cards: int, heads: int, card_context: bool, dropout: float):
        super().__init__()
        self.num_cards, self.card_context = num_cards, card_context
        self.norm = nn.LayerNorm(d)
        self.own = nn.Linear(d, d)
        self.pooled = nn.Linear(4 * d, d, bias=False)
        self.card = nn.Linear(2 * d, d, bias=False) if card_context else None
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True) if heads else None
        self.out = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, h, g, w, inc, possible):
        """h (N, H, d), g (N, d), w (N, H, 3) pooling weights (range0, range1, uniform over
        the possible hands), inc (N, H, cards) 1 where the hand holds the card, possible
        (N, H) bool."""
        z = self.norm(h)
        N, H, d = z.shape
        pooled = torch.einsum("nhk,nhd->nkd", w, z).reshape(N, 3 * d)
        u = self.own(z) + self.pooled(torch.cat([pooled, g], -1)).unsqueeze(1)
        if self.card is not None:  # sums over the hands through each card, then back to hands
            zw = (z.unsqueeze(2) * w[..., :2].unsqueeze(-1)).reshape(N, H, 2 * d)
            per_card = torch.bmm(inc.transpose(1, 2), zw)
            u = u + self.card(torch.bmm(inc, per_card) * (self.num_cards / 4))
        if self.attn is not None:
            u = u + self.attn(z, z, z, key_padding_mask=~possible, need_weights=False)[0]
        return h + self.out(u)


class HandSetNet(nn.Module):
    """The network that shares its weights across hands (module docstring). `chunk` is the
    largest batch an evaluation loop should pass at once (a hand's state is `width` floats, so
    a sample costs H x width per layer)."""

    chunk = 128
    per_hand = True  # forward needs hx and hc

    def __init__(self, features: FeatureConfig, config: ModelConfig = ModelConfig(kind="set", depth=4, width=256)):
        super().__init__()
        if features.method != "hand_set":
            raise ValueError("HandSetNet needs features.method 'hand_set'")
        self.features, self.config = features, config
        d, R, S, C = config.width, features.num_ranks, features.num_suits, features.num_cards
        self.embed = nn.Embedding(C, config.embed_dim)
        ids = torch.arange(C)
        desc = torch.cat([nn.functional.one_hot(ids // S, R), nn.functional.one_hot(ids % S, S)], -1).float()
        self.register_buffer("card_desc", desc, persistent=False)
        board_width = features.board_dim * (config.embed_dim if features.board == "ids" else 1)
        self.glob = nn.Sequential(nn.Linear(board_width + NUM_SCALARS, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.Sequential(nn.Linear(2 * (config.embed_dim + R + S) + 2 + HAND_FEATURES + 4, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.blocks = nn.ModuleList(SetBlock(d, C, config.heads, config.card_context, config.dropout)
                                    for _ in range(config.depth))
        self.dec = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, board, scalars, h0, h1, hx, hc):
        """board, scalars as ValueNet; h0 / h1 (N, H) the two ranges; hx (N, H, HAND_FEATURES)
        from features.hand_features; hc (N, H, 2) card ids -> (v0, v1), each (N, H), in pots.
        Hands that can't be held (hx[..., 2] = 0) get no weight in any pooling."""
        dt = h0.dtype
        hc = hc.long()
        hx = hx.to(dt)
        S = self.features.num_suits
        if self.features.board == "ids":
            board = self.embed(board).flatten(1)
        g = self.glob(torch.cat([board.to(dt), scalars.to(dt)], -1))
        possible = hx[..., 2] > 0
        count = possible.sum(-1, keepdim=True).clamp_min(1).to(dt)
        mass = [torch.where(possible, r, torch.zeros_like(r)) for r in (h0, h1)]
        rel = [m * count for m in mass]                      # 1 = the uniform range's mass
        rng = torch.stack([f(u) for u in rel for f in (torch.log1p, lambda u: u / (1 + u))], -1)
        cards = torch.cat([self.embed(hc), self.card_desc.to(dt)[hc]], -1).flatten(2)
        pair = (hc[..., 0] // S == hc[..., 1] // S).to(dt)
        suited = (hc[..., 0] % S == hc[..., 1] % S).to(dt)
        x = torch.cat([cards, pair.unsqueeze(-1), suited.unsqueeze(-1), hx, rng], -1)
        h = self.enc(x) + g.unsqueeze(1)
        w = torch.stack([mass[0], mass[1], possible.to(dt) / count], -1)
        inc = nn.functional.one_hot(hc, self.features.num_cards).sum(2).to(dt)
        for block in self.blocks:
            h = block(h, g, w, inc, possible)
        v0, v1 = self.dec(h).to(dt).unbind(-1)  # the correction in the inputs' precision (autocast)
        if self.config.zero_sum:
            with torch.autocast(v0.device.type, enabled=False):
                v0, v1 = zero_sum_correction(v0, v1, mass[0], mass[1])
        return v0, v1


def build_model(features: FeatureConfig, config: ModelConfig) -> nn.Module:
    """The network of `config.kind`."""
    return HandSetNet(features, config) if config.kind == "set" else ValueNet(features, config)


def save_model(path, model, **info) -> None:
    """The weights with both configs (and any `info`), so load_model needs nothing else."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": model.features.to_dict(), "model": model.config.to_dict(),
                "state_dict": model.state_dict(), "info": info}, path)


def load_model(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_model(FeatureConfig.from_dict(blob["features"]), ModelConfig.from_dict(blob["model"]))
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
        self.per_hand = getattr(members[0], "per_hand", False)
        if hasattr(members[0], "chunk"):
            self.chunk = members[0].chunk

    def forward(self, board, scalars, h0, h1, **hand):
        outs = [m(board, scalars, h0, h1, **hand) for m in self.members]
        n = len(outs)
        return sum(o[0] for o in outs) / n, sum(o[1] for o in outs) / n


def load_ensemble(paths, device="cpu") -> Ensemble:
    """An Ensemble of the saved ValueNets at `paths`."""
    return Ensemble([load_model(p, device) for p in paths]).to(device).eval()
