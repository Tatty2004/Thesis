"""Network inputs, numpy only: ranges as histograms over equity bins, the boards, the pot.

A hand's turn-end equity on a board is its average showdown share over the river runouts
against a uniform opponent range (0 where it can't be held), so `equity` is (H, 2) per turn
deal. The bins are fixed on [0, 1], not quantiles, so a bin means the same thing on every
board:

  avg_1d    n bins over the mean of the two equities              (n_bins bins)
  joint_2d  an n x n grid over (equity A, equity B), A major       (n_bins**2 bins)
  hand      every hand its own bin (n_bins = H): the range itself, no abstraction. Only
            meaningful for samples that share one hand list (one flop pair): the reference
            that separates what binning costs from what the network costs
  hand_set  like "hand" (every hand its own bin), and each sample also carries per-hand
            descriptors (`hand_features`: equities, possible flag, how each card meets each
            board) and the hands' card ids, for a network that shares its parameters across
            hands (model.HandSetNet). Needs the shard's hand list.

Bin b of method avg_1d holds equities in [b/n, (b+1)/n), the last bin also holds 1.0.

A range's histogram is the range mass in each bin (so it sums to the range's mass over the
hands counted). Per-bin outputs map back to hands through the same assignment: a hand
takes its bin's value (`to_hands`). Everything here takes a leading batch axis or not.

The board encoding is board-relative and game-agnostic: card ids are `rank * num_suits +
suit` (the core convention), and the game only enters through num_ranks and num_suits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

METHODS = ("avg_1d", "joint_2d", "hand", "hand_set")
PER_HAND = ("hand", "hand_set")  # every hand its own bin
BOARDS = ("onehot", "ids", "none")


@dataclass(frozen=True)
class FeatureConfig:
    """How samples become network inputs. Saved with every model (to_dict / from_dict).

    method       "avg_1d" or "joint_2d"
    n_bins       bins per axis (avg_1d: n_bins bins; joint_2d: n_bins x n_bins)
    board        "onehot" (rank one-hots and suit patterns, `board_features`), "ids" (card
                 ids for a learned embedding, `board_ids`), or "none"
    num_ranks, num_suits   the deck, for decoding card ids
    board_cards  cards per board at the end of the turn
    """
    method: str = "joint_2d"
    n_bins: int = 20
    board: str = "onehot"
    num_ranks: int = 13
    num_suits: int = 4
    board_cards: int = 4

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        if self.board not in BOARDS:
            raise ValueError(f"unknown board encoding {self.board!r}")
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")

    @property
    def num_bins(self) -> int:
        return self.n_bins if self.method in ("avg_1d",) + PER_HAND else self.n_bins ** 2

    @property
    def num_cards(self) -> int:
        return self.num_ranks * self.num_suits

    @property
    def board_dim(self) -> int:
        """Width of the board input: floats for "onehot", card ids for "ids", 0 for "none"."""
        k = self.board_cards
        if self.board == "onehot":
            return 2 * (k * self.num_ranks + k * (k - 1) // 2 + k) + k * k
        return 2 * k if self.board == "ids" else 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**d)


# Equity bins ------------------------------------------------------------------------------


EDGE = 1e-4  # see _axis


def _axis(x: np.ndarray, n: int) -> np.ndarray:
    """Bin of each equity. Equities are small fractions, so many sit exactly on a bin edge
    (0.3, 0.5), where float32 (the shards) and float64 (a solver's equities) round to
    different sides: on the small deck 22% of the hands landed in another joint_2d bin at
    solve time than in training. Equities are compared in float32 and an edge counts as
    reached from EDGE / n below it, so both paths agree."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    return np.clip(np.floor(x * n + EDGE).astype(np.int64), 0, n - 1)


def bin_index(equity: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(..., H, 2) equities -> (..., H) bin of each hand."""
    equity = np.asarray(equity)
    n = cfg.n_bins
    if cfg.method in PER_HAND:  # every hand its own bin (n_bins = H): no abstraction, one hand list only
        if equity.shape[-2] != n:
            raise ValueError(f"method 'hand' needs n_bins = the number of hands ({equity.shape[-2]})")
        return np.broadcast_to(np.arange(n), equity.shape[:-1]).copy()
    if cfg.method == "avg_1d":
        return _axis(np.asarray(equity, dtype=np.float32).astype(np.float64).mean(axis=-1), n)
    return _axis(equity[..., 0], n) * n + _axis(equity[..., 1], n)


def histogram(rng: np.ndarray, bins: np.ndarray, num_bins: int, possible: np.ndarray | None = None) -> np.ndarray:
    """(..., H) range mass and (..., H) bins -> (..., num_bins) mass per bin. Hands where
    `possible` is False count for nothing, whatever their range says."""
    rng = np.asarray(rng, dtype=np.float64)
    if possible is not None:
        rng = np.where(possible, rng, 0.0)
    rng, bins = np.broadcast_arrays(rng, bins)
    lead = rng.shape[:-1]
    n = int(np.prod(lead))
    flat = bins.reshape(n, -1) + np.arange(n)[:, None] * num_bins
    out = np.bincount(flat.ravel(), weights=rng.reshape(n, -1).ravel(), minlength=n * num_bins)
    return out.reshape(lead + (num_bins,))


def bin_means(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, num_bins: int) -> tuple:
    """Weighted mean of `values` (..., H) in each bin, with `weights` (..., H) (a range):
    returns (means, mass), each (..., num_bins); means are 0 in bins with no mass."""
    mass = histogram(weights, bins, num_bins)
    total = histogram(np.asarray(weights, dtype=np.float64) * values, bins, num_bins)
    return np.divide(total, mass, out=np.zeros_like(total), where=mass > 0), mass


def to_hands(bin_values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """(..., num_bins) per-bin values and (..., H) bins -> (..., H): each hand its bin's value."""
    bin_values, bins = np.asarray(bin_values), np.asarray(bins)
    lead = np.broadcast_shapes(bin_values.shape[:-1], bins.shape[:-1])
    bv = np.broadcast_to(bin_values, lead + bin_values.shape[-1:])
    b = np.broadcast_to(bins, lead + bins.shape[-1:])
    return np.take_along_axis(bv, b, axis=-1)


# Per-hand descriptors ---------------------------------------------------------------------


HAND_FEATURES = 11


def hand_cards(hands: np.ndarray) -> np.ndarray:
    """(..., H, 2) card ids with each hand's higher card id first (card ids are rank-major),
    so a hand's two cards have one order."""
    return -np.sort(-np.asarray(hands, dtype=np.int64), axis=-1)


def hand_features(equity: np.ndarray, hands: np.ndarray, boards: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Per-hand descriptors, (..., H, HAND_FEATURES) float32, from `equity` (..., H, 2), the
    hand list `hands` ((H, 2) or (..., H, 2) card ids) and `boards` (..., 2, k):
      0-1   turn-end equity on board A and B
      2     1 if the hand can be held (neither card on a board)
      3-6   for the hand's higher card then its lower card, on board A then B: the share of
            the board's cards with the card's rank
      7-10  the same for the card's suit
    Everything is relative to the boards; which hand is which only enters through the cards."""
    equity, boards = np.asarray(equity), np.asarray(boards, dtype=np.int64)
    lead = equity.shape[:-2]
    c = np.broadcast_to(hand_cards(hands), lead + equity.shape[-2:-1] + (2,))
    S, k = cfg.num_suits, boards.shape[-1]
    b = boards[..., None, None, :, :]                          # (..., 1, 1, 2, k)
    x = c[..., None, None]                                     # (..., H, 2, 1, 1)
    possible = ~(x == b).any(axis=(-3, -2, -1))
    rank = ((x // S) == (b // S)).sum(-1) / k                  # (..., H, 2 cards, 2 boards)
    suit = ((x % S) == (b % S)).sum(-1) / k
    H = c.shape[-2]
    return np.concatenate([equity, possible[..., None], rank.reshape(lead + (H, 4)),
                           suit.reshape(lead + (H, 4))], axis=-1).astype(np.float32)


# Boards, pot and stack --------------------------------------------------------------------


def canonical_boards(boards: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(..., 2, k) card ids with each board's flop (all but its last card) sorted by rank
    then suit, high first, and the turn card last. Card order within a flop carries no
    information, so this makes the encoding independent of it."""
    boards = np.asarray(boards, dtype=np.int64)
    flop = -np.sort(-boards[..., :-1], axis=-1)  # card id order is rank-major
    return np.concatenate([flop, boards[..., -1:]], axis=-1)


def board_features(boards: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(..., 2, k) card ids -> (..., board_dim) floats: per board, each card's rank one-hot,
    same-suit indicators for every pair of its cards, and the one-hot of its largest suit
    count (1..k); then same-suit indicators between every card of board A and of board B."""
    b = canonical_boards(boards, cfg)
    k, R, S = cfg.board_cards, cfg.num_ranks, cfg.num_suits
    if b.shape[-2:] != (2, k):
        raise ValueError(f"boards must be (..., 2, {k})")
    rank, suit = b // S, b % S
    lead = b.shape[:-2]
    iu = np.triu_indices(k, 1)
    parts = []
    for side in (0, 1):
        r, s = rank[..., side, :], suit[..., side, :]
        parts.append(np.eye(R)[r].reshape(lead + (k * R,)))
        same = s[..., :, None] == s[..., None, :]
        parts.append(same[..., iu[0], iu[1]].astype(float))
        counts = (s[..., :, None] == np.arange(S)).sum(axis=-2).max(axis=-1)
        parts.append(np.eye(k)[counts - 1])
    cross = suit[..., 0, :, None] == suit[..., 1, None, :]
    parts.append(cross.reshape(lead + (k * k,)).astype(float))
    return np.concatenate(parts, axis=-1)


def board_ids(boards: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(..., 2, k) card ids -> (..., 2k) canonical card ids, for an embedding."""
    b = canonical_boards(boards, cfg)
    return b.reshape(b.shape[:-2] + (2 * cfg.board_cards,))


def encode_boards(boards: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """The board input of the configured kind: float32 features, int64 ids, or (..., 0)."""
    boards = np.asarray(boards)
    if cfg.board == "onehot":
        return board_features(boards, cfg).astype(np.float32)
    if cfg.board == "ids":
        return board_ids(boards, cfg)
    return np.zeros(boards.shape[:-2] + (0,), dtype=np.float32)


NUM_SCALARS = 2


def pot_features(pot, stack) -> np.ndarray:
    """(...,) pot and chips behind -> (..., 2): log pot and log(1 + stack / pot)."""
    pot, stack = np.asarray(pot, dtype=np.float64), np.asarray(stack, dtype=np.float64)
    return np.stack([np.log(pot), np.log1p(stack / pot)], axis=-1).astype(np.float32)


def normalize(rng: np.ndarray, possible: np.ndarray | None = None) -> tuple:
    """A range over the possible hands, normalized to sum 1, and its mass before. A range with
    no mass becomes uniform over the possible hands (or over all hands if none is marked),
    so the network always sees a valid range. Works on (..., H)."""
    rng = np.asarray(rng, dtype=np.float64)
    if possible is None:
        possible = np.ones(rng.shape, dtype=bool)
    possible = np.broadcast_to(possible, rng.shape)
    rng = np.where(possible, rng, 0.0)
    mass = rng.sum(axis=-1, keepdims=True)
    fallback = np.where(possible.any(axis=-1, keepdims=True), possible, True).astype(float)
    fallback /= fallback.sum(axis=-1, keepdims=True)
    out = np.where(mass > 0, rng / np.where(mass > 0, mass, 1.0), fallback)
    return out, mass[..., 0]
