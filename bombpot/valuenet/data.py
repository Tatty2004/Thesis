"""Training data: HDF5 shards (one per generator job) read lazily in row blocks and
featurized on the fly; a held-out split by flop pair; and synthetic shards for tests.

Shard layout (experiments/valuenet/README.md is the authority; N samples, H hands):
  flops (N, 2, 3) int, turn (N, 2) int    card ids of each board's flop and turn card
  hands (H, 2) int                        the hand list (metadata here: nothing below uses it)
  range0, range1 (N, H) float32           normalized to 1 over the hands possible at the deal
  equity (N, H, 2) float32                turn-end equity on board A and B vs a uniform range
  pot, stack (N,) float32                 chips in the pot, chips behind (each)
  cfv0, cfv1 (N, H) float32               root counterfactual values in pots
  subgame_exploitability (N,) float32, iters (N,) int

Targets. The network predicts one value per equity bin and player. A bin's target is the
range-mass-weighted mean of its hands' values for that player,
    target_p[b] = sum_{i in b} range_p[i] cfv_p[i] / sum_{i in b} range_p[i]   (0 if no mass),
and the loss weights bin b of player p by that mass, histogram_p[b]. So empty bins and
impossible hands (range 0) carry no gradient, the bin targets satisfy the zero-sum identity
exactly when the hand values do (sum_b h_p[b] target_p[b] = sum_i range_p[i] cfv_p[i]), and
the weighted loss is the per-hand loss minus the within-bin spread, which no per-bin output
can remove (reported as the floor of the per-hand error).

Split. A sample belongs to the held-out set when the hash of its flop pair (each flop sorted,
the two flops sorted, so board order doesn't matter) falls below `holdout_fraction`. All
samples of a flop pair land on the same side, whichever shard holds them.

h5py is imported inside the functions that need it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from bombpot.valuenet.features import (FeatureConfig, bin_index, bin_means, encode_boards, histogram,
                                       pot_features)

FIELDS = ("flops", "turn", "range0", "range1", "equity", "pot", "stack", "cfv0", "cfv1")


def flop_pair_hash(flops: np.ndarray, salt: str = "valuenet-split") -> np.ndarray:
    """(N, 2, 3) flops -> (N,) numbers in [0, 1), the same for every sample of a flop pair."""
    f = np.sort(np.asarray(flops, dtype=np.int64), axis=-1)
    out = np.empty(len(f))
    for n, (a, b) in enumerate(f):
        key = sorted([tuple(a), tuple(b)])
        h = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
        out[n] = int.from_bytes(h, "little") / 2.0 ** 64
    return out


def heldout_mask(flops: np.ndarray, fraction: float, salt: str = "valuenet-split") -> np.ndarray:
    return flop_pair_hash(flops, salt) < fraction


def featurize(raw: dict, cfg: FeatureConfig) -> dict:
    """Raw shard rows (FIELDS) -> network inputs, bin targets and weights:
    board, scalars, h0, h1, t0, t1 (float32, (n, B) for the bin arrays), plus bins (n, H)
    and the raw ranges and values, for the per-hand error."""
    boards = np.concatenate([raw["flops"], raw["turn"][..., None]], axis=-1)
    bins = bin_index(raw["equity"], cfg)
    B = cfg.num_bins
    out = {"board": encode_boards(boards, cfg), "scalars": pot_features(raw["pot"], raw["stack"]), "bins": bins}
    for p in (0, 1):
        r, v = raw[f"range{p}"], raw[f"cfv{p}"]
        t, h = bin_means(v, r, bins, B)
        out[f"h{p}"], out[f"t{p}"] = h.astype(np.float32), t.astype(np.float32)
        out[f"range{p}"], out[f"cfv{p}"] = r.astype(np.float32), v.astype(np.float32)
    return out


class ShardSet:
    """The *.h5 shards of one directory (not recursive), read in blocks of rows.

    Opens each file once and keeps only the flops (for the split) in memory; everything else
    is read block by block (`block` rows of every field, contiguous), so memory stays small
    and I/O stays block-oriented."""

    def __init__(self, directory, cfg: FeatureConfig, holdout_fraction: float = 0.1, block: int = 512,
                 salt: str = "valuenet-split"):
        import h5py

        self.cfg, self.block = cfg, block
        self.paths = sorted(Path(directory).glob("*.h5"))
        if not self.paths:
            raise FileNotFoundError(f"no *.h5 shards in {directory}")
        self.files = [h5py.File(p, "r") for p in self.paths]
        self.heldout = [heldout_mask(f["flops"][:], holdout_fraction, salt) for f in self.files]

    def close(self) -> None:
        for f in self.files:
            f.close()

    def count(self, split: str) -> int:
        return int(sum((m if split == "heldout" else ~m).sum() for m in self.heldout))

    def _blocks(self):
        return [(i, a) for i, f in enumerate(self.files) for a in range(0, len(f["pot"]), self.block)]

    def read(self, i: int, a: int, split: str) -> dict | None:
        """Featurized rows [a, a + block) of shard i that belong to `split` ("train",
        "heldout" or "all"), or None if there are none."""
        f = self.files[i]
        b = min(a + self.block, len(f["pot"]))
        keep = self.heldout[i][a:b]
        if split == "train":
            keep = ~keep
        elif split == "all":
            keep = np.ones_like(keep)
        if not keep.any():
            return None
        raw = {k: f[k][a:b][keep] for k in FIELDS}
        return featurize(raw, self.cfg)

    def batches(self, split: str, batch_size: int, rng: np.random.Generator | None = None, buffer: int = 8):
        """Featurized batches. With `rng`: blocks in random order, `buffer` blocks at a time
        pooled and shuffled; without: in order."""
        blocks = self._blocks()
        if rng is not None:
            blocks = [blocks[j] for j in rng.permutation(len(blocks))]
        for s in range(0, len(blocks), buffer):
            parts = [x for x in (self.read(i, a, split) for i, a in blocks[s:s + buffer]) if x is not None]
            if not parts:
                continue
            pool = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
            n = len(pool["scalars"])
            order = rng.permutation(n) if rng is not None else np.arange(n)
            for c in range(0, n, batch_size):
                idx = order[c:c + batch_size]
                yield {k: v[idx] for k, v in pool.items()}


# Synthetic shards -------------------------------------------------------------------------


def synthetic_values(equity, range0, range1, pot, stack, boards, num_suits: int, num_ranks: int,
                     noise: float, rng: np.random.Generator):
    """A known smooth zero-sum function of (equities, ranges, pot, stack, board), in pots, for
    one sample: with m = mean equity and g = (eqA - eqB)^2 per hand,
        v_p[i] = s k (m[i] - E_opp[m]) + s / 2 (g[i] - E_opp[g]) + noise,
    s = 1/2 + stack / (2 (stack + pot)), k = 1 + 1/2 [board A paired] - 0.3 [board B has 3+ of
    a suit]; then shifted so that sum r0 v0 + sum r1 v1 = 0. Impossible hands get 0."""
    poss = (range0 > 0) | (range1 > 0)
    m, g = equity.mean(-1), (equity[:, 0] - equity[:, 1]) ** 2
    ranks, suits = boards // num_suits, boards % num_suits
    paired = len(set(ranks[0].tolist())) < len(ranks[0])
    flushy = np.bincount(suits[1], minlength=num_suits).max() >= 3
    s = 0.5 + 0.5 * stack / (stack + pot)
    k = 1.0 + 0.5 * paired - 0.3 * flushy
    out = []
    for opp in (range1, range0):
        v = s * k * (m - opp @ m) + 0.5 * s * (g - opp @ g) + noise * rng.standard_normal(len(m))
        out.append(np.where(poss, v, 0.0))
    err = range0 @ out[0] + range1 @ out[1]
    return [np.where(poss, v - err / 2, 0.0) for v in out]


def write_synthetic_shard(path, n: int, seed: int, num_ranks: int = 13, num_suits: int = 4,
                          flop_pairs: int = 20, noise: float = 0.02) -> None:
    """A shard in the real layout with synthetic equities, ranges and values
    (`synthetic_values`). Hands are the pairs of the deck's cards outside the flops, indexed
    as pairs of slots (so H = C(cards - 6, 2), 1,035 on 52 cards); each sample draws one of
    `flop_pairs` flop pairs and two turn cards, which make the hands through them impossible."""
    import h5py

    rng = np.random.default_rng(seed)
    C = num_ranks * num_suits
    slots = C - 6
    iu = np.triu_indices(slots, 1)
    H = len(iu[0])
    pairs = [rng.choice(C, 6, replace=False) for _ in range(flop_pairs)]
    data = {"flops": np.zeros((n, 2, 3), np.int16), "turn": np.zeros((n, 2), np.int16),
            "range0": np.zeros((n, H), np.float32), "range1": np.zeros((n, H), np.float32),
            "equity": np.zeros((n, H, 2), np.float32), "pot": np.zeros(n, np.float32),
            "stack": np.zeros(n, np.float32), "cfv0": np.zeros((n, H), np.float32),
            "cfv1": np.zeros((n, H), np.float32)}
    for s in range(n):
        six = pairs[rng.integers(flop_pairs)]
        rest = np.setdiff1d(np.arange(C), six)
        t = rng.choice(len(rest), 2, replace=False)
        data["flops"][s] = six.reshape(2, 3)
        data["turn"][s] = rest[t]
        poss = ~np.isin(iu[0], t) & ~np.isin(iu[1], t)
        x = rng.random(H)
        eq = np.stack([np.clip(x + 0.25 * rng.standard_normal(H), 0, 1),
                       np.clip(0.5 * x + 0.5 * rng.random(H) + 0.15 * rng.standard_normal(H), 0, 1)], -1)
        eq[~poss] = 0
        ranges = []
        for _ in (0, 1):
            r = poss * rng.random(H) ** rng.uniform(0.5, 3) * np.exp(rng.normal(0, 2) * x)
            ranges.append(r / r.sum())
        pot = float(2 * np.exp(rng.uniform(0, np.log(50))))
        stack = max(101.0 - pot / 2, 0.0)
        boards = np.concatenate([data["flops"][s], data["turn"][s][:, None]], -1).astype(np.int64)
        v0, v1 = synthetic_values(eq, ranges[0], ranges[1], pot, stack, boards, num_suits, num_ranks, noise, rng)
        data["equity"][s], data["range0"][s], data["range1"][s] = eq, ranges[0], ranges[1]
        data["pot"][s], data["stack"][s], data["cfv0"][s], data["cfv1"][s] = pot, stack, v0, v1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, chunks=True if v.ndim > 1 else None)
        f.create_dataset("hands", data=np.stack(iu, -1).astype(np.int16))
        f.create_dataset("subgame_exploitability", data=np.zeros(n, np.float32))
        f.create_dataset("iters", data=np.zeros(n, np.int32))
        f.attrs.update({"synthetic": True, "seed": seed, "num_ranks": num_ranks, "num_suits": num_suits,
                        "units_cfv": "pots (root value / pot)", "units_pot": "chips",
                        "units_stack": "chips behind, each", "units_exploitability": "pots"})
