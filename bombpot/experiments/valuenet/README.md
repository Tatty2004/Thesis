# Value-net training data: shard format

Written by stream `subgame` (2026-11-09). Producer: `bombpot/experiments/valuenet/datagen.py`;
read by stream `net` (`bombpot/valuenet/`, `experiments/valuenet/train.py`).

```
python -m bombpot.experiments.valuenet.datagen --n N --seed S --out FILE.h5 \
    [--tol 1e-3] [--iters 5000] [--template limit] [--variant dcfr|cfr+] [--deck 13,4] [--per-flop 1]
```
Slurm: `scripts/ionic/valuenet_datagen.sbatch` (array job, one shard per task, seed = task id).
Needs h5py: run with `/n/fs/scratch/rb5709/envs/torch/bin/python`.

## A sample
- **Flop pair**: two disjoint 3-card flops drawn uniformly from the deck (52 cards by default).
- **Turn**: one card per board, uniform over the rest.
- **Pot** P: log-uniform on [2, 200] chips; **stack** = 101 - P/2 each (chips behind; stored, not
  yet used by the betting).
- **Ranges**: over the hands possible at that turn deal (no card shared with either board),
  zero elsewhere, normalized to sum 1. 80% recursive random splitting (DeepStack): the hands,
  sorted by turn-end equity (half the time) or shuffled, are split at a random point and each
  part gets a uniform random share of the mass, recursively; 10% near-uniform (1 + 0.1 U);
  10% very peaked (1 to H/20 hands).
- **Subgame**: the river below that turn deal (every ordered (river A, river B) pair), each
  player has P/2 in, river betting from the template, solved with DCFR (alternating; alpha 1.5, beta 0,
  gamma 2; `--variant cfr+` for CFR+, about 2.9x slower to the same tolerance) until its exploitability is <= tol pots (checked every 10 iterations) or
  `--iters` iterations.
- **Template**: only `limit` exists until the no-limit engine (stream `engine`) is merged: the
  limit river (bet 4 chips, a bet and one raise), with the sampled pot only setting the chips
  already in. Shards made with it are for pipeline testing; real training data needs the
  no-limit template (pot-relative sizes, so values in pots generalize across P).

## Datasets (N samples, H hands per flop pair; H = 1035 on 52 cards)

| name | shape | dtype | meaning |
|---|---|---|---|
| `flops` | (N, 2, 3) | int8 | board A's and board B's flop cards, sorted |
| `turn` | (N, 2) | int8 | turn card of board A, of board B |
| `hands` | (P, H, 2) | int8 | the hand lists (all 2-card hands not on the flops, in `holdem_public_tree` order) |
| `hand_set` | (N,) | int32 | which row of `hands` sample n uses (one row per flop pair) |
| `range0`, `range1` | (N, H) | float32 | normalized ranges (above) |
| `equity` | (N, H, 2) | float32 | turn-end equity per hand per board (below) |
| `pot`, `stack` | (N,) | float32 | P and the stack behind, in chips |
| `cfv0`, `cfv1` | (N, H) | float32 | root counterfactual values, **in pots** (below) |
| `subgame_exploitability` | (N,) | float32 | the solved subgame's exploitability, in pots |
| `iters` | (N,) | int32 | CFR+ iterations used |
| `seconds`, `mass` | (N,) | float32 | solve time; reach mass sum_ij range0[i] range1[j] [i, j disjoint] |

Cards: `card = rank * num_suits + suit`, ranks `23456789TJQKA`, suits `cdhs`.
File attributes: `num_ranks`, `num_suits`, `board_cards`, `num_boards`, `ante`, `stack`,
`bet_sizes`, `template`, `pot_range`, `tol_pots`, `iters_cap`, `variant`, `seed`, `git`,
`per_flop`, `card_encoding`, `n`, and `units` (the paragraph below).

## Units

The river subgame is solved with chance weight kappa = 1 / (rest (rest - 1)) per valid river
deal (rest = cards left after both boards and both hands): the river chance only, with no
turn-deal or hand-pair factor, so for any hand pair the valid river deals' weights sum to 1.
With normalized ranges the values are then "values given the ranges":

    cfv_p[i] = sum_j range_o[j] [i, j disjoint] E_river[payoff of p | hands i, j] / P

with payoffs counting all chips (the P/2 each already has in included), so
sum_i range0[i] cfv0[i] + sum_j range1[j] cfv1[j] = 0 up to float32 rounding. Hands that can't
be held have cfv 0. `subgame_exploitability` = (br0 . range0 + br1 . range1) / 2 / mass / P.

`equity[n, i, b]` = hand i's mean showdown share on board b over the river deals it can see,
against a uniform opponent range (`CardStreet.equity` on the river street, averaged over the
deals where the hand is possible); 0 for impossible hands.

**Inverse transform** (what a leaf provider such as NetLeaf must do to return the range solver's
units, `range_cfr.backward`'s `leaf`): at turn deal d with closing reaches R0, R1 (unnormalized,
(H,) each) and pot P, zero R_o on hands impossible at d, normalize to get the input ranges, then

    agg_p = cfv_p x P x sum(R_o) x turn.kappa[d]

where `turn.kappa[d]` is the turn street's chance weight (`pt.streets[1].kappa`). This is
tested exactly in `tests/holdem/test_datagen.py` (a fixed strategy evaluated on the real river
with unnormalized reaches equals the transformed values).
