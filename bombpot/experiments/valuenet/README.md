# Value-net experiments

## Safe turn re-solving (2026-11-22, branch vn-turn)

`turn_resolve.py --trunk DIR --flop-line r0.5c --turn 5h2c [--player 0] [--iters 300] [--cfv auto|avg|br]
[--simultaneous]` re-solves one turn root (one turn deal after one flop line) for one player
with the gadget of `core/solvers/leaf_resolve.py` (resolve.py's, one street up), the network
valuing every turn end exactly as the trunk does (a one-deal `NetLeaf`), and prints the
re-solved turn strategy. T values: `avg` = the trunk's turn-root values averaged over its
iterations (`real_trunk --turn-cfv` writes `turncfv.npy`, 445 MB for one flop pair), `br` = the
opponent's best response to the trunk's turn (needs only `strategy.npz`; one slice is read).

Real deck, trunk `real/attn_op1_r2_it300`, one GPU (`bench_turn_resolve.py`, 6 random roots x
both players, network on the GPU with AMP, the turn tree on the CPU), seconds per re-solve:

| GPU | 100 its alt | 300 its alt | 100 its sim | 300 its sim |
|---|---|---|---|---|
| RTX 2080 Ti | 1.8-2.1 | 5.4-6.2 | 1.0 | 3.1 |
| RTX 3090 | 1.6 | 4.9 | 0.9 | 2.7 |
| RTX A5000 | 1.8-2.3 | 5.3-6.8 | 1.0 | 3.0 |

Gadget exploitability (NashConv / mass, in pots of the turn root), median over roots: alt 1.5-1.9e-3
at 100, 6-7e-4 at 300; sim (simultaneous updates: one network pass per iteration) 2.5-2.9e-3 at
100, 7-8e-4 at 300 (sim needs about 1.5x the iterations for the same gap). 85-90% of the time
is the network: about 8.1 ms per leaf call of 13 rows on a 2080 Ti (forward alone 5.9 ms;
torch.compile reduce-overhead 5.2 ms, not adopted), two calls per alternating iteration; the
tree is about 2.8 ms per iteration on the CPU. GPU memory 62 MB, host RSS 1.6 GB; one-time
setup about 10 s (game, network, flop strategy), about 0.3 s per root (strategy slice, leaf).

## Real-deck solves and grades: use 300 iterations and re-solve tolerance 1e-3 (2026-11-18)

`real_trunk.py` now defaults to `--iters 300` and `real_grade.py` to `--tol 1e-3`. Measured on
Ah7c2d / KsKd9h with the attention HandSetNet retrained on round-1 on-policy data (exploitability,
chips per hand):

| trunk DCFR iterations | river re-solve tol 1e-2 | 3e-3 | 1e-3 |
|---|---|---|---|
| 100 (`real/attn_op1_r2`) | 0.0880 (full grade) | 0.044 (est.) | 0.032 (est.) |
| 300 (`real/attn_op1_r2_it300`) | 0.0783 (full grade) | | 0.023 (est.) |

- **The 1e-2 tolerance was most of the old grade (about 0.055 of 0.088).** At 1e-3 the gain
  flattens out. Cost per turn deal: 1e-2 about 8 GPU-min, 3e-3 about 11, 1e-3 about 17 (A5000 /
  A6000); a full 1e-3 grade is about 570 GPU-h. No re-solve hit the 5,000-iteration cap.
- **300 iterations buy about 0.010 at either tolerance.** The trunk's exploitability against
  the network is 0.0142 / 0.0070 / 0.0054 at 100 / 200 / 300 iterations (run 300 reproduces the
  100-iteration run's checks exactly), so more iterations buy little.
- What is left at 300 / 1e-3 (about 0.023) is mostly the network's error, plus the trunk's 0.005
  and a little re-solve slack.
- "est." = `real_grade_tol.py`: every 10th turn deal (207) re-solved at the tighter tolerance and
  substituted into the full grade, the change scaled by 10 (first order; the best response is a
  max, so the true value is probably a little higher). The script reproduces both full grades
  exactly before substituting. Results: `results/valuenet/real/tol_sweep.json`, `tol_it300.json`.

# Value-net training data: shard format

Written by stream `subgame` (2026-11-09). Producer: `bombpot/experiments/valuenet/datagen.py`;
read by stream `net` (`bombpot/valuenet/`, `experiments/valuenet/train.py`).

```
python -m bombpot.experiments.valuenet.datagen --n N --seed S --out FILE.h5 \
    [--tol 1e-3] [--iters 5000] [--template pot|limit] [--spread 2] [--variant dcfr|cfr+] [--deck 13,4] [--per-flop 1]
```
Slurm: `scripts/ionic/valuenet_datagen.sbatch` (array job, one shard per task, seed = task id).
Needs h5py: run with `/n/fs/scratch/rb5709/envs/torch/bin/python`.

## A sample
- **Flop pair**: two disjoint 3-card flops drawn uniformly from the deck (52 cards by default).
- **Turn**: one card per board, uniform over the rest.
- **River betting** (`--template pot`, the default: the real game of PLAN_valuenet.md, no-limit,
  1/2-pot and pot, 2 aggressive actions per street, ante 1 and 100 behind): a river line drawn
  uniformly from the real game's river lines that still have betting (`pt.classes[2]`, all-in
  classes left out: their value is a plain showdown), with pot P_l = 2 x its contrib; then
  P = P_l x exp(U(-ln s, ln s)), s = `--spread` (default 2; 1 keeps the real pots), with each
  player keeping at least 0.5 behind. The river template for that pot and stack is walked from
  the engine (a river state with both contributions set to P/2), and equals the real class
  template at the real pots (`tests/holdem/test_datagen.py`). `line` and `pot_spread` record
  where it came from. `--template limit`: P log-uniform on [2, 200], the limit river (bets of 4
  chips), the pot only setting the chips already in (the M1 baseline; pipeline tests).
- **Stack**: 101 - P/2, the chips **behind** for each player at the start of the river.
- **Ranges**: over the hands possible at that turn deal (no card shared with either board),
  zero elsewhere, normalized to sum 1. 80% recursive random splitting (DeepStack): the hands,
  sorted by turn-end equity (half the time) or shuffled, are split at a random point and each
  part gets a uniform random share of the mass, recursively; 10% near-uniform (1 + 0.1 U);
  10% very peaked (1 to H/20 hands).
- **Subgame**: the river below that turn deal (every ordered (river A, river B) pair), each
  player has P/2 in, river betting from the template, solved with DCFR (alternating; alpha 1.5, beta 0,
  gamma 2; `--variant cfr+` for CFR+, about 2.9x slower to the same tolerance) until its exploitability is <= tol pots (checked every 10 iterations) or
  `--iters` iterations.

## Files and hand lists

A file holds **many flop pairs**: by default every sample draws its own flop pair (`--per-flop K`
keeps a pair for K samples, each with its own turn). H is the same for every sample of a file
(all 2-card hands avoiding both flops: C(46, 2) = 1035 on 52 cards, 36 on the 5 x 3 test deck),
but which cards make up hand i depends on the flops. So `hands` (P, H, 2) holds one hand list
per flop pair, in `holdem_public_tree(game, upto=2).hands` order, and `hand_set[n]` says which
row sample n uses: hand i of sample n is `hands[hand_set[n], i]`. Every per-hand array
(`range0`, `cfv0`, `possible`, ...) is indexed by that i.

## Datasets (N samples)

| name | shape | dtype | meaning |
|---|---|---|---|
| `flops` | (N, 2, 3) | int8 | board A's and board B's flop cards, sorted |
| `turn` | (N, 2) | int8 | turn card of board A, of board B |
| `hands` | (P, H, 2) | int8 | one hand list per flop pair (above) |
| `hand_set` | (N,) | int32 | which row of `hands` sample n uses |
| `possible` | (N, H) | bool | hand avoids both boards (flops and turns); a possible hand can still have range 0 |
| `range0`, `range1` | (N, H) | float32 | ranges, zero where not possible, normalized to sum 1 |
| `equity` | (N, H, 2) | float32 | turn-end equity per hand per board (below) |
| `pot` | (N,) | float32 | P, chips in the pot at the start of the river (P/2 from each player) |
| `stack` | (N,) | float32 | chips **behind** for each player at the start of the river (101 - P/2) |
| `cfv0`, `cfv1` | (N, H) | float32 | root counterfactual values of the average strategy, **in pots** (below) |
| `br0`, `br1` | (N, H) | float32 | each hand's best-response root value against the opponent's average strategy, in pots (`br >= cfv`; equal up to the tolerance on hands the range holds). The values a leaf must hand out: right for hands with little or no range too (gate C1, 2026-11-12). Shards written before 2026-11-12 lack them |
| `zero_sum_residual` | (N,) | float32 | range0 . cfv0 + range1 . cfv1 (computed in float64; 0 up to rounding) |
| `subgame_exploitability` | (N,) | float32 | the solved subgame's exploitability, in pots |
| `iters` | (N,) | int32 | solver iterations used |
| `line` | (N,) | int32 | the real river line the betting came from (index among `pt.classes[2]`'s lines in street order; -1 for limit) |
| `pot_spread` | (N,) | float32 | P / P_l, the spread applied to that line's pot (1 for limit) |
| `seconds`, `mass` | (N,) | float32 | solve time; reach mass sum_ij range0[i] range1[j] [i, j disjoint] |
Cards: `card = rank * num_suits + suit`, ranks `23456789TJQKA`, suits `cdhs`.
File attributes: `num_ranks`, `num_suits`, `board_cards`, `num_boards`, `ante`, `stack`,
`bet_sizes`, `template`, `pot_range`, `tol_pots`, `iters_cap`, `variant`, `seed`, `git`,
`per_flop`, `card_encoding`, `n`, and `units` (the paragraph below).

## Units

The river subgame is solved with chance weight `river.scale` = 1 / (rest (rest - 1)) per valid
river deal (rest = cards left after both boards and both hands): the river chance only, with no
turn-deal or hand-pair factor, so for any hand pair its valid river deals' weights sum to 1.
cfv0 and cfv1 are the root values of **one** strategy profile, the subgame's average strategy:

    cfv_p[i] = sum over opponent hands j compatible with i (no shared card) of
               range_o[j] x E_river[payoff of p | hands i, j] / P

**Card removal:** the sum runs over the compatible j only and is **not** renormalized per hand
(it is not divided by the compatible mass sum_j compatible range_o[j]). Payoffs count all chips
(the P/2 each already has in included), so `zero_sum_residual` = sum_i range0[i] cfv0[i] +
sum_j range1[j] cfv1[j] = 0 up to rounding. Hands that can't be held have cfv 0.
`subgame_exploitability` = (br0 . range0 + br1 . range1) / 2 / mass / P.

`equity[n, i, b]` = hand i's mean showdown share on board b over the river deals it can see,
against a uniform opponent range (`CardStreet.equity` on the river street, averaged over the
deals where the hand is possible); 0 for impossible hands.

**Inverse transform** (what a leaf provider such as NetLeaf must do to return the range solver's
units, `range_cfr.backward`'s `leaf`). The leaf gets `closing[c] = (R0, R1)`, each (D, L_c, closes_c,
H), from `forward(pt, hsig, closing=True)`. At turn deal d, line l of turn class c, close j, with
pot P (twice the contrib of the river line it continues), the input ranges are R_p[d, l, j]
zeroed where not possible at d and normalized, and

    agg[c][d, l, j, i] = cfv_p[i] x P x (sum over possible h of R_o[d, l, j, h]) x turn.kappa[d]

with `turn.kappa[d]` = `holdem_public_tree(game, upto=2).streets[1].kappa[d]`, the turn street's
chance weight of one valid (hand0, hand1, turn deal), and **no other factor**: the real river's
chance weight per river deal is `turn.kappa[d] x river.scale`, and `river.scale` is already
inside cfv. (Not times the number of river deals: the sum over river deals is inside cfv too.)
P for a turn closing line is 2 x the `contrib` of the river line it continues (the river class
line with that `parent_class`, `parent_line`, `parent_close`).

Tested to 1e-12 in `tests/holdem/test_datagen.py`: for every line and close of three turn deals
of the small deck, exact cfvs in these units (a fixed river strategy, river-only chance weights,
normalized ranges), rescaled this way, reproduce the agg of a leaf that replays the real river
(the leaf that `tests/holdem/test_leaf.py` ties to the full three-street tree).
