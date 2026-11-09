# Plan: the value-network solver for real double-board Hold'em

Written 2026-11-09 after M1. Read the root `CLAUDE.md` and `bombpot/holdem/CLAUDE.md` first. This
plan replaces the river half of PLAN_full_solver.md (M1's sampled river with equity buckets) and
adds the real betting game. Steps B1-B2 and S1, S3 of that plan live on here under new names.

## Why (decided with the user, 2026-11-09)

M1 solved one flop pair in 3 days to exploitability 0.039 chips, with a 7 GB checkpoint and a 21 h
exact check. Profiling showed 80% of an iteration in single-threaded numpy (river bucket lookup
and scatter), but the deeper problem is the design: the river (3.9 M runouts per flop pair) is
sampled and abstracted inside the solve, so every flop pair costs days, and there are 9,273,888
strategically distinct flop pairs (22,100 x 18,424 / 2 unordered pairs, modulo the 24 suit
permutations; the single-board count is 1,755). No tabular design covers that.

**The new design** cuts the game at the end of the turn, as DeepStack does:

1. **A counterfactual value network.** Input: the two four-card boards, the pot and stacks, and both
   players' ranges in a board-relative representation. Output: each hand's counterfactual value for
   both players at the end of the turn, averaged over the river runouts. Trained by supervised
   regression on river subgames that our own exact solver solves.
2. **An exact flop+turn solve per flop pair** with the network as the leaf value at every turn-end
   state. Minutes to an hour per flop pair, for any flop pair.
3. **Exact river re-solving.** A river reached in play is solved exactly from the ranges the
   flop+turn strategy brings there, with the safe gadget (Burch et al. 2014; Brown and Sandholm
   2017), so re-solving never raises exploitability.
4. **Exact grading.** The flop+turn strategy is explicit and the river re-solve is deterministic, so
   the streaming exact evaluator (with a river provider) grades the combined strategy on held-out
   flop pairs. The network's error shows up there as a number.

Neural methods are allowed from here on (the user lifted the rule on 2026-11-09). Everything else
in the project's working rules still holds: verify each step against an exact reference before
building on it.

## The game (fixed, 2026-11-09)

- 52 cards, two boards, fixed flop pair, flop / turn / river (`board_cards = (3, 1, 1)`).
- **Antes 1 each** (pot 2), **100 behind** each (stack 101 including the ante).
- **No-limit** with the menu {1/2 pot, pot} for bets and raises. A bet of X pot is X times the
  current pot. A raise of X pot is: call, then raise by X times the pot after the call. A size at or
  above the remaining stack becomes **all-in**; there is no separate all-in action. Once a player is
  all-in there is no more betting: the remaining boards are dealt and the hand goes to showdown.
  Minimum raise: a raise must be at least the previous bet or raise on the street, otherwise only
  all-in is allowed (with this menu it never binds, but the engine enforces it so ACPC agrees).
- **Raise cap per street: 2 aggressive actions** (a bet and one raise), the same cap as the limit
  game. Counted (`scripts/count_sizes.py` must reproduce this once the engine exists):

  | aggressive actions per street | nodes | actions | lines into the river | turn-end leaves | flop+turn state (float64) |
  |---:|---:|---:|---:|---:|---:|
  | 2 (**default**) | 14 | 38 | 169 | 349,830 | about 24 GB |
  | 3 | 30 | 86 | 841 | 1,740,870 | about 119 GB |

  The cap is a config value. Cap 3 needs float32 state and a GPU port of the passes first.
- Player 0 acts first on every street. Two private cards each. Each board is half the pot.
- The limit game of M1 (`Holdem` defaults) must keep working unchanged: it is the baseline.

## Architecture and interfaces (agreed names, so parallel work fits together)

### A. Betting engine: `core/betting.py`

`PotPoker`, game-agnostic like `LimitPoker`, driven by the same event list. Parameters: `ante`,
`stack` (chips behind, per player), `sizes` (pot fractions for bets and raises, e.g. `(0.5, 1.0)`),
`max_aggressive` per street, `limit` in {"no-limit", "pot-limit"} (pot-limit caps a size at the
pot; identical for sizes <= 1). Actions are strings: `f`, `c` (check or call), `r<fraction>` (e.g.
`r0.5`, `r1`), and `a` (all-in, which replaces every size that reaches the stack; at most one `a`).
Reuses `PokerState` (contrib already carries the chips; stack remaining = ante + stack - contrib).
`LimitPoker` is untouched. `Holdem` takes `betting="limit" | "pot"` plus the engine's parameters,
with the limit defaults unchanged.

Acceptance (B1): on tiny decks, DCFR on the compiled tree equals the LP value; `game_diff` against
OpenSpiel `universal_poker` no-limit on one board for the same action sequences (payoffs, legal
actions, chance); hand-checked all-in and min-raise cases; every existing test passes.

### B. Public tree with line classes: `core/public.py`, `core/solvers/range_cfr.py`

Today every line of a street shares one betting template. With pot-fraction sizes and stacks the
*shape* of a street's betting depends on the line (all-in truncation), and the chips always do. So:

- Each street's lines are grouped into **classes** with identical betting shape. `pt.classes[t]` is a
  list of `LineClass(template: Template, lines: int array, contrib: (L_c,) chips each player has in,
  stack: (L_c,) chips each has behind, parent_line: (L_c,), parent_close: (L_c,))`. A class may have
  a template with **zero nodes and one close** (all-in: no betting; at the last street a showdown).
- Arrays are per class: `sigma[t][c][k]` shaped `(D_t, L_c, H, A)`. `forward`, `backward`,
  `street_reach`, `street_values`, `next_root`, `close_values` loop over classes; `next_root` gathers
  from the parent class's closing reach via `parent_line`/`parent_close`, and `close_values`
  scatters back the same way.
- A limit game has exactly one class per street, and its results must be **bit-identical** to
  today's (`tests/test_range_cfr.py`, `tests/holdem/*` unchanged in substance; `tree_coordinates`,
  `to_tree_sigma`, `from_tree_sigma`, `sampled_cfr.py`, `experiments/` adapted to the new indexing).

Acceptance (B2): lockstep agreement with the tree solver on small decks for the pot game (full
and bucketed); LP value matches; all-in lines verified by hand on a tiny deck; limit game
bit-identical.

### C. Leaf values at the turn: `core/solvers/range_cfr.py`

`backward(pt, hsig, reach, p, mode, leaf=None)`. When `leaf` is given, `pt` holds the flop and turn
only and the turn's closing lines take their values from the provider instead of a showdown:

```
leaf.values(p, t, closing)  ->  agg
```

`closing[c]` is the turn street's closing reach for class `c`: a tuple `(reach0, reach1)` each shaped
`(D, L_c, closes_c, H)` (so `forward` must return the last street's closing reach). The result `agg[c]`
is shaped `(D, L_c, closes_c, H)`: player p's counterfactual value per hand **in the solver's units**
(chance weight x opponent reach x payoff, exactly like a showdown terminal), so the rest of the
backward pass is unchanged. In mode `"br"` the leaf still returns equilibrium values (the leaf is a
value function, not a strategy; DeepStack does the same).

Providers:
- `ExactRiverLeaf` (small decks, tests): solves every (deal, line, close) river subgame with the
  subgame solver (D) to a tolerance, and returns its root values. The reference for everything.
- `NetLeaf` (F): normalizes the ranges, calls the network in one batch per street, and rescales
  the output by chance weight x opponent total reach x pot.

Acceptance (C1, small deck): with `ExactRiverLeaf`, a flop+turn DCFR solve, graded exactly with the
river played by the subgame solver (provider in E), reaches an exploitability within the leaf and
re-solve tolerances of the full-width lossless solve's. The provider path is tested for units by
giving it a leaf that replays the real river (`Lossless`), which must reproduce `backward` without
`leaf` to 1e-12.

### D. River subgame solver: `core/solvers/subgame.py`

```
solve_subgame(street, template, contrib, stack, reach0, reach1, variant="cfr+", iters, tol) -> Subgame
```

`street` is a `CardStreet` over the river deals of one turn deal (from `HoldemRiver.street` on all
children), `template` the river betting for this pot and stacks (from the engine), `reach0/reach1`
the root ranges `(H,)`. Returns root counterfactual values for both players `(H,)` each, in solver
units, the average strategy, and the final exploitability of the subgame (so every sample carries
its own error). Vectorize over river deals as the range solver does; this is R4-scale work per
subgame. This solver is also the real-time river solve (S7 of the old plan).

Acceptance: equals `RangeCFR` on the same subgame compiled as a game (small deck), CFV units checked
against `street_values`.

### E. Safe river re-solver and exact grading: `core/solvers/resolve.py`, `sampled_cfr.last_street_sums`

- `resolve(subgame inputs, opponent_cfv)` adds the gadget (the opponent chooses per hand between
  entering the subgame and taking the counterfactual value `opponent_cfv` it had under the
  blueprint) and solves with the subgame solver. Returns the re-solving player's river strategy.
- The exact evaluator takes a **river provider**: `provider(parents, children) -> hsig` for the last
  street, either a stored strategy (today) or a re-solve per (turn deal, line). `last_street_sums`
  must group chunks by turn deal so one re-solve serves all of that deal's rivers.

Acceptance (small deck): re-solving every river from an exact equilibrium keeps exploitability
within tolerance; from a poor blueprint, safe re-solving never raises exact exploitability by more
than the tolerance, while unsafe re-solving can (report both); the provider path with a stored
strategy equals today's `exact_exploitability` bit for bit.

### F. Value network: `bombpot/valuenet/` (a layer: imports `core` only), data in `experiments/valuenet/`

- **Featurization** (`valuenet/features.py`): a range `(H,)` becomes a histogram over
  **equity bins**: `avg_1d` (bins over the mean of the two boards' turn-end equities) and `joint_2d`
  (a grid over (equity A, equity B)); `n_bins` configurable (e.g. 50 and 20 x 20). Turn-end equity
  per hand per board = average showdown share over the river runouts against a uniform range,
  computed once per turn deal. Optional extra dims: per-hand equity spread over rivers.
- **Model** (`valuenet/model.py`, PyTorch): MLP on [board encoding, pot/stack normalized, both
  range histograms] -> per-bin counterfactual values for both players as fractions of the pot, with
  a zero-sum correction layer (DeepStack). Hands take their bin's value.
- **Data** (`experiments/valuenet/datagen.py`): random flop pair, random turn cards, random pot
  and stacks from the lines the engine can produce, random ranges by recursive random splitting
  (DeepStack's recipe) plus ranges taken from actual flop+turn solves once those exist; solve the
  river subgame (D) to tolerance; store in **one HDF5 shard per job** (inputs, targets, boards, pot,
  the subgame's own exploitability). Slurm array job, 8-16 CPUs each. **Measure the cost per sample
  on the real deck first**, and size the dataset from it (budget: 3,000 CPU-hours, 40 jobs at once,
  agreed 2026-11-09).
- **Training** (`experiments/valuenet/train.py`): Huber loss on pot-normalized values, weighted by
  range mass; held-out flop pairs; one GPU job (`scripts/ionic/valuenet_train.sbatch`).
- `NetLeaf` implements C's interface with batched inference (CPU or GPU).

Acceptance: on the small deck, a network trained on that deck's subgames gives a flop+turn solve
whose exact exploitability (river re-solved) is reported next to C1's exact-leaf number: that gap
is the network's cost and the first scientific result. Report it whatever it is.

### G. Real deck

1. Data generation and training (F) on the real deck. 2. Flop+turn solve for Ah7c2d / KsKd9h with
`NetLeaf`, timed. 3. Exact grading with the river re-solved (E), sharded. 4. Three random held-out
flop pairs. 5. The M1 limit baseline on the same pairs for comparison, if time allows.

## Environments

- Tests and tabular code: `.venv` (Python 3.13, no torch) as before.
- Anything with torch: `/n/fs/scratch/rb5709/envs/torch/bin/python` (torch, numba, numpy, scipy,
  POT, h5py, pytest; `bombpot` installed editable). Home has a 5 GB quota, so torch lives on scratch.
  `scripts/ionic/env.sh` takes `PY=` to pick it. Net tests skip when torch is missing.
- GPUs on Ionic's `all` partition: A100, L40, A40, A6000, A5000, RTX 3090/6000/2080, GTX 1080
  (`sinfo -o "%P %G"`). Request one with `--gres=gpu:1`.

## Parallel streams and gates

The streams below run in parallel in separate git worktrees off branch `valuenet`, each on its own
branch (`vn-engine`, `vn-subgame`, `vn-resolve`, `vn-net`), merged back by the coordinating session
in the order engine -> subgame -> resolve -> net. Each stream keeps a short `STATUS.md` section in
its commit messages and reports measured numbers, not estimates.

| Stream | Builds | Gate |
|---|---|---|
| engine | A, then B | A: tree-solver DCFR = LP on tiny pot games; ACPC agreement; all tests pass. B: limit game bit-identical; pot game lockstep with the tree solver; LP matches |
| subgame | D, C (hook + `ExactRiverLeaf`), the data generator skeleton, the real-deck sample cost | D's equality with `RangeCFR`; C1 on the small deck; a measured seconds-per-sample on the real deck at a stated tolerance |
| resolve | E | the three safety checks on the small deck; provider path bit-identical with a stored strategy |
| net | F (features, model, training, `NetLeaf`) against the data spec, synthetic data until real shards exist | unit tests; a training run on small-deck shards once subgame delivers them; then F's acceptance |

Until B lands, subgame, resolve and net develop on the **limit** small deck (5 ranks x 3 suits,
flops 2c3c4d / 5c6d4c, as in `tests/holdem/test_holdem_river.py`); their code must not assume one
class per street beyond what `pt.classes` gives them.

## Working rules for this plan

- Verify before building: the exact references are the tree solver, the LP, the full-width lossless
  range solve, `universal_poker`, and brute force.
- Compare solvers in lockstep, not free-running (exact ties, see PLAN_full_solver.md).
- New numba kernels go through `core/jit.kernel` and get a parallel/serial agreement test.
- Nothing large in git. Data, checkpoints and models go under `results/valuenet/` (scratch).
- One commit per working step. Pushing needs the user's SSH key (see M1_STATUS.md 03:20).
- Report failures as failures.
