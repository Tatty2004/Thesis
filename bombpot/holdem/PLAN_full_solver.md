# Plan: the full double-board Hold'em solver (option c)

Written 2026-10-06 as a handoff for a Claude instance picking up this work. First read the root `CLAUDE.md` and `bombpot/holdem/CLAUDE.md`. They hold the project rules and conventions, the status of steps R1 to R5, and the open decisions.

## Goal (agreed with the user on 2026-10-06)

Option (c) has three parts:

1. **A full-game strategy.** For one pair of flops on the 52-card deck, a strategy for the flop, turn and river.
   - Built with the sampled R5 solver (`core/solvers/sampled_cfr.py`), using river equity buckets, hundreds per turn deal.
   - Its exploitability is measured exactly in the full game, with no grouping.
2. **Exact river re-solving.** For any runout and betting history, re-solve the river exactly from the ranges the full-game strategy brings there.
   - Use *safe* subgame solving, so the combined strategy is never more exploitable than the strategy it started from.
3. **The thesis experiment at the river.** It measures three things:
   - how much the river grouping costs;
   - whether 2D grouping beats 1D;
   - how much re-solving recovers.

   Run it first on a smaller deck, where the exact no-grouping solution can be computed, then on the real deck.

## Fixed decisions (don't revisit without the user)

- **Betting:** limit, ante 1, bets [2, 4, 4], at most a bet and one raise per street. This is the `Holdem` default.
- **The turn stays exact.** Only the river is grouped.
- **Stack:** the package is `bombpot`, and numba is part of the stack.
- **River grouping methods:**
  - `avg_1d`: the baseline.
  - `product`: each board's equity bucketed on its own.
  - `kmeans_2d`: joint (equity A, equity B).

  EMD applies to the turn (histograms over river outcomes), not to the river.

## Working rules (the user's own; see the `verify-before-building` memory)

- **Verify before building.** Finish and verify each step before starting the next. Verification means agreement with an exact, independent reference: the LP, OpenSpiel, the tree solver, the full-width range solver, or brute force.
- **Stop and ask before:**
  - adding a dependency;
  - deciding anything under "Open decisions";
  - using cluster time beyond a smoke test;
  - changing a test that failed. Say why the change is justified, and show the evidence.
- **Commits and experiments:**
  - One commit per working step, pushed to `origin`.
  - Run experiments only from a clean commit, through `bombpot.experiments.run`.
  - Report results faithfully, failures included.
- **Docs:** after each step, update the Status section of `bombpot/holdem/CLAUDE.md`. Explain things to the user in plain language.

## Lessons already learned (read before coding)

- **Exact ties.** Where a regret is exactly zero by symmetry, two correct implementations get 0 and ±1e-19, and regret matching then splits their runs. So compare solvers in lockstep (the same strategy in, the same regrets out), not by letting both run freely.
- **numba kernels.** Write new kernels with `core/jit.kernel`, which picks a parallel or serial variant by problem size.
  - numba's on-disk cache ignores the `parallel` flag, so the serial variant is compiled from a renamed copy.
  - `tests/test_kernels.py` checks that the two variants agree bit for bit. Extend it for every new kernel, because small tests only ever run the serial variants.
- **Measured costs:**
  - A real-deck R4 iteration takes about 2.7 s and about 4.5 GB.
  - The exact streamed evaluation of a real-deck three-street strategy costs about 12.5 ms per river deal. There are 3.9M river deals, so that's about 14 h on 8 cores.
  - The evaluation peaks at about 6 GB with chunk 256. Chunk 8192 ran out of memory.
- **`SampledRangeCFR.log_row` runs the full exact evaluation.** Never log by default on the real deck.
- **`HoldemRiver`** needs fixed flops, a shared deck and board cards (k, 1, 1).
- **The river abstraction has imperfect recall.** Averages add up the own reach of all of an infoset's hands. That differs from the perfect-recall buckets in `RangeCFR`, which take the mean.
- **OpenSpiel's `universal_poker`** information string isn't perfect recall across rounds (see the root CLAUDE.md).

## Steps

### S1. Safe river re-solver (laptop)

**Read first:**
- Burch, Johanson and Bowling, "Solving Imperfect Information Games Using Decomposition" (AAAI 2014): the re-solving gadget.
- Moravčík et al., DeepStack (Science 2017): continual re-solving.
- Brown and Sandholm, "Safe and Nested Subgame Solving for Imperfect-Information Games" (NeurIPS 2017).

**Build** `core/solvers/resolve.py`. It must stay game-agnostic, using `range_cfr.street_reach`, `range_cfr.street_values` and a Street.

- **Inputs:**
  - one last-street deal (a CardStreet holding that deal), with its template and contributions;
  - both players' reach vectors at its root, for every entering line, from the full-game strategy;
  - the re-solving player p;
  - the opponent's counterfactual value per hand and line at the root under the starting strategy, computed with `street_values`.
- **The resolve gadget.** At the root, for each opponent hand and line, the opponent chooses between "follow" (enter the subgame) and "terminate" (take its starting counterfactual value).
  - Solve with DCFR.
  - Output p's river strategy at that deal for every line.
  - Re-solve each player in turn.
- Vectorize over the 25 entering lines. Optionally also vectorize over the river deals of one turn deal.

**Acceptance** (`tests/holdem/test_holdem_resolve.py`, on the small three-street deck that `test_holdem_river.py` uses):

1. **From an exact equilibrium:** full-width DCFR to at most 1e-8, or the LP where it fits. Re-solving every river keeps exploitability within the start's plus the re-solve tolerance, and the game values agree.
2. **Safety:** from imperfect strategies (bucketed, or few iterations), re-solving every river never raises the exact exploitability beyond the re-solve tolerance.
3. **Per hand:** at the re-solved root, the opponent's best-response counterfactual value is at most its starting value plus the tolerance.
4. **Optional baseline:** unsafe re-solving without the gadget, to show the difference.

### S2. River groupings (laptop)

- Add `product` to `core/abstraction/river.py`: √k centers per board, from 1D k-means on that board's equity. The id is the pair of board buckets.
- Support k in the hundreds. Fitting per turn deal must stay fast, for example with vectorized k-means or quantile initialization. Measure it.
- Optional, so ask first: features beyond equity against a uniform range, such as hand strength against clusters of opponent hands (Johanson et al. 2013).

**Acceptance:**
- **Buckets are well formed:** ids are in range, impossible hands get the spare id, and centers are ordered weakest first.
- **Buckets are deterministic** for a given seed.
- **Evaluation matches:** the streamed evaluation of a grouped strategy equals the full-width evaluation of its lift (extend `test_holdem_river.py`).

Note: no grouping that shares a row across river cards can reproduce the exact game. The exact reference is `Lossless` (a row per river deal), which is feasible only on small decks.

### S3. Split exact evaluation (laptop, then Ionic)

- Split `exact_exploitability` into shard jobs over turn deals. Each writes its partial `agg` sums for BR0, BR1 and the game value.
- A combine step adds the shards and finishes the upper streets.
- The river strategy comes from a provider function: either a blueprint lookup or a per-river-deal re-solve. That lets S6 grade the combined strategy.

**Acceptance:** on the small deck, combined shards equal the unsharded evaluation exactly, with both providers.

**Stop here and report to the user before any cluster work.**

### S4. Ionic setup (needs the user's access)

- The user gets access to Ionic, the CS department's Slurm cluster. Ask them for:
  - node memory and cores;
  - which partitions and GPUs they can use;
  - storage paths.
- **Environment:** a Python 3.11 venv, `pip install -e . --no-deps`, plus numpy, scipy, numba, POT and pyyaml. open_spiel is needed only for tests.
- **Slurm job scripts** in `scripts/ionic/`:
  - bucket fitting, as an array job over turn-deal shards;
  - the sampled solve, as one multi-core job;
  - the sharded evaluation, as an array job;
  - river re-solving, as an array job;
  - the combine step.

  Set `NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK` and a writable `NUMBA_CACHE_DIR`.

**Acceptance:** the small-deck pipeline on Ionic reproduces the laptop numbers exactly.

### S5. Exact river experiment on a smaller deck (Ionic)

- **Deck,** chosen by node memory:
  - 6 ranks × 4 suits: about 20–40 GB of storage. The current full-width `RangeCFR` may fit with headroom.
  - 7 × 4: about 75–150 GB. Needs a full-width solver that streams the river by turn deal.
- **Exact reference:** full-width DCFR with no grouping, run to a small exploitability. Record the convergence curve.
- **Grid** (confirm with the user):
  - methods `avg_1d`, `product` and `kmeans_2d`;
  - k in {25, 50, 100, 200, 400}, capped by the number of hands per river deal;
  - 3 seeds;
  - several flop pairs.
- **For each configuration:**
  1. Run the sampled solve, then measure its exploitability exactly (total error).
  2. Re-solve every river (S1), then measure exactly again.
- **Output,** through the experiment runner: a table and a plot of total error against k for each method, with and without re-solving, compared with the exact solution. Don't assert that error falls with k; log it either way.

**Stop and report the results to the user.**

### S6. Real-deck run (Ionic)

- Start with the flops Ah7c2d / KsKd9h:
  1. Fit buckets, with k chosen from S5.
  2. Pilot the sampled solve: measure seconds per iteration, then choose the batch size, sample count and iteration count.
  3. Run the sharded exact evaluation.
  4. Re-solve every river, batched by turn deal (an estimated few hours across about 100 jobs).
  5. Evaluate the combined strategy exactly.
- Add more flop pairs only if the user agrees to the cluster time.

### S7. Runout lookup (laptop)

A small CLI or function. Given a flop pair, a turn, a river and a betting history, it returns the exact river strategy (from S1), starting from a saved full-game strategy.

## Open decisions (ask the user)

1. Ionic's specs, which set the S5 deck size.
2. The S5 grid: methods, k values, seeds and flop pairs.
3. For S6, how many flop pairs.
4. Whether to add richer river features (the optional part of S2).

## Literature

- **Pluribus** ([supplementary materials](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf)):
  - Precomputed strategy: lossless on the first round, then 200 buckets per later round, from k-means on hand features.
  - Real-time search: 500 buckets for later rounds, computed separately for each flop.
  - 1 to 14 raise sizes depending on the spot, all fractions of the pot.
  - Training: linear MCCFR.
  - In search: a vector form of Linear CFR that samples one set of public board cards per thread.
- **Public chance sampling** ([Johanson et al. 2012](https://bowlingmh.github.io/papers/12aamas-pcs.pdf)): sampled values are importance-weighted by 1/q, and averages are accumulated with own reach at the sampled infosets only.
- **How commercial PLO solvers cope** ([GTO Wizard](https://blog.gtowizard.com/how-gto-wizard-solved-plo/)):
  - MonkerSolver abstracts the later streets.
  - GTO Wizard solves one street at a time, with neural-network values for later streets.
