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

- **Betting: two games.**
  - **Limit:** ante 1, bets [2, 4, 4], at most a bet and one raise per street (the `Holdem` default). Used for the large grouping comparisons, where exact checks are cheap and the result is clean.
  - **Pot-limit with real sizes** (decided 2026-10-27), for a few headline solves:
    - antes 1 each (a pot of 2) and stacks of 20 behind, a stack-to-pot ratio of 10;
    - bets and raises of ½ pot or pot;
    - at most a bet and two raises per street (provisional: confirm in B3).

    It's built in steps B1–B5.
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

## Ionic (measured 2026-10-27)

- **Access:** log in at `cycles.cs.princeton.edu`. Jobs go through Slurm, under account `allcs`.
- **Partitions:**
  - `cs`: 6 nodes, 104 threads and 503 GB each.
  - `all`: 50 nodes, 40–128 threads and 220–503 GB.
  - `lowprio`: the same nodes as `all`, but jobs can be preempted.

  Jobs are limited to 7 days. CPU-only jobs that asked for `cs` landed in `all`.
- **Slurm CPUs are hyperthreads.** `--cpus-per-task=8` gets 4 physical cores. R4 took 5.6 s per iteration there (2019 Xeon Gold 5218), against 2.7 s on the laptop. Request more CPUs, sized from the S0 scaling test.
- **Python:** the system Python is 3.9. Modules provide 3.13 (anaconda3/2025.12), and `.venv` uses 3.13.9, not 3.11.
- **Storage:**
  - Home has a 5 GB quota: keep only code, the venv and logs there.
  - Large outputs go to `/n/fs/scratch/rb5709`: shared 10 TB, purge policy unknown, assume not backed up. Point `results/` there with a symlink.
  - Ask the user whether project space has been granted. Keep final results there, or copy them back to the laptop.

## Steps

**Order of work:**
1. S0, then M1.
2. S1, S2 and S3.
3. S5, then S6.

B1–B3 (real bet sizes) can be built on the laptop while M1 runs on Ionic.

### S0. Ionic checks and scaling test (Ionic)

- **Checks (started 2026-10-27):**
  - the full test suite, with `--runslow`, on a compute node;
  - R4 (`configs/holdem/flop_turn.yaml`), which must reproduce exploitability 5.17e-4 and value −0.062097 at 200 iterations.
- **Scaling test:**
  - Run the R4 game for about 25 iterations at 8, 16, 32 and 64 CPUs (and a whole node if possible).
  - Record seconds per iteration and peak memory.
  - Use the result to choose the CPUs for every later job, and report it.

### M1. First real-deck three-street solve, limit (before S1)

The goal is the first exploitability number for actual double-board Hold'em. M1 also measures the speed and the iterations to converge, which every later budget depends on. It uses only verified code plus new plumbing.

1. **Plumbing (new code).** Test it on the small deck first.
   - Save and load the solver's full state (regrets, averages, iteration count, RNG state) and its average strategy.
   - A run script: fit buckets, solve without logging, and checkpoint every N iterations.
   - An evaluation script: load a saved strategy and run `exact_exploitability`.

   **Acceptance:**
   - A saved and reloaded strategy grades exactly as the in-memory one.
   - Resuming from a checkpoint continues bit-identically to a run that was never interrupted.
2. **Fit river buckets** for all 2,070 turn deals with `fit_equity_buckets`, using k = 200 (provisional) and 50 rivers each. Measured on the laptop: 1.7 s per turn deal, about 1 h in all.
3. **Solve** with `kmeans_2d`, and with `avg_1d` too if the user agrees to the compute.
   - Measured on the laptop: 7.5 s per iteration at batch 64 × 4 rivers (2.9 GB).
   - The number of iterations needed is unknown.
4. **Run the exact check on several checkpoints,** each in its own job, running in parallel. At k = 200 each takes about 21 h at laptop speed. The checkpoints show convergence, and the last one is the result.

**Measured cost:** at k = 200, the bucket lookup takes about 2 s per 256 river deals (0.18 s at k = 16). That's about a quarter of an iteration and 40% of an exact check. A compiled lookup is optional performance work: ask first.

**Stop and report after M1.**


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

This is the real-time river solve. It re-solves one river for one betting history, which is a small game:
- one pair of boards;
- about 861 hands per player;
- one river betting tree.

It should take seconds, whatever the bet sizes. It is not the exact check, which has to grade every river under every history.

### Real bet sizes: B1–B5

The game is the pot-limit one under Fixed decisions.

Size estimates come from a rough count, not from the repo's engine, so confirm them in B3:
- **Flop and turn:** about 10× the limit game, about 46 GB.
- **River:** about 7×. An exact check takes about 6 days at laptop speed, about a day on one node if the code scales.

- **B1. Betting engine in `core/`,** game-agnostic.
  - It supports bet sizes as fractions of the pot, stacks, all-in and a raise cap, with pot-limit or no-limit as a setting. `LimitPoker` must keep working.
  - **Acceptance:**
    - No-limit with {pot} matches OpenSpiel's `universal_poker` (no-limit, betting abstraction `fcpa`) on one board: the same legal actions, histories and payoffs.
    - Pot-limit, on small decks: DCFR converges to the LP value.
    - All existing tests pass.
- **B2. The range solver and public tree with these rules,** including all-in lines, which have no betting on later streets.
  - **Acceptance:**
    - Lockstep agreement with the tree solver on small decks.
    - The LP value matches.
- **B3. Count the real sizes** with the engine (extend `scripts/count_sizes.py`). Confirm the menu and raise cap with the user.
- **B4. Real-deck flop and turn with real sizes,** solved exactly and full-width on one node.
- **B5. All three streets with real sizes,** using M1's pipeline. This needs S3's split exact check.

## Open decisions (ask the user)

1. ~~Ionic's specs~~ Known (see Ionic): nodes have up to 503 GB, so S5 can use the 7 × 4 deck.
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
