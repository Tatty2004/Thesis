# M1 overnight: instructions for Claude on Ionic

Written 2026-10-27 for an unattended overnight session on Ionic. The goal is to build, verify and launch **M1**: the first full solve of real double-board Hold'em on the 52-card deck (flop, turn and river, both boards) for one pair of flops, with its exploitability measured exactly.

First read the root `CLAUDE.md`, `bombpot/holdem/CLAUDE.md`, and the "Ionic" and "M1" sections of `bombpot/holdem/PLAN_full_solver.md`. This file adds the step-by-step procedure and the checks each step must pass before the next one starts.

## What a successful night looks like

A solve runs for days and an exact check takes many hours, so M1 can't *finish* overnight. By morning:
1. Save/resume and the M1 run scripts are written, tested, committed and pushed (branch `m1`).
2. The whole pipeline has run end to end on the small deck and on a short real-deck pilot, with every check passing.
3. River buckets are fitted for all 2,070 turn deals.
4. The long solves are running on Ionic, writing healthy checkpoints.
5. The exact check of an early checkpoint has been submitted.
6. `logs/m1/STATUS.md` says what passed, the key numbers, the job ids, and what's still running.

## Fixed choices (don't change these)

- **Game:** `Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1))`. That's the real 52-card deck with limit betting (ante 1, bets 2/4/4, a bet and one raise per street), which is the `Holdem` default.
- **Solver:** `SampledRangeCFR` (`core/solvers/sampled_cfr.py`), with seed 0.
- **River buckets:** `fit_equity_buckets` (`core/abstraction/river.py`) with k = 200, 50 rivers per turn deal, seed 0, restarts 3. Run two solves, independent and side by side:
  - `kmeans_2d`: the thesis method;
  - `avg_1d`: the 1D baseline.

  (The user can drop `avg_1d`. If this line has been deleted, run only `kmeans_2d`.)
- **Sampling:** batch 64 turn deals × 4 rivers each, unless the pilot (Step 4) shows that batch 256 × 4 handles more river deals per second. Use whichever is faster, and record the choice.
- **Code changes:** no performance work, including the bucket-lookup speedup mentioned in the plan, and no changes to existing algorithms.

## Ground rules for working unattended

1. **Steps in order.** Each step ends in a **gate**: a list of checks. Go on to the next step only if every check passes. Record each gate in `logs/m1/STATUS.md`: time, PASS or FAIL, the numbers, and job ids.
2. **When a check fails:**
   - Diagnose it. You may fix code *you wrote tonight*, then rerun the whole gate.
   - Never change an existing test, never loosen a check, and never change the behavior of existing verified code. Additive changes, such as a new optional argument whose default keeps today's behavior, are fine if the full test suite still passes.
   - If a gate still fails after two honest attempts, or the fix needs a decision from the user: **stop.** Write what you found in `STATUS.md`, leave anything already running alone, and end the session.
3. **Git:**
   - Work on a branch `m1` made from up-to-date `main`. Commit at every passing gate, and push the branch.
   - Never push to `main` and never force-push.
   - Launch real-deck jobs only from a clean, committed state, and record the git hash in every run folder.
4. **Storage:**
   - Home has a 5 GB quota. Everything large goes under `results/`, which links to `/n/fs/scratch/rb5709`: buckets, checkpoints, evaluation outputs.
   - Keep M1's total under 300 GB, and check with `du` at each gate.
5. **Cluster etiquette:**
   - Use partition `all`, not `lowprio`, so long jobs aren't preempted.
   - Request CPUs per Step 0's rule, not whole nodes.
   - Have at most 2 solve jobs and 3 evaluation jobs running at once. Never cancel jobs you didn't submit.
6. **Waiting on jobs:**
   - Check `squeue` and the job logs every 10–30 minutes, depending on the job's length.
   - Don't busy-loop.
   - A job that dies is a failed check: read its log, then fix it and resubmit, or stop.
7. **Stop instead of launching** multi-day jobs if the pilot is far from expectations:
   - more than 60 s per iteration;
   - more than 150 GB of memory;
   - a projected exact check longer than 6 days.

## Staying on track: context, restarts and recovery

The session may run out of context and be summarized, hit a usage limit, or be restarted by the user. The real state of the run is on disk and in Slurm, so:

1. **`logs/m1/STATUS.md` is the source of truth.** Keep this block at its top, and update it at every gate and every job submission:
   ```
   Current step: <n and name>
   Last passed gate: <n>, <time>, commit <hash>
   Running jobs: <job id, what, submitted at, expected end>
   Next action: <one line>
   ```
   Add the detailed, timestamped entries below it.
2. **After a summary or a restart, re-read this file and `STATUS.md` before doing anything.** Then confirm the state from the system itself: `git log --oneline -5`, `git status`, `squeue -u $USER`, and the run folders under `results/m1/`.
   - Never resubmit a job that's already running or finished.
   - Never redo a passed gate's side effects.
   - If `STATUS.md` and the system disagree, trust the system, correct `STATUS.md`, and say so in it.
3. **Keep the context small:**
   - Read logs with `tail` and `grep`, never whole.
   - Don't print large arrays or full test output: use `pytest -q` and look at failures only.
   - Read large files in parts.
4. **Stay inside the task.** Do only what the steps below say. Don't refactor, tidy or improve code outside M1, start other plan steps, or explore side questions. If something outside M1 looks wrong, note it in `STATUS.md` for the user and carry on.
5. **When unsure, stop.** If you're unsure whether an action is allowed, or the situation doesn't match these instructions, write what you see in `STATUS.md` and stop. A stopped run costs a few hours; a wrong multi-day run costs days.

## Step 0: Starting point

- `git checkout main && git pull`, then `git checkout -b m1`.
- **The speed fix's scaling test** has run (2026-10-27, node202): 32 CPUs 1.16 s, 64 CPUs 1.01 s, a whole node with 52 threads 0.81 s; hyperthreads slightly hurt.
- **CPUs for M1's jobs:** 64 CPUs with 32 threads, one per physical core (`THREADS=32`). That's about half a node per job.
  - It hasn't been timed yet, so time it in Step 4's pilot next to 64 CPUs with 64 threads.
  - Use the faster of the two for the solves and evaluations.
- **The full test suite with `--runslow`** must pass on this commit (`scripts/ionic/tests.sbatch --runslow`). If that run hasn't happened, run it.

**Gate 0:**
- The scaling table is recorded, and the CPU choice is written in `STATUS.md`.
- Every test passes.
- `results/` resolves to scratch.

## Step 1: Save and resume the sampled solver (code)

In `core/solvers/sampled_cfr.py`, which must stay game-agnostic, add two methods:
- `SampledRangeCFR.save(path)`: writes one `.npz` with:
  - every `regret`, `cum` and `sigma` array;
  - `t`, `solve_seconds` and `value`;
  - the random generator's full state (`self.rng.bit_generator.state`, as JSON);
  - `batch`, `samples` and the seed.

  Save `sigma` itself, don't recompute it, so that resuming is bit-identical. Write to a temporary file and rename it, so a crash never leaves a half-written checkpoint.
- `SampledRangeCFR.load(path)`: restores all of that into a solver built with the same public tree, river and abstraction. It must refuse, with a clear error, if any array's shape doesn't match.

Also make the bucket objects round-trip:
- `EquityBuckets` saves and loads its centers, method and k (plus the fit's `samples` and seed, as metadata).
- `exact_exploitability` gets an optional `progress` callback, called every N chunks with (chunks done, total), so long evaluations can log an ETA. The default is `None`, which keeps today's behavior.

New tests go in `tests/holdem/test_holdem_checkpoint.py`. Use the small three-street game from `tests/holdem/test_holdem_river.py` (5 ranks × 3 suits, fixed flops) with both `Lossless` and fitted `kmeans_2d` buckets:
- **Round trip:** save, then load into a fresh solver. Every array is bit-identical, and `t`, `solve_seconds`, `value` and the generator state are equal.
- **Resume:** run 10 iterations uninterrupted. Separately, run 5 iterations, save, load into a fresh solver, and run 5 more. Regrets, averages and sigma must be bit-identical, and the next sampled deals must be identical.
- **Evaluation from a checkpoint:** `exact_exploitability` on the loaded solver's average strategy equals the in-memory solver's result exactly.
- **Buckets:** saved and reloaded buckets give identical ids for every hand at a sample of deals.

**Gate 1:**
- The new tests pass.
- The full test suite passes, run on a compute node. `--runslow` isn't needed again unless you changed `exact_exploitability`, in which case run `tests/holdem/test_holdem_river.py --runslow`.
- Commit and push the `m1` branch.

## Step 2: The M1 run script, rehearsed on the small deck

Write `bombpot/experiments/m1.py` with three commands:
- `fit --config C`:
  - builds the game, `holdem_public_tree(game, upto=2)` and `HoldemRiver`;
  - fits the buckets;
  - writes `buckets.npz` plus `meta.json` (config, git hash, dirty flag, Python and library versions, CPU model, timing).
- `solve --config C [--resume]`:
  - loads the buckets and builds the solver;
  - runs until a time budget given in the config, which must stop at least 30 minutes before the job's walltime;
  - saves a checkpoint every `checkpoint_hours`, and at the end;
  - logs one line every few minutes: iteration, seconds per iteration, elapsed time, peak memory;
  - with `--resume`, continues from the newest checkpoint;
  - keeps the newest 2 checkpoints plus the first checkpoint after 2, 4, 8, 16, 32 and 64 hours of solving, and deletes the rest.
- `evaluate --checkpoint P`:
  - loads the checkpoint and computes its average strategy;
  - runs `exact_exploitability` with progress logging (chunks done and ETA every few percent);
  - writes `eval.json` next to the checkpoint: exploitability, best-response values, game value, seconds, git hash.
- **Health check** (inside `solve`, at every checkpoint):
  - every array is finite;
  - every average-strategy row sums to 1 within 1e-9;
  - `t` increased.

  Write the result to the log. A failure stops the solve.

Configs go in `bombpot/experiments/configs/holdem/`:
- `m1_smoke.yaml`: the small deck from the river tests. Include a `Lossless` option for it.
- `m1_kmeans2d.yaml` and `m1_avg1d.yaml`: the real deck, with the fixed choices above.

Run folders are `results/m1/<config name>/`. Slurm scripts go in `scripts/ionic/`: `m1_fit.sbatch`, `m1_solve.sbatch` and `m1_eval.sbatch`, sourcing `env.sh`.

**Rehearsal**, through those same Slurm scripts with `m1_smoke.yaml` on a compute node:
1. Fit, then solve with checkpoints a few minutes apart.
2. Stop the solve partway: let a short time budget end it, then rerun with `--resume`.
3. Evaluate two checkpoints.

**Gate 2:**
- **Resume is invisible:** a solve that was stopped and resumed has bit-identical final regrets and averages to an uninterrupted solve with the same total iterations. Run both to compare.
- **Evaluation agrees:** with `Lossless` on the small deck, the exploitability from `evaluate` equals the in-process `exact_exploitability` of the same solver state exactly, and it is lower at the later checkpoint.
- **Values make sense:** every evaluation has exploitability ≥ 0, and the game value lies between −BR1 and BR0 (allowing 1e-9).
- **Health checks pass, and retention works:** the expected checkpoints are kept and the rest deleted.
- Commit and push.

## Step 3: Fit the real-deck buckets

Run `m1_fit.sbatch` for each method. The laptop took about 1.7 s per turn deal, about an hour in all.

**Gate 3, for each method:**
- **Shape and range:** the centers are shaped (2070, 200, dim), finite, and inside [0, 1]. Each turn deal's centers are ordered weakest first.
- **Deterministic:** refitting 3 turn deals (the first, a middle one, the last) with the same seed gives identical centers.
- **Ids are well formed:** at 5 random turn deals × 5 random rivers, every hand that can be held gets an id in 0..199, and every hand that can't gets 200.
- **Occupancy is reported, not judged:** report the minimum, median and maximum number of hands per bucket.
- `meta.json` records a clean git hash.

## Step 4: Real-deck pilot

**First, pick up any newer `main`.** A speed fix for the river code (bucket lookup, scatter and gather) may land on `main` tonight.
- Run `git fetch && git merge origin/main` into `m1`.
- If the merge brought changes, rerun the full test suite and your Step 1–2 tests before going on (a gate).
- If the merge conflicts, stop and report.
- The fitted buckets from Step 3 stay valid: the fix doesn't change results.

Submit `m1_solve.sbatch` with `m1_kmeans2d.yaml` and a 45-minute time budget, checkpointing every 15 minutes, with the chosen CPUs.

**Measure:**
- seconds per iteration at batch 64 × 4 and at 256 × 4, and the river deals per second of each;
- peak memory;
- checkpoint size and save time.

**Check resume on the real deck:** from a checkpoint, run 2 more iterations resumed and 2 more uninterrupted, from the same state. Regrets and averages must be bit-identical.

**Time the exact check:** run `evaluate` on a pilot checkpoint for about 20 minutes and read the ETA from its progress lines. Stop it after that; its only purpose is timing.

**Gate 4:**
- Resume is bit-identical on the real deck.
- Health checks pass.
- The numbers are inside rule 7's limits.
- Write in `STATUS.md`:
  - the chosen batch;
  - seconds per iteration;
  - the projected hours per pass over all 2,070 turn deals;
  - the projected exact-check time.

## Step 5: Launch the long solves

Submit one `m1_solve.sbatch` per method:
- the chosen CPUs and `--mem` at twice the pilot's peak;
- `--time=3-00:00:00`, with a time budget that stops 30 minutes before;
- `checkpoint_hours: 2`.

If a solve reaches its walltime, resubmit it with `--resume`. Resubmit at most once overnight, then leave it for the user.

**Gate 5 (after the first checkpoint of each solve):**
- The checkpoint loads.
- The health check passes.
- Seconds per iteration is within 25% of the pilot's.
- Memory is under its request.

**Sanity printout (not a gate):** at each checkpoint, log player 0's flop strategy (bet or check frequencies) for a few hands. Use any hands that are possible on these flops, for example:
- K♣K♥: quads on board B;
- A♣A♦: three aces on board A;
- 7♦7♥: a set on board A;
- 9♣9♦: a full house on board B;
- 5♠6♠: a straight draw on board A;
- 3♣4♦: little on either board.

Also log player 0's overall flop bet frequency. The user will judge whether these look sensible.

## Step 6: Exact checks

Once each solve has a checkpoint from about 2 hours of solving, submit `m1_eval.sbatch` on it:
- walltime 1.5× the Step 4 ETA, at most 7 days;
- `--mem` at twice the pilot evaluation's peak.

Overnight, submit at most one evaluation per method. Later checkpoints (8, 16, 32 hours) are for the user to request.

**Gate 6 (when an evaluation finishes, or in the morning, its progress so far):**
- Exploitability is ≥ 0.
- The game value lies between −BR1 and BR0 (allowing 1e-9).
- `eval.json` records a clean git hash and the checkpoint's iteration.

## Step 7: Morning report

Finish `logs/m1/STATUS.md` with a short summary, in plain language:
- which gates passed;
- the commits;
- for each solve: iterations so far, seconds per iteration, checkpoints, job id, time left;
- the exact-check results so far, or their ETAs;
- disk usage;
- anything surprising.

Push the `m1` branch, which has code only. Then stop. Don't start S1 or any other plan step.

## Reference numbers (laptop, M1 Pro, 8 cores, before the speed fix)

| What | Laptop |
|---|---|
| R4 iteration (flop and turn, full width) | 2.7 s, about 4.5 GB |
| Bucket fit, k = 200, 50 rivers | 1.7 s per turn deal (about 1 h for all) |
| Sampled iteration, batch 64 × 4, k = 200 | 7.5 s, 2.9 GB |
| Sampled iteration, batch 256 × 4, k = 200 | 41.7 s, 6.2 GB |
| Exact check, k = 200 | about 21 h (3.9M river deals; the bucket lookup is about 40%) |
| Iteration 25 of R4 (correctness reference) | exploitability 0.0340556029671678 |
