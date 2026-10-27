# M1 status

Current step: 5, Launch the long solves
Last passed gate: 4, 07:32, commit `af52b9b` (code)
Running jobs (submitted 07:30, 3-day walltime, budget 71.5 h, so expected end about 2026-10-30 07:00): long solve 31994837 (m1_kmeans2d, node202), long solve 31994838 (m1_avg1d, node204)
Next action: Gate 5 at each solve's first checkpoint (about 09:30), then the Step 7 report, then stop
Push: **blocked**, so commits on `m1` are local only for now (see 03:20 below)

Times are EDT, 2026-10-27. Newest entries at the bottom.

**User instruction, 03:28:** "if you do pass anything then feel free to push to step 5 (the long solve)". My reading: if every gate through Gate 4 passes and the pilot is inside rule 7's limits, launch Step 5 (one long solve per method) tonight, and run Gate 5 on their first checkpoints. Step 6 (exact checks) still waits for the user's review. Any failed gate still stops the run.

## Log

### 03:20, Step 0 started
- **Pull.** `git pull` over SSH failed: the deploy key `~/.ssh/github_thesis` has a passphrase, and this session has no terminal and no ssh-agent to supply it. The repo is public, so I pulled `main` over HTTPS instead (`git fetch https://github.com/Tatty2004/Thesis.git main:refs/remotes/origin/main`, then a fast-forward): `ec589c6` to `ffe170f`. Branch `m1` was made from `ffe170f`.
- **Pushing needs the key.** There's no credential helper and no `gh` login, and I won't use any other credentials. Until the key is unlocked, every step is committed on `m1` locally and listed here as unpushed. I try the push at every push point. To unlock it, run this on spin (key cached 16 h):
  `ssh-agent -a /tmp/rb5709-m1-agent.sock -t 16h >/dev/null && SSH_AUTH_SOCK=/tmp/rb5709-m1-agent.sock ssh-add ~/.ssh/github_thesis`
  Otherwise, in the morning `git push -u origin m1` pushes everything.
- **Tests.** Job 31994379 (`sbatch -x node202 scripts/ionic/tests.sbatch --runslow`, the user's, node010) is the full suite with `--runslow` on `ec589c6`. `ffe170f` differs from `ec589c6` only in two docs (`M1_CLAUDE.md`, `PLAN_full_solver.md`), so it tests this commit's code. I'm waiting for it rather than submitting a duplicate. It reads the live working tree, so I'm not touching `bombpot/` or `tests/` until it ends.
- **`results/`** resolves to `/n/fs/scratch/rb5709` (scratch: 10 TB, 2.3 TB free; home quota 5 GB, 2.0 GB used).

### Scaling test (S0), recorded for Gate 0
Job 31994378 on node202 (Xeon Gold 5320, 104 threads), commit `ec589c6` (after the speed fix): 25 R4 iterations per setting.

| CPUs | Threads | s/iteration (mean after the first) | Peak memory |
|---:|---:|---:|---:|
| 8 | 8 | 2.31 | 7.5 GB |
| 16 | 16 | 1.52 | 7.4 GB |
| 32 | 32 | 1.16 | 7.4 GB |
| 64 | 64 | 1.01 | 7.4 GB |
| 104 | 104 | 0.94 | 7.5 GB |
| 104 | 52 | 0.81 | 7.4 GB |

- Every setting gives exploitability 0.03405560296716777 at iteration 25, the R4 correctness reference (0.0340556029671678).
- Before the speed fix (job 31994142, same node), every setting took 3.7–4.8 s per iteration.
- Hyperthreads slightly hurt: 104 CPUs with 52 threads beats 104 with 104.

**CPU choice for M1's jobs:** 64 CPUs with 32 threads (`THREADS=32`, one per physical core), about half a node. Step 4's pilot times it against 64 CPUs with 64 threads, and the faster one is used for the solves and evaluations.

### 03:31, Gate 0: PASS
- Scaling table and CPU choice: recorded above.
- Every test passes: job 31994379 (node010, Xeon Gold 5218, 32 CPUs), `tests.sbatch --runslow`: **408 passed, none skipped**, 31 min 18 s, peak 11.1 GB. The code is `ec589c6`'s, identical to `ffe170f`. The earlier run's 4 failures (job 31993952, `af3caa7`) are gone.
- `results/` resolves to `/n/fs/scratch/rb5709`. `results/m1/` created, 2.5 KB.

### 03:45, Step 1 code committed (`2cf050f`), Gate 1 tests submitted
- `SampledRangeCFR.save(path)` / `.load(path)` in `core/solvers/sampled_cfr.py`:
  - One uncompressed `.npz`: every `regret`, `cum` and `sigma` array (`sigma` saved, not recomputed), `t`, `solve_seconds`, `value`, the generator's state as JSON, `batch`, `samples` and `seed`. The solver now keeps `self.seed`.
  - Written to `<name>.tmp`, fsynced, then renamed.
  - `load` checks the array names, shapes and dtypes before changing anything, and raises a `ValueError` naming the mismatched array.
- `EquityBuckets.save(path)` / `EquityBuckets.load(path)` in `core/abstraction/river.py`: centers, method and k, plus `fit` metadata (`samples`, `seed`, `restarts`), which `fit_equity_buckets` now fills in. Centers and ids are unchanged.
- `exact_exploitability(..., progress=None, progress_every=1)`: calls `progress(chunks done, total)` every N chunks and after the last. With the default `None`, behavior is unchanged.
- `tests/holdem/test_holdem_checkpoint.py`, 10 tests, all passing locally on spin (4 min 45 s):
  - round trip and resume, each with `Lossless` and `kmeans_2d`;
  - evaluation from a checkpoint, the bucket side going through save and load too;
  - bucket round trip, for both methods;
  - a mismatched load is refused and changes nothing;
  - progress callback.
- Jobs: 31994580 runs the full suite. 31994581 runs `test_holdem_river.py --runslow`, needed because `exact_exploitability` changed.

### 03:55, Gate 1: PASS
- New tests pass: all 10 in `test_holdem_checkpoint.py`, on node009 (in the full run below) and on spin.
- Full suite on a compute node: job 31994580 (node009, clean `2cf050f`): **397 passed, 21 skipped** (the slow tests, as intended without `--runslow`), 10 min 54 s. No failures or errors.
- `exact_exploitability` changed, so `tests/holdem/test_holdem_river.py --runslow` ran: job 31994581 (node010, `2cf050f`), **12 passed**, including the slow convergence test.
- Commit: `2cf050f` (code), plus status commits. **Not pushed:** the SSH key is still locked (no agent at `/tmp/rb5709-m1-agent.sock`).

### Measured on spin (Xeon Silver 4316) for sizing
- Real-deck bucket fit, k = 200, 50 rivers: **kmeans_2d 2.27 s per turn deal** (about 1.3 h for 2,070), **avg_1d 0.25 s** (about 9 min). The fit is sequential (one generator through the turn deals), so `m1_fit.sbatch` asks for 4 CPUs, not 64.
- Real-deck layout: flop, turn and river templates have 1, 5 and 25 entering lines and 14 actions over 6 nodes. Regret, average and strategy are about 2.4 GB each once every turn deal has been sampled, so **a checkpoint is about 7.1 GB**, uncompressed (the pilot will measure it).

### 04:36, Step 2 code committed (`f8c5791`), rehearsal phase A submitted
- `bombpot/experiments/m1.py`:
  - **`fit`:** builds the game, `holdem_public_tree(game, upto=2)` and `HoldemRiver`, then calls `fit_equity_buckets` **one turn deal at a time on one shared generator**. `default_rng` passes a Generator through unaltered, so the centers are bit-identical to one call; `test_m1.py` checks this, and `check-fit --full` checks it on the smoke deck. This lets it log progress and record the generator's state before each turn deal (`fit_states.json`). Gate 3's "refit 3 turn deals with the same seed" needs those states, because the generator runs through all turn deals in order. Writes `buckets.npz`, `meta.json` (config, git hash and dirty flag, versions, CPU model, timing), then runs **`check-fit`** (Gate 3's checks, written to `fit_checks.json`).
  - **`solve`:**
    - Time budget counted from process start; refuses to start unless it ends at least 30 min before the job's walltime (read from `squeue`).
    - Checkpoints every `checkpoint_hours` and at the end, after the health check. The check covers finite arrays, average-strategy rows summing to 1 within 1e-9, and `t` increasing; a failure stops the solve without saving.
    - Logs t, s/it, elapsed, solve hours and peak memory every `log_minutes`.
    - Retention: newest 2 plus the first at or after each `milestone_hours`; `.eval.json` files are never deleted.
    - Options: `--resume`, `--tag` (a separate `solve-<tag>/`), `--iterations`, overrides of budget, checkpoint interval and batch, and `--resume-check N` (at the end, N iterations uninterrupted vs resumed from the last checkpoint, compared bit for bit).
    - At each checkpoint it logs player 0's flop bet frequency (overall and for `flop_hands`). With `eval_in_process` (small deck only) it also runs the exact check in-process.
  - **`evaluate`:** loads a checkpoint, runs `exact_exploitability` with progress lines (chunks, s/chunk, projected total, ETA) every 2% or 2 minutes, and writes `ckpt_t<t>.eval.json` next to the checkpoint. With `--max-minutes` it stops early and writes `.timing.json`.
  - **`compare`:** checks whether two checkpoints are bit-identical.
- **Configs:**
  - `m1_smoke.yaml`: small deck, kmeans_2d with k = 3.
  - `m1_smoke_lossless.yaml`: small deck, lossless river. A separate file is how I read "include a Lossless option".
  - `m1_kmeans2d.yaml` and `m1_avg1d.yaml`: the real deck, with the fixed choices written out. `test_m1.py` checks they equal `Holdem(flops=("Ah7c2d","KsKd9h"), board_cards=(3,1,1))`, k 200, 50 rivers, seed 0, restarts 3, batch 64 × 4.
- **Slurm scripts:** `scripts/ionic/m1_fit.sbatch`, `m1_solve.sbatch` and `m1_eval.sbatch`, logging to `logs/m1/`. Solve and evaluate default to 64 CPUs with `THREADS` = half of them; fit asks for 4 CPUs.
- **Local dry run on spin (throwaway config, deleted afterwards):** fit with checks; solve stopped by budget; `--resume`; an uninterrupted twin, **bit-identical** to the resumed run (t 1108); `evaluate` equal **exactly** to the in-process check at two checkpoints; the timing stop; `--resume-check 2` PASS.
- **Rehearsal:** all pinned to node010 (Gold 5218), because bit-identical comparisons across jobs need the same CPU type. Node types in `all`:
  - node2xx and node302: Gold 5320, the scaling test's type;
  - node0xx below 020: Gold 5218;
  - node020–026 and node10x: Gold 5220R;
  - node403: Gold 6338;
  - node9xx: Silver 4214.

### 04:45, rehearsal phase A done, phase B submitted
- Fits (jobs 31994672, 31994675): `m1_smoke` checks **all PASS**, including `one_call_identical` (the per-turn-deal fit equals one `fit_equity_buckets` call) and a clean git hash. Lossless: nothing to fit.
- Walltime guard works in Slurm: "budget ends 04:40:43, the job's walltime 05:17:05: 36 min apart".
- Solves stopped by their 3.6-minute budget, then `--resume`d:
  - `m1_smoke`: t 2733, then t 5453 (0.07 s/it).
  - `m1_smoke_lossless`: t 2292, then t 4683 (0.08 s/it).
  - All health checks PASS.
  - Lossless in-process exact exploitability falls: 0.199 → 0.074 → 0.056 → 0.041 → 0.032.
- Tests on a compute node (job 31994678): `test_m1.py`, the checkpoint tests and the rules, **19 passed**.
- Added `m1.py check --checkpoint P` (loads into a fresh solver, health check, flop strategy), for Gate 5's "the checkpoint loads". Commit `fdec2b7`.

### 04:55, takeover by a new Claude instance
- Read the host rules, both CLAUDE.md files, `M1_CLAUDE.md`, this file and `M1_HANDOFF.md`, then checked the system: branch `m1` at `f27de52`, clean tree; in `squeue` only 31994703 (running) and 31994704 (pending, dependency). The header's job list was stale (the twins and all four evaluations had already ended, all COMPLETED, exit 0), so I corrected it.
- **Twin comparisons** (`m1.py compare`, uninterrupted vs stopped-and-resumed):
  - `m1_smoke`, t 5453 (job 31994695 vs `solve/`): **bit-identical**.
  - `m1_smoke_lossless`, t 4683 (job 31994696 vs `solve/`): **bit-identical**.
  - Both twins' health checks PASS (max row-sum error 3.3e-16). The lossless twin's in-process exploitability at t 4683 is 0.0286184, equal to the resumed run's evaluation.

### 05:02, Gate 2: PASS
Code at `f27de52` (no code changes since the rehearsal), all jobs on node010 (Gold 5218).
- **Resume is invisible:** uninterrupted twins equal the stopped-and-resumed solves bit for bit: `m1_smoke` at t 5453 (31994695), `m1_smoke_lossless` at t 4683 (31994696). See 04:55.
- **Evaluation agrees (lossless):** `evaluate` equals the in-process `exact_exploitability` exactly at t 1807 (31994697) and t 4683 (31994698), and falls: 0.0740 → 0.0286.
- **Values make sense:** 4 evaluations (31994697–31994700) and every in-process check: exploitability ≥ 0 and −BR1 ≤ value ≤ BR0 within 1e-9.
- **Health checks:** all PASS: 16 in the rehearsal, 2 in the twins, 10 in the retention recheck. No FAIL anywhere.
- **Retention:** recheck 31994703 (budget stop at t 3001) then 31994704 (`--resume`, to t 5961), both COMPLETED. `m1_check_retention.py results/m1/m1_smoke/solve-retention`: rule keeps [1335, 2708, 4993, 5673, 5961], on disk the same, every one of the 5 deletions as the rule says, **match: True**. Milestones 0.02/0.04/0.08 h kept t 1335/2708/4993, and the rule held across the resume. This supersedes the unverifiable t 882 deletion in `m1_smoke_lossless/solve` (old logs lacked hours).
- Disk: `results/m1` 1.6 GB.

### 05:01, Step 3: fits submitted
- Push: still blocked (no agent at `/tmp/rb5709-m1-agent.sock`), so `4b4023a` and earlier are local only.
- `sbatch -w node208` was refused ("Requested node configuration is not available"). `--test-only` shows the scheduler places `all` jobs on node202–204 through the `cs` node set, which doesn't include node205–208, so node208 can't be targeted. node204 (Gold 5320, same type as the pilot nodes) was accepted. node202 and node302 stay free for the pilot.
- 31994725: `m1_fit.sbatch --config m1_kmeans2d.yaml` (4 CPUs, 8G, 4 h), node204.
- 31994726: `m1_fit.sbatch --config m1_avg1d.yaml` (4 CPUs, 8G, 1 h), node204.
- Commit `4b4023a` (clean `bombpot/`).

### 05:17, Step 3: avg_1d fit done, all checks PASS (Gate 3 half)
Job 31994726, node204 (Gold 5320), COMPLETED in 14 min: 36 s to build, 519 s to fit (0.25 s per turn deal), peak 0.34 GB (sacct MaxRSS 0.38 GB).
- Shape (2070, 200, 1), finite, in [0, 1], weakest first: PASS.
- Deterministic: refits of turn deals 0, 1035, 2069, and 0 from the seed: all identical.
- Ids well formed at 5 turn deals × 5 rivers: PASS.
- `meta.json`: git `af52b9b`, dirty false.
- **Occupancy (reported, not judged):** over 25 river deals with 861 possible hands each: min 0, median 0, max 102 hands per bucket; **64% of the 200 buckets are empty at a given river deal.** The centers are fitted per turn deal on 50 sampled rivers, so at any one river most 1D centers get no hands. Odd enough to flag, but the procedure says report only. kmeans_2d's numbers will show whether this is particular to 1D.

### 06:18, Gate 3: PASS (both methods)
kmeans_2d: job 31994725, node204, COMPLETED in 1 h 15 min: 4,148 s to fit (2.00 s per turn deal), peak 0.34 GB (MaxRSS 0.44 GB).

| Check | kmeans_2d | avg_1d |
|---|---|---|
| Shape | (2070, 200, 2) | (2070, 200, 1) |
| Finite, in [0, 1], weakest first | PASS | PASS |
| Refits of turn deals 0, 1035, 2069, and 0 from the seed | identical | identical |
| Ids at 5 turn deals × 5 rivers (0..199 for possible hands, 200 otherwise) | PASS | PASS |
| `meta.json` git | `af52b9b`, clean | `af52b9b`, clean |
| Occupancy (hands per bucket, 25 river deals, 861 hands each): min / median / max | 0 / 0 / 108 | 0 / 0 / 102 |
| Buckets empty at a given river deal | 73.5% | 64.1% |

- **Occupancy, reported not judged:** both methods leave most of their 200 buckets empty at any one river deal. On average 861 hands share about 53 (kmeans_2d) or 72 (avg_1d) occupied buckets. The centers are fitted per turn deal over 50 sampled rivers, and one river concentrates the hands' equities, so this may be expected. It does mean the effective number of river buckets per deal is far below 200. **Worth the user's look before reading much into k = 200.**
- Disk: `results/m1` 1.6 GB (the buckets are small).

### 06:17, Step 4: merge, pilot submitted
- **Merge:** fetched `main` over HTTPS: no commits beyond `m1`'s base, so the merge is a no-op. No changes, so no retest was needed (the gate's retest applies only "if the merge brought changes").
- Push: still blocked.
- **node302 can't be targeted** (same refusal as node208). The timing runs went to node204 (Gold 5320) instead. node204 has another user's job on 20 of its CPUs, so all four settings, b64-t32 included as their own baseline, run there one after another and are compared with each other. The main pilot runs on idle node202.
- Jobs, all with `m1_kmeans2d.yaml` from clean `c99a877` (code `af52b9b`):
  - 31994796, main pilot: `THREADS=32 sbatch -w node202 --time=01:45:00 --mem=48G m1_solve.sbatch --tag pilot --budget-hours 0.75 --checkpoint-hours 0.25 --resume-check 2`. The scheduler placed it in partition `cs` (not preemptible).
  - Timing chain, each `-w node204 --time=00:50:00 --mem=48G`, `--budget-hours 0.25 --checkpoint-hours 1`, each after the previous one (`afterany`):
    - 31994797 `THREADS=64 --tag pilot-b64-t64`
    - 31994798 `THREADS=32 --tag pilot-b64-t32`
    - 31994799 `THREADS=32 --batch 256 --tag pilot-b256-t32`
    - 31994800 `THREADS=64 --batch 256 --tag pilot-b256-t64`
- At most 2 solve jobs run at once.

### 07:05, Step 4: main pilot done, timing runs 1–3 done
- **Main pilot 31994796** (node202, b64, 32 threads), COMPLETED in 46 min:
  - 367 iterations at **7.151 s/it** after the first (14.1 s with compiling), 35.8 river deals/s.
  - Peak: 9.89 GB in the solve (log), 16.50 GB during the resume check, which holds two solvers. **sacct MaxRSS 24.2 GB**, higher than both. I size `--mem` from sacct's number.
  - Checkpoints: t 125, 251, 367, each **6.61 GiB** (7,094,668,434 bytes), saved in 17.9 / 12.3 / 12.2 s.
  - Health checks: 3 of 3 PASS (max row-sum error 4.4e-16).
  - Retention: kept t 251 and 367 (newest 2; no milestone yet), deleted t 125. That's the rule.
  - **Real-deck resume check (2 iterations resumed from t 367 vs 2 uninterrupted): PASS**, with regret, cum, sigma, t and generator all bit-identical.
  - Player 0's flop bet frequency (overall) by checkpoint: 0.72 → 0.46 → 0.31. Per hand at t 367: KcKh 0.17, AcAd 0.17, 7d7h 0.42, 9c9d 0.20, 5s6s 0.30, 3c4d 0.16. It's still moving a lot this early.
- **Timing runs** (node204, 15 min each):

| Job | Batch | Threads | s/it | River deals/s | Peak (log) | MaxRSS |
|---|---:|---:|---:|---:|---:|---:|
| 31994797 | 64 | 64 | 7.124 | 35.9 | 9.82 GB | 16.5 GB |
| 31994798 | 64 | 32 | 7.192 | 35.6 | 9.80 GB | 16.5 GB |
| 31994799 | 256 | 32 | 28.418 | 36.0 | 18.60 GB | 19.4 GB |
| 31994800 | 256 | 64 | 28.440 | 36.0 | 18.61 GB | 19.5 GB |
| 31994796 (node202) | 64 | 32 | 7.151 | 35.8 | 9.89 GB | 24.2 GB (incl. the two-solver resume check) |

- Exact-check timing 31994814 submitted 07:04: `THREADS=32 sbatch -w node202 --time=00:40:00 --mem=32G m1_eval.sbatch --checkpoint results/m1/m1_kmeans2d/solve-pilot/ckpt_t000000367.npz --max-minutes 20`.

### 07:32, Gate 4: PASS
- **Exact-check timing** 31994814 (node202, 32 threads, checkpoint t 367): stopped after 20 min as asked, at 237 of 15,299 chunks (256 river deals each, 3,916,440 in all), 5.05 s per chunk, steady. **Projected 21.5 h for a full exact check.** Peak 9.35 GB (sacct MaxRSS 9.7 GB). `.timing.json` records clean git.
- **Checks:**
  - Resume bit-identical on the real deck: **PASS** (31994796: regret, cum, sigma, t and generator).
  - Health checks: **PASS**, all 9 across the five pilot and timing solves.
  - Rule 7: 7.15 s/it (limit 60); peak 24.2 GB (limit 150); exact check 21.5 h (limit 6 days). **Inside all three.**
- **Batch: 64 × 4 (unchanged).** At 256 × 4 the river deals per second were 36.0 vs 35.6–35.9 at 64. That's 0.3–1.1% higher, inside the spread among the batch-64 runs themselves (0.8%), so the pilot doesn't show 256 to be faster. 256 also needs about twice the memory and gives a quarter of the iterations. The configs are unchanged.
- **Threads: 32 per 64 CPUs (unchanged from Step 0).** 64 threads was 0.9% faster at batch 64 and 0.1% slower at batch 256: no real difference. 32 matches Step 0's finding that hyperthreads don't help.
- **Projections** at 7.15 s/it, batch 64:
  - one pass over all 2,070 turn deals = 2070 / 64 × 7.15 s = **3.9 min** (0.064 h);
  - a 71.5 h solve ≈ 36,000 iterations ≈ 1,110 passes;
  - exact check ≈ 21.5 h per checkpoint at 32 threads on Gold 5320.
- **Memory request for Step 5:** twice the pilot's peak. The largest measured is MaxRSS 24.2 GB (it includes the resume check; a plain solve measured 16.5 GB), so **`--mem=50G`**.
- Disk: `results/m1` 34 GB (pilot and timing checkpoints, 6.6 GB each). Scratch has 2.3 TB free.
- No merge changes, so the Step 1–2 tests didn't need rerunning.

### 07:30, Step 5: long solves submitted
Gates 0–4 all passed and the pilot is inside rule 7, so per the user's instruction (03:28) both long solves are launched. Step 6 (exact checks) is **not** started.
- 31994837: `THREADS=32 sbatch -w node202 --time=3-00:00:00 --mem=50G scripts/ionic/m1_solve.sbatch --config bombpot/experiments/configs/holdem/m1_kmeans2d.yaml`
- 31994838: the same, `-w node204`, `--config .../m1_avg1d.yaml`
- No overrides: the configs give batch 64 × 4, seed 0, `budget_hours: 71.5`, `checkpoint_hours: 2`, milestones 2/4/8/16/32/64 h. Run folders `results/m1/m1_kmeans2d/solve/` and `results/m1/m1_avg1d/solve/`.
- Clean commit `a7d7121` (code `af52b9b`).
- **Expected disk:** each solve keeps at most 2 newest + 6 milestone checkpoints × 6.6 GiB ≈ 53 GiB, about 106 GiB for both, plus 34 GB already there. Under 300 GB.
- If one reaches its walltime: resubmit once with `--resume`. (Expected about 2026-10-30 07:00, after this session.)
- avg_1d hasn't been piloted itself. Its arrays are the same shape (k = 200), so it's judged against the pilot's 7.15 s/it.
