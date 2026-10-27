# M1 status

Current step: 2, The M1 run script, rehearsed on the small deck
Last passed gate: 1, 03:55, commit `2cf050f`
Running jobs: none
Next action: test `m1.py` locally, commit it, then run the rehearsal through the Slurm scripts
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
