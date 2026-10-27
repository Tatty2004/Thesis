# M1 status

Current step: 1, Save and resume the sampled solver
Last passed gate: 0, 03:31, commit `ffe170f` (gate recorded in the Step 0 commit on `m1`)
Running jobs: none
Next action: write save/load, bucket round trip, the evaluation progress callback, and their tests
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
