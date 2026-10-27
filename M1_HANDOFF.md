# M1 handoff: taking over the overnight M1 run

Written 2026-10-27 04:50 EDT by the Claude instance that ran Steps 0–2, for the instance taking over. The repo is `/n/fs/ugrad/ug27/rb5709/Thesis` on Ionic (host spin), branch **`m1`**.

## Read first, in this order
1. `/etc/claude-code/CLAUDE.md`: host rules. Never traverse `/n/fs`, scope all I/O, no sudo, wait at least 60 s between Slurm status checks.
2. Root `CLAUDE.md` and `bombpot/holdem/CLAUDE.md`.
3. `bombpot/holdem/M1_CLAUDE.md`: **the procedure, with a gate after every step. Follow it exactly.**
4. `M1_STATUS.md` (repo root): the source of truth. Its top block holds the current step, last passed gate, running jobs and next action; timestamped entries follow.
5. This file: what isn't obvious from the above.

Then confirm the state from the system before acting: `git log --oneline -5`, `git status`, `squeue -u $USER`, `ls results/m1/*/`. If `M1_STATUS.md` disagrees with the system, trust the system, fix the status file, and say so in it.

## What the user has said (binding)
- **Permission for Step 5:** "if you do pass anything then feel free to push to step 5 (the long solve)". If every gate through Gate 4 passes and the pilot is inside rule 7's limits, launch Step 5 (one long solve per method) and run Gate 5 on their first checkpoints. **Step 6 (exact checks) is not authorized.** Then write Step 7's report and stop. A failed gate still means stop.
- **Don't ask permission for routine things** (commits, submissions, fixing your own code, reruns): "if I come back and I see 'permission to do xyz' i'm gonna be mad unless it's very very important." Act, and log it in `M1_STATUS.md`. Stop only where `M1_CLAUDE.md` says to stop.

## Pushing is blocked (not a gate failure)
`~/.ssh/github_thesis` has a passphrase, and there is no TTY or agent. Pulls work over HTTPS because the repo is public:
`GIT_TERMINAL_PROMPT=0 git fetch https://github.com/Tatty2004/Thesis.git +refs/heads/main:refs/remotes/origin/main`.
At every push point, check `SSH_AUTH_SOCK=/tmp/rb5709-m1-agent.sock ssh-add -l`. If it lists the key, run `SSH_AUTH_SOCK=/tmp/rb5709-m1-agent.sock git push -u origin m1`. Otherwise commit locally and keep the "Push: blocked" line in the status file. Never look for or use other credentials. Never push to `main`, never force-push.

## Where things stand (04:50)
- **Gate 0: PASS** (03:31) and **Gate 1: PASS** (03:55). The details are in `M1_STATUS.md`.
- **Step 2:** all code is committed: `m1.py` at `55120df`, configs, Slurm scripts, `tests/holdem/test_m1.py`. The rehearsal on the small deck is nearly done. **Gate 2 is not yet judged.**

| Gate 2 check | State |
|---|---|
| Resume is invisible | Twins 31994695 (`m1_smoke`, `--tag straight --iterations 5453`) and 31994696 (`m1_smoke_lossless`, `--iterations 4683`) were running at 04:50. When they end, compare each with its stopped-and-resumed counterpart (commands below). Both must print `bit-identical`. |
| Evaluate equals in-process (lossless), and lower later | **Done, PASS.** Exact equality at t 1807 and t 4683; 0.0740 → 0.0286. |
| Values make sense | **Done, PASS.** 4 evaluations and 16 in-process checks: exploitability ≥ 0 and −BR1 ≤ value ≤ BR0. |
| Health checks | **Done, PASS.** 16 of 16. |
| Retention | `m1_smoke`: matches the rule. `m1_smoke_lossless`: one deletion (t 882) can't be verified exactly, because old logs lacked solve hours, so retention is rechecked: jobs 31994703, then 31994704 (`m1_smoke --tag retention`, then `--resume`), queued behind the twins. When they finish, run `scripts/ionic/m1_check_retention.py results/m1/m1_smoke/solve-retention`. It must print `match: True`, with deletions all as the rule says. |

Compare commands for the twins:
```
M=".venv/bin/python -m bombpot.experiments.m1"
$M compare results/m1/m1_smoke/solve/ckpt_t000005453.npz results/m1/m1_smoke/solve-straight/ckpt_t000005453.npz
$M compare results/m1/m1_smoke_lossless/solve/ckpt_t000004683.npz results/m1/m1_smoke_lossless/solve-straight/ckpt_t000004683.npz
```
If all of that passes, record Gate 2 (time, numbers, job ids, commit) in `M1_STATUS.md`, commit, try the push, and go to Step 3.

## How `m1.py` works (`python -m bombpot.experiments.m1 -h`)
- **Commands:** `fit` (then `check-fit` automatically), `check-fit [--full]`, `solve`, `evaluate`, `check`, `compare`.
- **Run folder** `results/m1/<config stem>/`: `config.yaml` (refuses a config with a different game or buckets), `buckets.npz`, `meta.json`, `fit_states.json`, `fit_checks.json`.
- **Solve folders** `solve[-<tag>]/`: `ckpt_t<t>.npz` plus `.json`, `.eval.json` (from `evaluate`), `solve.log`, `runs.jsonl`.
- **`solve` options:** `--resume`, `--tag`, `--budget-hours`, `--checkpoint-hours`, `--log-minutes`, `--batch`, `--iterations`, `--resume-check N`. Without `--resume` it refuses a folder that already has checkpoints. The budget counts from process start, and it refuses to start unless the budget ends at least 30 min before the job's walltime (read from `squeue`).
- **The fit runs one turn deal at a time on a shared generator.** That is bit-identical to one `fit_equity_buckets` call (tested), and it records the generator state before each turn deal. Gate 3's "refit 3 turn deals" needs those states, because the generator runs through all turn deals in order.
- **Slurm scripts** in `scripts/ionic/`:
  - `m1_fit.sbatch`: 4 CPUs, 8G, 4 h. The fit is sequential and single-threaded, so it doesn't use 64 CPUs.
  - `m1_solve.sbatch` and `m1_eval.sbatch`: 64 CPUs, `THREADS` defaults to half the CPUs, 32G.
  - Logs go to `logs/m1/%x-%j.out`. Submit from the repo root.
- **Wait for jobs** with `run_in_background` loops that watch the job's log file (`grep "stopped at t"`), not with bare `sleep` (the harness blocks it), and not by tight-polling `squeue`.

## Facts measured tonight (use them for sizing)
- **Real-deck bucket fit** (spin, Silver 4316): kmeans_2d 2.27 s per turn deal (about 1.3 h for 2,070); avg_1d 0.25 s (about 9 min).
- **Real-deck layout:** 1,035 hands, 2,070 turn deals, 1,892 river children per turn deal. Templates have 1/5/25 entering lines and 14 actions over 6 nodes; the flop root is player 0 with `('c','r')`.
- **Real-deck state:** about 2.4 GB each for regret, cum and sigma. **A checkpoint is about 7.1 GB** (uncompressed). The bucket lookup's temporary is about 1.7 GB at batch 64×4 and about 7 GB at 256×4.
- **Small deck:** 0.07–0.08 s/it at batch 16×4. A lossless smoke checkpoint is about 0.9 GB.
- **Node types in `all`:**
  - node2xx and node302: Gold 5320, 104 threads (the scaling test's type);
  - node0xx below 020: Gold 5218, 64;
  - node020–026 and node10x: Gold 5220R;
  - node403: Gold 6338;
  - node9xx: Silver 4214.
- **Bit-identical comparisons across jobs need the same CPU type, and the same CPU and thread settings,** because floating-point rounding differs by CPU. Pin such jobs with `-w`. The rehearsal used node010.

## Remaining plan

### Step 3: fit the real-deck buckets
```
CFG=bombpot/experiments/configs/holdem
sbatch -w node208 scripts/ionic/m1_fit.sbatch --config $CFG/m1_kmeans2d.yaml
sbatch -w node208 --time=01:00:00 scripts/ionic/m1_fit.sbatch --config $CFG/m1_avg1d.yaml
```
Pick any Gold 5320 node with free CPUs (check `sinfo -p all -N -o "%N %C %e %T"`), but keep node202 and node302 free for the pilot if you can.

Gate 3 is in each run's `fit_checks.json` and `fit.log`: shape (2070, 200, dim), finite, in [0, 1], weakest first, deterministic refits (turn deals 0, 1035 and 2069, plus 0 from the seed), ids well formed, and a clean git hash. Report occupancy (min, median, max hands per bucket) without judging it. Check M1's disk use with `du -sh results/m1` at every gate (limit 300 GB).

### Step 4: real-deck pilot
1. `git fetch` (HTTPS, as above), then `git merge origin/main`. If it brought changes, rerun the full suite (`sbatch scripts/ionic/tests.sbatch`) and `tests/holdem/test_holdem_checkpoint.py` plus `tests/holdem/test_m1.py`: that's a gate. If it conflicts, **stop**.
2. **Main pilot**, on a Gold 5320 node:
   `THREADS=32 sbatch -w node202 --time=01:45:00 --mem=48G scripts/ionic/m1_solve.sbatch --config $CFG/m1_kmeans2d.yaml --tag pilot --budget-hours 0.75 --checkpoint-hours 0.25 --resume-check 2`.
   This gives s/it at batch 64 with 32 threads, peak memory (the "stopped at" line, before the resume check), checkpoint size and save time (the "saved" lines), and **the real-deck resume check**: its PASS or FAIL line, also in the checkpoint's `.json`.
3. **Timing runs**, the same node type, 15 min each, at most 2 solve jobs at once. Chain them on node302 with `--dependency=afterany:<previous>`, each with `--time=00:50:00 --mem=48G --budget-hours 0.25 --checkpoint-hours 1`:
   - `THREADS=64 ... --tag pilot-b64-t64`
   - `THREADS=32 ... --tag pilot-b64-t32`
   - `THREADS=32 ... --batch 256 --tag pilot-b256-t32`
   - `THREADS=64 ... --batch 256 --tag pilot-b256-t64`

   Compare s/it and river deals per second (`batch × 4 / s per iteration`, in each `runs.jsonl` end record).
4. **Exact-check timing** after the main pilot:
   `THREADS=32 sbatch -w node202 --time=00:40:00 --mem=32G scripts/ionic/m1_eval.sbatch --checkpoint results/m1/m1_kmeans2d/solve-pilot/<newest ckpt>.npz --max-minutes 20`.
   The projection is in `<ckpt>.timing.json`.
5. **Gate 4:**
   - resume bit-identical;
   - health checks PASS;
   - rule 7: at most 60 s per iteration, at most 150 GB of memory, a projected exact check of at most 6 days.

   Record:
   - the chosen batch and threads (faster by river deals per second);
   - s per iteration;
   - hours per pass over 2,070 turn deals (= 2070 / batch × s per iteration / 3600);
   - the projected exact-check time.

   If batch 256 wins, change `solver.batch` in both real-deck configs and commit, since `test_m1.py` allows 64 or 256.

### Step 5 (authorized by the user)
- One `m1_solve.sbatch` per method on Gold 5320 nodes, pinned with `-w` (one per node), with `THREADS=<choice>`, `--time=3-00:00:00` and `--mem=<2 × the pilot's peak, rounded up>`. No overrides: the configs already hold `budget_hours: 71.5` and `checkpoint_hours: 2`. Both must start from a clean commit.
- Push `M1_STATUS.md` after each submission, if pushing works.
- **Gate 5, about 2 h after start** (first checkpoint):
  - `m1.py check --checkpoint <it>` loads it and the health check passes (run it locally under `nice` or in a short job; it reads about 7 GB);
  - s per iteration within 25% of the pilot's;
  - memory under its request (`sacct -j <id> --format=MaxRSS` or the log's peak).

  Player 0's flop bet frequencies are in each "saved" log line; report them, but they aren't a gate.
- If a solve hits its walltime: resubmit with `--resume`, at most once overnight.

### Step 7: morning report
Finish `M1_STATUS.md` with the summary `M1_CLAUDE.md` asks for. The long solves are expected to keep running (the user authorized them). Note that, and list their job ids and expected end times. Commit, try the push, then stop. Don't start Step 6 or any other plan step.

## Gotchas
- `tests/test_rules.py` greps `core/` for game names, **including comments**. Never write "Holdem"/"holdem" in `core/`.
- Jobs run from the live working tree. Don't edit `bombpot/` or `tests/` while a job that hasn't imported them yet is pending, and commit before submitting, so runs record a clean hash.
- The smoke configs are tiny on purpose: minute-scale checkpoints, and milestones at 0.02/0.04/0.08 h.
- On the bucketed smoke, exploitability is flat at about 0.273 (the k = 3 floor). Only the lossless smoke has to decrease.
