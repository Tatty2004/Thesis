# Strategy viewer for a value-net solve

A page for walking one flop pair's solve action by action: both boards, a 13x13 hand grid colored
by the action mix, each player's range, and the individual hands of a cell. It runs on your own
machine: `viewer_server.py` listens on localhost only and serves `viewer_page.html`.

Two switches at the top right of the page:

- **Turn and river: Stored trunk / Live re-solve.**
  - *Stored trunk*: the flop and the turn come from the stored strategy (`strategy.npz`); a river
    is re-solved from it.
  - *Live re-solve*: the flop still comes from the stored strategy. When you pick the turn cards,
    that turn is re-solved on your machine for both players (the safe re-solving gadget of
    `core/solvers/leaf_resolve.py`, the value network at every turn end, the opponent held to the
    stored solve's values at the turn root). The river is then re-solved on top of the live turn:
    each player starts from its reach under its own re-solved turn, and the opponent is held to
    the values it got in that turn re-solve (not the stored solve's). This is DeepStack's
    continual re-solving, one street at a time (`live.py`).
- **River solve: Exact / Quick.**
  - *Exact*: the safe river re-solve over all 1,892 river cards of that turn and betting line at
    once, as the grade does.
  - *Quick*: just the river cards you picked, from both ranges, without the gadget (fast, not the
    graded strategy).

Every node shows a badge saying which strategy you are looking at (stored trunk, live turn
re-solve, live river re-solve), and each live solve shows its gadget gap in pots (NashConv over
the root's mass; smaller is better), the iterations kept and the seconds. A progress bar shows
while a solve runs. Everything solved is cached (in memory and on disk), so a second visit is
instant.

Why the live river is over every river card at once: the network values the end of the turn,
one number per hand averaged over the river cards. A safe re-solve of a single river card would
need the opponent's values for that card, which the network doesn't give. In the gadget over all
river cards the opponent decides between entering and its value before the river card is known,
which is exactly what the turn-end values are for.

The flop is always the stored one: a live flop needs a network at the end of the flop, which
doesn't exist yet. The hook is `live.LiveSolver.flop_strategy` / `flop_values`: a flop solve from
the root with a flop-end network, whose averaged values at the turn roots would become the turn
re-solve's T values the way the turn feeds the river.

## On the cluster: pack a solve

```
cd /n/fs/ugrad/ug27/rb5709/Thesis   # a checkout of the branch with the viewer (vn-live until merged)
PYTHONPATH=$PWD /n/fs/scratch/rb5709/envs/torch/bin/python -m bombpot.experiments.valuenet.viewer_pack \
    --trunk results/valuenet/real/<name> --out /n/fs/scratch/rb5709/viewer_pack/<name> [--model <net.pt>]
```

The pack (measured on `g4pre_Td5c2s_As9h6c`: 7.16 GB, 6.66 GiB; packing took about a minute):

| file | size | for |
|---|---|---|
| `strategy.npz` | 4.2 GB | the stored flop and turn |
| `leafcfv32.npy` | 2.9 GB | stored-trunk rivers; the live turn's T values (`backup`, below) |
| `net.pt` | 8 MB | the value network of the live re-solves (`--model`, default the trunk's) |
| `equity_<flops>_13x4.npy` | 17 MB | the network's per-deal hand features (recomputed per deal when missing) |
| `turncfv.npy` | 222 MB | only for trunks run with `real_trunk --turn-cfv` (float32) |
| `trunk.json`, `grade.json` | small | |

Live mode needs nothing beyond what every trunk already stores. The turn re-solve's T values (the
opponent's values at the turn root) come, in this order (`--live-cfv auto`):

1. `avg`: the trunk's turn-root values averaged over its iterations (`turncfv.npy`, from
   `real_trunk --turn-cfv`). This is what passed the small-deck gate T (0.0200 at 300
   iterations against the trunk's 0.0251).
2. `backup`: the trunk's averaged turn-end values (`leafcfv32.npy`) backed up through its average
   turn strategy, no network call and no new trunk. Same numbers as `avg` except that `avg`
   averages each iteration's turn strategy together with that iteration's turn-end values.
   GATE_T_BACKUP
3. `onpolicy`: the average turn strategy's values with the network at its turn-end ranges (one
   network call). GATE_T_ONPOLICY

The g4pre trunks have no `turncfv.npy`, so they use `backup`.

The network is a parameter (`--model`). Use the network the trunk was solved with: the T values
and the stored strategy come from that solve. A newer network (e.g. `net/final_multi_*`) belongs
with a trunk solved with it.

## On the laptop (once)

```
git clone git@github.com:Tatty2004/Thesis.git && cd Thesis && git checkout vn-live   # push vn-live from the cluster first
#   (or update an existing clone: git fetch && git checkout vn-live && git pull)
python3 -m venv .venv && .venv/bin/pip install -e . torch
rsync -avP <netid>@wash.cs.princeton.edu:/n/fs/scratch/rb5709/viewer_pack/<name> ~/viewer_data/
```

## Run

```
.venv/bin/python -m bombpot.experiments.valuenet.viewer_server --trunk ~/viewer_data/<name> --device mps
```

and open http://localhost:8765. `--device mps` (Apple GPU), `cuda` or `cpu`; the default is the
best one available. The first start builds the game and compiles the card kernels (about a
minute). Live options:

| option | default | |
|---|---|---|
| `--live-iters` | 200 | turn re-solve iterations (gate T passed at 100 and 300) |
| `--live-simultaneous` | off | one network pass per iteration: about half the time per iteration, needs about 1.5x the iterations for the same gap |
| `--live-check-every` | 20 | how often the gap is checked; the re-solve ends on the check with the smallest gap (`--live-last`: the last iterate's average) |
| `--live-cfv` | auto | `avg`, `backup`, `onpolicy` or `br` |
| `--live-unpaired` | off | by default both players' re-solves run side by side and share network calls |
| `--model` | `net.pt` in the pack, else the trunk's | |
| `--cache` | the pack folder | where solved rivers and live results are kept (`river_cache/`, `live_cache/<tag>/`) |
| `--warm` | off | load the network at start instead of on the first live request |

If the live turn is too slow on the laptop, `--live-iters 100` or `--live-simultaneous` are the
next settings to try (timings below). Before a demo, click through the spots you plan to show: every
solve is cached.

## Measured

`bench_live.py` through `Viewer.state` (what the page calls), trunk `g4pre_Td5c2s_As9h6c`, network
`sweep_multi_d192L4` with AMP, T values `backup`, 4 random turn roots after three flop lines, wall
seconds for both players' turn re-solves together, gadget gap in pots (median / max over both
players and roots):

| device | setting | live turn | gap median | gap max |
|---|---|---|---|---|
| RTX 2080 Ti (4 CPUs) | alternating 300, paired | 11.6 s | 6.5e-4 | 3.4e-3 |
| | alternating 300, one after the other | 13.7 s | 6.5e-4 | 3.6e-3 |
| | alternating 150, paired | 5.8 s | 1.2e-3 | 4.8e-3 |
| | simultaneous 300, paired | 6.5 s | 1.1e-3 | 2.9e-3 |
| | simultaneous 200, paired | 4.3 s | 2.2e-3 | 7.0e-3 |
| 8 x86 CPU cores (no GPU) | simultaneous 100, paired | 62 s | | |
CPU_ROWS

The default (alternating 200, paired) is about 7.7 s on a 2080 Ti by interpolation. River on the
2080 Ti after a live turn (checked-down turn, viewer `--tol 1e-2`): exact (all river cards, both
players) 10-11 s the first time, including building the card tables; quick 0.3-0.8 s. One-time
setup: about 10 s (game, network, flop strategy) plus about 10 s for the first live solve. Host
memory 2.4 GB, GPU memory 3.8 GB at most (the exact river's card tables).

The network is most of the turn's time on a GPU; on a CPU it is everything: a call takes about
0.3 s on 8 cores against about 10 ms on the 2080 Ti, so on a machine without a usable GPU the
live turn takes minutes. Pairing the players saves only 15% on the 2080 Ti because the turn tree
(numpy, under Python's global lock) can't overlap between the two threads.

Apple hardware (mps) is untested: the code keeps the network on the device and everything in
float64 on the host (`turn_resolve.OnDevice` for the network, `gpu_subgame.HOST64` for the river
re-solver), but nothing here could run it.
