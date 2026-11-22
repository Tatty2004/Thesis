# Strategy viewer for a value-net solve

A page for walking one flop pair's solve action by action: both boards, a 13x13 hand grid colored
by the action mix, each player's range, and the individual hands of a cell. The flop and the turn
come from the stored strategy. A river is solved when it is first visited (both players, all
1,892 river cards of that turn and betting line at once, exactly as the grade does) and then kept
in `<solve>/river_cache/`, so a second visit is instant.

It runs on your own machine: `viewer_server.py` listens on localhost only and serves
`viewer_page.html`.

## On the cluster: pack a solve

```
cd /n/fs/ugrad/ug27/rb5709/Thesis
PYTHONPATH=$PWD /n/fs/scratch/rb5709/envs/torch/bin/python -m bombpot.experiments.valuenet.viewer_pack \
    --trunk results/valuenet/real/<name> --out /n/fs/scratch/rb5709/viewer_pack/<name>
```

About 7 GB: strategy.npz (4.2 GB) and leafcfv32.npy (2.9 GB).

## On the laptop (once)

```
git clone git@github.com:Tatty2004/Thesis.git && cd Thesis && git checkout valuenet
python3 -m venv .venv && .venv/bin/pip install -e . torch
rsync -avP <netid>@wash.cs.princeton.edu:/n/fs/scratch/rb5709/viewer_pack/<name> ~/viewer_data/
```

## Run

```
.venv/bin/python -m bombpot.experiments.valuenet.viewer_server --trunk ~/viewer_data/<name>
```

and open http://localhost:8765. `--device mps` (Apple GPU), `cuda` or `cpu`; the default is the
best one available. The first start compiles the card kernels (about a minute).

Before a demo, click through the river spots you plan to show: they are solved once and cached.

Measured: one river spot (both players) took 4 s on an RTX 2080 Ti and 41 s on 16 x86 CPU threads,
on a 3-iteration test solve; a converged solve needs more iterations. Apple hardware is untested.
