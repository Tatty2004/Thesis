# Flop-end net: noisy cheap labels vs fewer exact labels (2026-11-22)

Question: at equal GPU cost, does a flop-end value net trained on many noisy labels beat one
trained on fewer near-exact labels, and would ~100k noisy labels reach the accuracy we need
(turn nets: ~0.009 held-out hand MAE)?

Builds on the cost pilot (branch vn-flopnet, 108565d..3eec50d; 5 states now summarized in
logs/flopnet_pilot/summary_5samples.json). Leaf for every label: the turn-end net
`net/sweep_multi_d192L4/best.pt`, as in the pilot (not the final_multi_* retrains).

## One knob: K, the turn deals solved per label

Every label below the flop-end state: simultaneous DCFR, 100 iterations, the turn-end net at
every turn close, AMP. The arms differ only in how many of the D = 2070 turn deals they solve.

| arm | label | pilot label error (pots, vs 300-it all-deal ref) | A5000-s / label |
|---|---|---|---|
| (a) noisy | K = 50 deals + check-down control variate | 0.0147 | ~29 |
| (c) mid | K = 400 deals + control variate | 0.0055 | ~215 |
| (b) exact | all 2070 deals | 0.0035 (iteration error only) | ~1100 |
| eval | all deals, 200 iterations | 0.0012 | ~2200 |

Cost model: 5.3 ms per deal-iteration on an A5000 (pilot, simultaneous); RTX 2080 Ti ~1.65x
slower. The control variate needs every deal's turn-end equity (~0.2 CPU-s per deal): 3 CPU
workers per job compute the next flop pair's while the GPU solves the current one. States per
flop pair: (a) 8, (c) 2, (b) 1, eval 1 (only (a) needs sharing to hide the equity cost; this
gives (a) fewer distinct flop pairs per label, a handicap against it).

## Sampling and split

States as flop_datagen.sample_state: a random flop pair, a uniform flop-end line (13), random
ranges for both players (datagen.random_range, ordered by flop-end strength). Flop pairs are split
by isomorphism class exactly as the turn-end data (data.heldout_mask, salt real-multi-0, 5%):
training arms draw only from the training side, the eval set only from the held-out side. The
three G.4 test pairs (Td5c2s/As9h6c, AsKh4d/KdQdTd, KhJd9c/7h4c2c) and the three val3 pairs are
never drawn. Within each arm, 10% of its flop-pair classes (salt flopnet-val) are its own
validation split for early stopping, in its own label type; the exact eval set never selects
anything.

## Budget (matched at ~55 A5000-h of labelling per arm)

| | labels | A5000-h | GPU-h projected (incl. slower GPUs, startup) |
|---|---|---|---|
| eval (held-out pairs, exact) | 90 | 55 | ~60 |
| (b) exact | ~180 | 55 | ~60 |
| (c) K = 400 | ~920 | 55 | 60-95 |
| (a) K = 50 | ~6,800 | 55 | 60-95 |
| training (~25 small runs) | | | ~15 |
| smoke | | | ~3 |
| **total** | | | **~260-330** (cap 500; stop and report if the projection passes 400) |

(b) and eval run as 3 h tasks (larger GPUs only, node007 excluded); (a) and (c) as 1 h tasks
(any GPU). At most ~38 GPUs at once (array throttles), below the 40 agreed.

Free same-state controls: every (b) state also stores K = 50 and K = 400 control-variate labels
drawn from its own solved deals (`k50cfv`, `k400cfv`), so nets trained on the same ~180 states
with exact / K = 400 / K = 50 labels isolate the effect of label noise at equal N.

## Training and evaluation

Net: HandSetNet with attention as the turn nets, scaled down to d128 L3 (4 heads), flop-end
inputs (3-card boards; per-hand flop-end equity and its spread over the turn deals as the
extra features). Config flopnet_test.yaml: Adam 1e-3, batch 16, clip 1, AMP, cosine over
max(40, min(400, 1.5M / N)) epochs, best epoch by the arm's own validation loss.

Score: hand MAE in pots (train.py's metric, range-weighted over both players) on the 90 exact
held-out states, at the best-validation epoch.

Learning curves (nested subsets, first N labels): (a) N = 250, 500, 1k, 2k, 4k, all; (c) 250,
500, all; (b) 90, all; same-state controls (b states with k400 / k50 labels). Two seeds for N <=
1k. Read-outs: MAE vs N per arm, MAE vs labelling GPU-h per arm (the matched comparison), and a
power-law fit of (a) extrapolated to 100k labels.

## Order

1. Commit code + this plan; smoke (one short (a) task, one short all-deal task on held-out pairs,
   a tiny training on them); check speed, memory, labels against the pilot's numbers.
2. Launch eval + (b) + (c) + (a) arrays; re-project the cost after the first tasks finish.
3. Train the curves; report.
