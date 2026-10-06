# Toy-games layer

Read with the root CLAUDE.md (rules, stack, solvers, conventions).

This layer has two jobs:
1. **Prove the code is correct** on games small enough to check exactly.
2. **Measure what bucketing costs**, exactly, and test whether 2D bucketing beats the obvious 1D baseline.

It says nothing about real PLO strategy, and it doesn't test hands where different hole cards hit different boards. That's the Hold'em layer's job.

**Status: done** (2026-09-28/29, commits `7a76758` to `dad9622`; Checks A and B in `902699e` to `6a09660`). The Hold'em layer uses this layer as the trusted reference its range solver must reproduce.

## Configs and commands
Configs live in `bombpot/experiments/configs/toygames/`:
- `benchmark.yaml`: full-game DCFR for N = 4 to 7, plus the LP up to N = 5.
- `grid.yaml`: the bucketing grid for N = 5, 6 (about 45 min).
- `grid_n7.yaml`: the same grid with N = 7 added (about 1.5 h more).
- `grid_smoke.yaml`: a 1-minute end-to-end check.
- `certify.yaml`: Check B for N = 4 to 7 (see below). Saves each game's strategies.
- `certify_smoke.yaml`: a 20-second end-to-end check of it (N = 4).

Tests are in `tests/toygames/`, about 2 min. `--runslow` adds about 5 min: the reference comparison at N = 4 with two streets, and OpenSpiel's grading at N = 4 (`independent`) and N = 5.

## Games

**Kuhn.** Deck J, Q, K. Ante 1, one card each, one betting round, bet 1, at most one bet. Player 0's value is -1/18.

**Leduc.** Must match OpenSpiel's `leduc_poker`. 3 ranks x 2 suits. Ante 1, one card each. Betting (size 2, max 2 raises), one board card, betting (size 4, max 2 raises). Pairing the board wins, else higher card. Fold is legal only when facing a bet. With rank-level keys it is identical to `leduc_poker(suit_isomorphism=True)` (288 infosets, 1,116 terminals). It also matches the card-level default up to suits: its 936 infosets map onto the 288.

**Board-Leduc** (new, defined here). Always a bomb pot: no betting before the first board cards.

| Param | Values | Default |
|---|---|---|
| `num_ranks` | N >= 3, 2 suits each | 3 |
| `num_boards` | 1, 2 | 2 |
| `streets` | 1, 2 | 1 |
| `deck_mode` | `shared`, `independent`, `identical` | `shared` |
| `ante` | float | 1 |
| `bet_sizes` | per street | `[2, 4]` |
| `max_raises` | per street | 2 |

- **Flow:** ante, one private card each, one card to each board, bet. With 2 streets: a second card to each board, bet. Then showdown.
- **Ranking per board:** your card matches any card on that board = pair of your rank, else high card of your rank. Pair beats high card, higher rank wins, equal splits. Each board wins half the pot. Folds skip evaluation.
- **Deck modes:** `shared` = one deck (real play, boards block each other). `independent` = each board from its own deck copy minus both private cards (no board-to-board blocking). `identical` = board B copies board A, and must reproduce `num_boards=1` exactly.
- **Rule readings** (both implementations follow them, see Check A):
  - A paired board gives nobody a pair: only your own card matching a board card counts. There are no kickers.
  - A tie needs equal private ranks, which tie both boards, so a one-board tie never happens.
  - The opening bet counts toward `max_raises`, so the default allows a bet and one raise per street.
  - In `independent`, both private cards come from one deck.
- **Why 2 streets:** with 1 street every card is known before betting, so it's a tiny river spot. With 2, first-street hands are distributions over future (equity A, equity B), which is what bucketing has to handle.

**Sizes** (2 streets, 2 boards, shared deck, rank-level keys, from `scripts/count_sizes.py`):

| N | Infosets (both players) | Terminals |
|---|---|---|
| 4 | 18,360 | 65,616 |
| 5 | 67,320 | 366,660 |
| 6 | 184,860 | 1,333,080 |
| 7 | 424,116 | 3,779,454 |

Expect LP up to N = 4 (try 5) and full-game DCFR plus exact best response through N = 6 or 7. N = 3 with 2 streets uses the whole deck, so use N >= 4 for bucketing.

## Correctness ladder

| Step | Game | Must pass |
|---|---|---|
| 1 | Kuhn | DCFR exploitability goes to 0. LP value = -1/18. LP player 1 strategy matches the known one. |
| 2 | Leduc | Infoset count, value, and exploitability of the same strategy match OpenSpiel. DCFR matches LP. |
| 3 | Board-Leduc, 1 street | Hand-checked payoffs (scoop, split, chopped board, fold). Chance probs sum to 1. Zero-sum terminals. `identical` = `num_boards=1`. Swapping boards keeps the value. DCFR matches LP. |
| 4 | Board-Leduc, 2 streets, N = 4 | Same as step 3. |

OpenSpiel catches game bugs, LP catches solver bugs, and `identical` extends both to the double-board game.

**Status (2026-09-29): steps 1 to 4 pass.**
- Tests (in `tests/toygames/`): `test_step1_kuhn.py`, `test_step2_leduc.py`, and `test_board_leduc.py` (steps 3 and 4). `test_step5_abstraction.py` covers the bucketing.
- Our CFR, CFR+ and DCFR reproduce OpenSpiel's average strategies at every iteration.
- A lockstep walk checks Leduc against OpenSpiel node by node.

**Independent checks (2026-09-29).** The ladder checks our game against OpenSpiel only for Kuhn and Leduc, and certifies two-board equilibria with our own best response. Two more checks close that gap.
- **Check A: the game is coded to the rules** (`tests/toygames/test_reference.py`).
  - `tests/toygames/reference_board_leduc.py` is a second implementation, written by a separate agent from a rules brief. It never saw the package (CLAUDE.md's prose was in its context). It deals real suited cards, every card equally likely, and has its own betting, pot and showdown code.
  - `tests/game_diff.py` walks both games and compares every rank-level history: chance probability summed over suits, payoff, player to act, legal actions, and the information partition.
  - They agree in all 16 configurations: N = 3 and 4, one and two streets, every deck mode, one and two boards.
    - Probabilities agree to 1e-15 (relative), and payoffs are identical.
    - The largest case, N = 4 `independent` with two streets, compares 121,332 histories built from 3.9 million card-level nodes.
    - Our LP gives the same value on the card-level reference.
  - `tests/toygames/mutants.py` injects 15 plausible bugs into copies of our game, and the comparison catches every one.
  - Limit: a misreading of the rules that both implementations share would pass. The rule readings above are the ones to confirm.
- **Check B: the strategies are equilibria** (`tests/toygames/test_equilibrium.py`, `experiments/certify.py`).
  - **OpenSpiel grading:** `tests/openspiel_game.py` wraps our game as an OpenSpiel Python game.
    - OpenSpiel's own best-response and expected-value code reproduce our best responses, value and exploitability to 1e-9. This holds at N = 4 in every deck mode, at N = 5 (`--runslow`), and for random strategies.
    - OpenSpiel's DCFR reproduces our iterates on two-board games (N = 3, every deck mode, first 5 iterations, to 1e-10).
  - **LP best response:** `best_response_lp` (HiGHS) reproduces our best response to under 1e-9, up to N = 7.
  - **Value brackets:** DCFR, CFR+, vanilla CFR and the LP (N <= 5) give brackets that overlap, and every bracket contains the LP value.
  - **Dominated actions** (`core/eval/audit.py`): the only ones are folding a hand that can't lose. No hand is ever drawing dead. The DCFR strategy plays them at a reach-weighted rate of 1e-10 to 5e-10 (a few folds per 10 billion such spots), far inside the best-response bound.
  - **Board swap:** swapping boards A and B in a strategy changes its exploitability by at most 3e-15.
  - **`certify.yaml` run** (N = 4 to 7, `shared` and `independent`, commit `6a09660`):
    - All 8 games pass.
    - The table is in `results/cd34feafa14e/certify.md`.
    - Each run folder keeps `strategy_dcfr.npz`, plus `strategy_lp.npz` for N <= 5.
  - Limit: this certifies an epsilon-equilibrium with epsilon about 1e-5 chips per hand (DCFR tolerance 1e-5). It is exact only where the LP runs (N <= 5), and only for the game as coded. Check A covers the coding.

## Bucketing experiment

**Features** (vs a uniform opponent range over remaining cards, per public state):
- Street 2: a point (equity A, equity B).
- Street 1: a 2D histogram over street-2 points, one per possible next deal. Its mean is the expected point.

**Methods**, compared at the same bucket count k:
- `avg_1d`: bucket on (equity A + equity B) / 2. **The baseline.**
- `product`: bucket each board separately. Only when k is a perfect square.
- `kmeans_2d`: k-means on the point (or its mean on street 1).
- `emd_2d`: k-means with EMD on street-1 histograms. Street 2 uses `kmeans_2d`.
- `lossless`: one bucket per card. Must match the unabstracted game exactly.

Buckets are per public state. Street-2 keys keep the street-1 bucket (perfect recall).

**Implementation choices** (decided 2026-09-28/29):
- **Opponent range:** "uniform opponent range" is the exact card-removal posterior from the deal model. With `shared` that is uniform over the remaining cards. With `independent` it also accounts for each board's copy excluding both private cards.
- **Street-1 histogram:** one exact atom per possible next deal (identical points merged), not binned.
- **k-means:** every k-means weights hands by probability and keeps the best of 20 k-means++ restarts per public state (lowest weighted sum of squared distances). Each public state gets its own random stream from the seed, shared by both players, so both players get the same buckets.
- **`emd_2d`:** EMD from POT with Euclidean ground distance. Each center is the weighted mixture of its members' histograms, and restarts are compared by the weighted sum of squared EMD.
- **Bucketed game:** the full tree with infosets merged, and terminals summed when they share a pair of abstract sequences. Solver, best response and exploitability run on it unchanged.
- **Solver tolerance:** 5e-5, well below the smallest nonzero total error (0.018).

**Measuring error:**
1. Solve the bucketed game with DCFR to a fixed tolerance. Its exploitability inside the bucketed game is **solver error**.
2. Map the strategy back to the full game and compute exact exploitability. That's **total error**.
3. Report both.

**Grid:** N in {5, 6} (7 if fast), `shared` and `independent`, all methods, k in {2, 3, 4, 5}, 5 k-means seeds each. Output a table and a plot of total error vs k per method.

**Questions:** Does `kmeans_2d` beat `avg_1d`? Does `emd_2d` beat `kmeans_2d` on street 1? Does the gap change between `shared` and `independent`? Negative answers are results too. Don't assert error falls as k grows (Waugh et al. 2009). Log it.

**Done (2026-09-29, commit `dad9622`, N = 5, 6 and 7).** Tables and plots are in `results/f5e9715c3f20/` (N = 5, 6) and `results/0ab89ea4b2e7/` (N = 5 to 7). `results/` is gitignored, so rerun `grid_n7.yaml` to regenerate them. Findings (total error in chips per hand, mean of 5 seeds):
- **`kmeans_2d` does not beat `avg_1d`.** The baseline has lower error in 101 of 120 (N, deck, k, seed) cases.
  - At N = 5 and 6 it wins every cell by 0.014 to 0.09. The exception is N = 5, k = 5, where k covers every hand and the 2D methods are lossless.
  - At N = 7 the gap closes, and `kmeans_2d` edges ahead at shared k = 2 and 5.
  - Brute-force optimal clusterings at N = 6, k = 2 and 3 still put `avg_1d` ahead.
  - A possible reason (hypothesis): each board is half the pot, so showdown value is linear in average equity.
- **`emd_2d` beats `kmeans_2d`, modestly.** It is lower in 87 of 120 cases and tied in 19. The margin is at most 0.017, and 0.01 or less in 17 of 24 cells. At N = 7 it is the best method in 4 of 8 cells.
- **Shared vs independent:** no consistent effect. The 2D penalty is smaller with independent boards at k = 2 and 3 for N = 5 and 6, but not at k = 4 and 5.
- **Other results:**
  - `product` is the worst method at k = 4, with 2 to 4 times the error of `avg_1d`.
  - Total error never increased with k for any method or seed.
  - When k is at least the number of hands per public state (N = 5, k = 5), the 2D methods are exactly lossless.

## Gotchas
- A tie on one board splits only that half.
- In `independent`, each board's second card comes from that board's own deck copy.
- **k-means needs restarts.** A single k-means++ start stalls in a poor local optimum on these few points; it missed the optimum in 16 to 44% of public states. A lower clustering objective doesn't always mean lower exploitability.
- **Exact ties defeat k-means.** Rank-level equities produce exact ties, such as evenly spaced 1D points, which Lloyd's iterations can't break, and restarts don't help. `avg_1d` at k = 3 misses the exact optimum in 5 to 10% of states at N = 5.
- **Some rules can never be tested in Board-Leduc.**
  - A one-board tie never happens: it needs equal ranks, which tie both boards. "A tie splits only that half" is tested through `settle()` directly.
  - No hand is ever drawing dead: the other card of your rank is either on a board, so you pair, or possibly in the opponent's hand, so you might tie.
- **N = 3 is degenerate.** With two shared boards and one street, the value is exactly 0. No hand can both win and lose, so betting extracts nothing.

## Build order
1. Base, Kuhn, tree, vanilla CFR, best response, LP. Step 1.
2. CFR+, DCFR, Leduc. Step 2.
3. Board-Leduc 1 street, all deck modes. Step 3.
4. 2 streets. Step 4. Benchmark N = 5 to 7. Freeze logging.
5. Features, bucketing, abstract game, lossless check, grid.

All five steps were done in order on 2026-09-28/29.

## Open decisions
- Bet sizes (default `[2, 4]`, copied from Leduc). Still the default everywhere.
- ~~Leave Python only if the N = 6 benchmark is too slow for the grid.~~ Resolved: stay in Python. DCFR takes 34 ms per iteration at N = 6 and 88 ms at N = 7. The LP solves N = 5 in 27 to 69 s.
- Whether to extend the grid to larger N (or larger k per public state), since the 1D advantage shrinks at N = 7. Superseded for now by the Hold'em layer (2026-10-06).
- Confirm the Board-Leduc rule readings (under Games). Claude proposed them on 2026-09-29 for Check A. They match the rules text, but the user hasn't explicitly confirmed them.
