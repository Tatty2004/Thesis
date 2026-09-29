# CLAUDE.md: Double-Board Toy Games

Toy-game layer of a Princeton COS senior thesis: the first GTO solver for heads-up double-board PLO bomb pots. The contribution is **joint equity bucketing**: grouping hands by strength on both boards at once.

This layer has two jobs:
1. **Prove the code is correct** on games small enough to check exactly.
2. **Measure what bucketing costs**, exactly, and test whether 2D bucketing beats the obvious 1D baseline.

It says nothing about real PLO strategy, and it doesn't test hands where different hole cards hit different boards (that's the double-board Hold'em layer).

## Rules
- **Nothing in `solvers/`, `eval/`, or `abstraction/` may reference a specific game.** They use the `Game` protocol and compiled tree only, and are reused unchanged for Hold'em and PLO.
- **Follow the build order.** Don't start a step while an earlier step's test fails.
- **Don't expand scope.** No Hold'em/PLO evaluation, neural methods, multiway, two-private-card variants, or performance work beyond what the experiment needs. If something seems necessary, stop and ask.

## Stack
Python 3.11+, numpy, scipy (`linprog` with HiGHS), POT (Earth Mover's Distance), pyyaml, pytest. `open_spiel` only inside `tests/` for cross-checks.

## Games

**Kuhn.** Deck J, Q, K. Ante 1, one card each, one betting round, bet 1, at most one bet. Player 0's value is -1/18.

**Leduc.** Must match OpenSpiel's `leduc_poker`. 3 ranks x 2 suits. Ante 1, one card each. Betting (size 2, max 2 raises), one board card, betting (size 4, max 2 raises). Pairing the board wins, else higher card.

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

**Measuring error:**
1. Solve the bucketed game with DCFR to a fixed tolerance. Its exploitability inside the bucketed game is **solver error**.
2. Map the strategy back to the full game and compute exact exploitability. That's **total error**.
3. Report both.

**Grid:** N in {5, 6} (7 if fast), `shared` and `independent`, all methods, k in {2, 3, 4, 5}, 5 k-means seeds each. Output a table and a plot of total error vs k per method.

**Questions:** Does `kmeans_2d` beat `avg_1d`? Does `emd_2d` beat `kmeans_2d` on street 1? Does the gap change between `shared` and `independent`? Negative answers are results too. Don't assert error falls as k grows (Waugh et al. 2009). Log it.

**Done when:** steps 1 to 4 pass, and the table and plot exist for N = 5 and 6.

## Solvers

**DCFR** (Brown and Sandholm 2019), alpha = 1.5, beta = 0, gamma = 2, alternating updates. Positive regrets x t^alpha/(t^alpha+1), negative regrets x t^beta/(t^beta+1), average strategy weighted by (t/(t+1))^gamma. Counterfactual values use opponent reach x chance reach. Same class also does vanilla CFR and CFR+.

**Sequence-form LP** (Koller, Megiddo, von Stengel 1994), player 0:
```
maximize f^T v   subject to  E x = e,  F^T v - A^T x <= 0,  x >= 0
```
Symmetric for player 1. Compare values within 1e-6.

**Exploitability** = NashConv / 2, in chips per hand, on the **average** strategy. Matches OpenSpiel.

## Conventions
- Player 0 acts first each street. `returns()` sums to zero.
- Infoset keys: ranks only, full history from every street, never the opponent's card. Test all three.
- Chance outcomes are rank-level with multiplicities, conditioned on cards already dealt.
- Seed everything. Each run writes config, seed, and git hash to `results/<config hash>/`.
- Log every k iterations: iteration, wall time, exploitability, each player's BR value, game value. Freeze after step 4.

## Gotchas
- Exploitability on the current strategy looks wrong for CFR and misleadingly good for CFR+.
- Swapped DCFR discounts still converge, just slowly. Test against vanilla CFR's value.
- A key missing street-1 history still converges, to the wrong answer. OpenSpiel's Leduc infoset count catches it.
- Raise counters reset each street. Bet size changes each street.
- A tie on one board splits only that half.
- In `independent`, each board's second card comes from that board's own deck copy.

## Layout
```
toygames/
  games/        base.py, kuhn.py, leduc.py, board_leduc.py
  tree.py       compile a Game to flat arrays
  solvers/      cfr.py, lp.py
  eval/         best_response.py, exploitability.py
  abstraction/  features.py, bucketing.py, abstract_game.py
  experiments/  configs/*.yaml, run.py
scripts/count_sizes.py
tests/
results/        gitignored
```

## Build order
1. Week of Sep 29: base, Kuhn, tree, vanilla CFR, best response, LP. Step 1.
2. Week of Oct 6: CFR+, DCFR, Leduc. Step 2.
3. Week of Oct 13: Board-Leduc 1 street, all deck modes. Step 3.
4. Week of Oct 20: 2 streets. Step 4. Benchmark N = 5 to 7. Freeze logging.
5. Late Oct to early Nov: features, bucketing, abstract game, lossless check, grid.

Dates are targets. The order is fixed.

## Open decisions
- Bet sizes (default `[2, 4]`, copied from Leduc).
- Leave Python only if the N = 6 benchmark is too slow for the grid.