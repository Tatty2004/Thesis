# Hold'em layer

Read with the root CLAUDE.md (rules, stack, solvers, conventions).

**Goal:** solve and measure real heads-up double-board Hold'em bomb pots: a 52-card deck and standard hand ranking. Everything built here is what real Hold'em, and later PLO, runs on.

The toy games can't test the case that matters: in Board-Leduc one private card decides both boards, so equity on board A and on board B move together. With two hole cards, one card can hit board A and the other board B.

**Agreed (2026-10-05/06):**
- Build toward the real 52-card game from the start. Small decks survive only as test configurations of the same game. The separate Board-Hold'em experiment and the river-spot experiment are dropped.
- Board-Leduc's ante and bet sizes for the small test games.
- If compute gets tight, the user can get Princeton cluster access (see Compute).

## Why the tree solver can't do this
`core/tree.py` stores every terminal, which tops out around 10 million. After one pair of flops, real double-board Hold'em has about 10^11 terminals with flop and turn betting, and about 10^15 with the river.

Two boards square the runouts: 46 x 45 = 2,070 turn deals and 3.9 million turn-and-river deals per flop pair, against 2,352 runouts on one board. At 10^11 terminals the arrays alone take terabytes, so no cluster holds them either. The fix is the algorithm.

**Bomb pots split into flop pairs.** Nobody acts before the flops, so given the flops both ranges are uniform over the remaining cards. Each pair of flops is an independent game, and the full game's value and exploitability are averages over flop pairs. Solving sampled flop pairs studies the real game, and they are independent jobs.

## Algorithm
Still **DCFR** (alpha = 1.5, beta = 0, gamma = 2), with the same math organized around ranges ("vector-form" CFR, as in commercial solvers and DeepStack).
- **Public tree.** A node is the boards plus the betting. Every (hand, public node) pair is one infoset, so this is exactly CFR on the same game.
- **Down the tree:** each player's reach is a vector over hands. At the acting player's node, the strategy is a hands x actions matrix, and each action scales that player's reach by its column. At a deal, hands that share a card with the new board cards drop out.
- **At terminals:** each hand's counterfactual value is the payoff applied to the opponent's reach vector.
  - **Fold:** the opponent's total reach over hands that don't share a card with mine, times the payoff.
  - **Showdown:** sort hands by strength on the board and take prefix sums of the opponent's reach. "Reach of weaker hands minus reach of stronger hands" then costs O(n), not O(n^2). Card removal is corrected exactly with per-card reach sums (inclusion-exclusion) (Johanson et al. 2011).
  - Each board is half the pot, so the two boards' showdown terms are computed separately and added.
- **Up the tree:** values combine as usual, and each hand updates its own regrets with DCFR discounting.
- **Bucketing:** per public state, hands map to buckets, and regrets and strategies are summed per bucket. Exploitability is one best-response pass on the unbucketed game.
- **River (R5):** a full pass over 3.9M runouts per iteration is too slow, so each iteration samples the turn and river cards (public chance sampling, Johanson et al. 2012) and keeps the vector form over hands.

## The game (`holdem/game.py`)

| Param | Values | Default |
|---|---|---|
| `num_ranks` | 2 to 13: the lowest N of 2, 3, ..., A | 13 |
| `num_suits` | 1 to 4 | 4 |
| `board_cards` | cards dealt to each board on each street | `[3, 1]` (flop, turn) |
| `num_boards` | 1, 2 | 2 |
| `deck_mode` | `shared`, `identical` | `shared` |
| `flops` | each board's street-1 cards, fixed instead of dealt | none |
| `ante`, `bet_sizes`, `max_raises` | as in Board-Leduc | 1, `[2, 4]`, 2 |

- **Flow:** ante. Two private cards each, player 0 first, from one deck. On each street, `board_cards[t]` cards go to each board (A, then B), then a limit betting round. Then showdown. There is no betting before the first board cards.
- **Hand on a board:** the best five-card hand from your two cards plus that board's cards (all of them when there are fewer than five). Standard ranking: straight flush, quads, full house, flush, straight, trips, two pair, pair, high card, with kickers. The ace plays low only in A-2-3-4-5, so the wheel needs the full deck.
- **Each board is worth half the pot.** Equal hands split that board's half. Board cards play for both players.
- **Deck modes:** `shared` = one deck (real play). `identical` = board B copies board A, and must reproduce `num_boards=1`.
- **Cards:** `card = rank * num_suits + suit`, written like `Ah` (ranks `23456789TJQKA`, suits `cdhs`). A hand or a street's board cards are dealt as one chance event, sorted.
- **Keys:** `player:hole cards:board A|board B:betting`.
- These are exactly the rules of OpenSpiel's `universal_poker` (ACPC) on one board, including board cards dealt before the first betting round.

## Steps

| Step | Build | Must pass |
|---|---|---|
| R1 | **The game** (above), card-level, with the hand evaluator. | Evaluator: hand-checked cases; the direct algorithm equals the best of every five-card subset; agrees with ACPC's evaluator on random 5-, 6- and 7-card deals (forcing cards through `universal_poker`), full and short decks. Game on tiny decks: one board matches `universal_poker` under `game_diff`; two boards match a card-by-card reference whose showdowns come from ACPC; `identical` = one board; swapping boards keeps the value; zero-sum, chance sums to 1; DCFR = LP; OpenSpiel's grading reproduces our best responses. |
| R2 | **Range solver** (new, in `core/solvers/`, game-agnostic). The game supplies each public state's hands and its showdown and fold payoffs as operators on a range vector. Start with dense matrices. | Reproduces the tree solver's DCFR iterates and best responses to rounding error on Board-Leduc and tiny Hold'em decks, full and bucketed. |
| R3 | **Fast showdown**: sort by strength, take prefix sums, correct for card removal. The same trick computes equity features. | Equals the dense operator on random ranges at small decks. Real deck: board swap and suit-permutation symmetry. |
| R4 | **Real 52-card game, flop and turn, per flop pair.** | Full solve and exact exploitability on a few flop pairs, with time per iteration and memory measured (estimated at seconds and about 2 GB). Then the bucketing grid over sampled flop pairs, with k up to about 50 to 100 and `emd_2d` on flop histograms over 2,070 turn points. |
| R5 | **Add the river** with sampled runouts (Monte Carlo CFR). Abstraction becomes necessary. | Exploitability estimated: an exact best response per flop pair is about 10^12 operations (hours), and local best response gives a lower bound. Template for PLO. Decide after R4's timings. |

No river is the one artificial cut in R4. It is the largest real-deck game that stays exactly measurable.

## Compute
- R1 to R3 run on the laptop (M1 Pro, 8 cores, 16 GB).
- R4's grid is many independent flop-pair solves. If the laptop is too slow, use Princeton's **Adroit** cluster: any NetID holder can request an account through a form, with no sponsor needed. **Della** needs a faculty sponsor (ask the thesis advisor).
- Tell the user as soon as a step looks too slow for the laptop.

## Build order (targets)
1. Week of Oct 5: R1.
2. Week of Oct 12: R2.
3. Week of Oct 19: R3.
4. Late Oct to Nov: R4.
5. Then decide on R5.

## Status
- 2026-10-06: package reorganized into `core/`, `toygames/` and `holdem/` (commit `cf24d8c`).
- **R1 passes (2026-10-06).** Code: `holdem/cards.py`, `evaluator.py`, `game.py`. Tests in `tests/holdem/` (about 25 s, `--runslow` adds about 5 min):
  - **Evaluator** (`test_holdem_evaluator.py`):
    - Agrees with ACPC's evaluator on about 30,000 random showdowns (52 cards with 3, 4 and 5 board cards; short decks with fewer than five cards) and on hand-picked close calls (wheel, steel wheel, quads vs quads, board full house).
    - Equals the best of every five-card subset on 6,000 random 6- and 7-card hands.
    - Reproduces the textbook category counts over all 2,598,960 five-card hands (slow).
  - **One board vs `universal_poker`** (`test_holdem_reference.py`): `game_diff` agrees on every chance probability, payoff, player and legal action in 9 configurations, the largest with 1.05 million terminal records.
    - One street: the information partitions are identical too.
    - Two streets: `universal_poker`'s information string sorts all board cards together, so it forgets which street a card came on. It merges infosets we keep apart, and never the reverse. With our key made equally street-blind, the partitions are identical.
  - **Two boards vs `tests/holdem/reference_holdem.py`:** a naive implementation that deals card by card, has its own betting and chip code, and has ACPC judge every showdown. It agrees in 7 configurations (`shared` and `identical`, one and two streets). 9 injected bugs (`holdem_mutants.py`) are all caught.
  - **Game** (`test_holdem_game.py`):
    - Hand-checked 52-card pots: scoop, split where different hole cards hit different boards, a chopped board, folds, the biggest pot, flush and wheel.
    - Structure: `identical` gives the same tree as one board, and swapping boards changes no deal, showdown or exploitability.
    - **A bomb pot is the probability-weighted average of its fixed-flop games** (LP values agree to 1e-9).
    - Solvers and keys: DCFR = LP, keys hold own cards, boards street by street and betting, and nothing else.
    - OpenSpiel's best response and on-policy value reproduce ours to 1e-9.
  - Limit: the two-board reference was written by Claude, the same author as the game, so it is less independent than Board-Leduc's Check A. ACPC judges its showdowns, and `universal_poker` independently covers all of one-board play.
- Card-level trees grow fast: N = 4, S = 2 with two boards and two streets already has 307k infosets and 474k terminals. The tree solver only handles tiny decks, and R2's range solver takes over from there.
- R2 is next.

## Open decisions
1. **Bet sizes for the real game** (limit, sizes to be decided). Not needed until R4.
2. **Where the layer ends:** R4, or R5 as well. Decide after R4's timings.
