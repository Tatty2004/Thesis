# Strategy viewer: handoff plan

Written 2026-10-27 ~12:00 EDT for the Claude instance that builds it. The repo is `/n/fs/ugrad/ug27/rb5709/Thesis` on Ionic.

## The goal
The user wants a **GTO Wizard-style viewer of the M1 solver's strategy**. They pick the cards and walk the betting tree action by action. At each decision they see a 13×13 hand grid colored by the action mix, with range weights, and they can click a cell to see each combo's exact strategy.

**The user chose (2026-10-27):** a **web page published privately on claude.ai** (an Artifact). The page loads data files exported from a checkpoint. On the river it computes each hand's bucket in the browser, and tests check that against the Python code. The user rejected a Jupyter notebook and a flop-and-turn-only page.

**Already explained to the user, so don't re-litigate:**
- The flops are fixed (Ah7c2d on board A, KsKd9h on board B). The user picks the turn and river cards, not the flop.
- The betting is limit: check, bet, call, raise and fold, with bets of 2 on the flop and 4 on the turn and river, and at most a bet and one raise per street. **There's no half-pot or pot**, though the user asked for them. Pot-limit runs with ½-pot and pot sizes are planned later (`PLAN_full_solver.md`, steps B1–B5). So **derive every action label and amount from the betting template**, never hard-code them, and the viewer can be reused when those runs exist.
- On the river, strategy is stored per bucket (200 equity buckets per turn deal), so hands in one bucket show the same strategy. Show each hand's bucket.
- The solve is young: the 2 h checkpoint's strategy is still changing fast.

## Read first
1. `/etc/claude-code/CLAUDE.md`: host rules. Never traverse `/n/fs`, keep I/O scoped, no sudo, and wait at least 60 s between Slurm status checks.
2. The root `CLAUDE.md` and `bombpot/holdem/CLAUDE.md`. Layering rule: `core/` never imports a layer. The export script uses Hold'em, so it goes in `bombpot/experiments/`.
3. `M1_STATUS.md`: what's running. **Don't touch the M1 jobs.**

## Constraints
- **Running jobs (as of 11:36):**
  - long solves 31994844 (kmeans_2d, node202) and 31994845 (avg_1d, node204), ending about Fri Oct 30 07:36;
  - exact checks 31995077 (avg_1d, ending about 20:50 Oct 27) and 31995076 (kmeans_2d, ending about 11:30–12:00 Oct 28).

  They've already imported their code. Still, **don't change existing code in `bombpot/` or `tests/`**: a `--resume` would pick it up, and M1 forbids changing verified algorithms. Add new files only.
- **Git:** make branch `viewer` from `m1` and commit there. Commit code and the page source, never exported data. Pushing needs the user's SSH key unlocked (see the memory note `thesis-push-ssh-key`). Commit locally and say so.
- **Data goes under `results/viewer/`** (scratch, `/n/fs/scratch/rb5709`), never in git or home. Home has a 5 GB quota.
- **Run the export locally under `nice`** with `NUMBA_NUM_THREADS=8`, or as a short Slurm job (4 CPUs, 16G). It reads about 2.4 GB of arrays from a 6.6 GB checkpoint.
- **Several login hosts** (spin, rinse) share home but not processes or `/tmp`. Check `hostname` before concluding anything about another session.
- **The user dislikes routine permission prompts.** Act and report. Ask only real design questions.

## The data, exactly
Verified from the code and checkpoint `results/m1/m1_kmeans2d/keep/ckpt_t000001008.npz` (kmeans_2d, t 1008, 2 h of solving).

**Cards:**
- `card = rank * 4 + suit`, ranks `23456789TJQKA` = 0..12, suits `cdhs` = 0..3. So Ah = 50, 7c = 20, 2d = 1.
- `game.flops == ((1, 20, 50), (30, 45, 47))`: board A Ah7c2d, board B KsKd9h (each sorted).

**Hands:**
- `pt.hands` = `itertools.combinations(pool, 2)`, where `pool` is the 46 cards not on either flop, ascending. That's **1,035 hands**, both players sharing one list.
- At any one deal, a hand is impossible if it shares a card with a board card. After the turns that leaves 946 possible; after the rivers, 861.

**Streets and deals:**

| Street t | Deals | Entering lines | Cards |
|---|---|---|---|
| 0, flop | 1 | 1 | fixed |
| 1, turn | 2,070 (46 × 45) | 5 | turn A and turn B. **Export `HoldemRiver(game, pt).turn_cards` (D1 × 2, in deal order); don't recompute the ordering in JS.** |
| 2, river | 1,892 per turn deal (44 × 43, ordered) | 25 | The river strategy forgets the river cards (one row per turn deal), so the page needs the river cards only to compute equity. |

**Betting template:** the same on every street; bets are 2 / 4 / 4. Build it with `bombpot.core.public.betting(game)`, which returns `(templates, contrib)`.

| Node k | Player | History | Actions | Children |
|---|---|---|---|---|
| 0 | P0 | `` | c (check), r (bet) | node 1, node 2 |
| 1 | P1 | `c` | c (check), r (bet) | close 0, node 3 |
| 2 | P1 | `r` | f, c (call), r (raise) | fold, close 1, node 4 |
| 3 | P0 | `cr` | f, c, r | fold, close 2, node 5 |
| 4 | P0 | `rr` | f, c | fold, close 3 |
| 5 | P1 | `crr` | f, c | fold, close 4 |

- **Closes:** `cc`, `rc`, `crc`, `rrc`, `crrc`.
- **Lines on the next street:** `line_next = line * 5 + close`. So the river has 25 lines = flop close × 5 + turn close.
- **`contrib[t][line]`:** the chips each player has in when street t starts. On the flop that's [1]; the turn and river lists are in the template output. The pot at a street's start is `2 * contrib`. Add the in-street bets from the template's histories and the bet size.
- **Payoffs:** each board is worth half the pot.

**Strategy arrays in a checkpoint (uncompressed `.npz`):**
- `cum_t{t}_k{k}`: the cumulative average-strategy weights for street t, node k, shaped `(deals, lines, hands_or_ids, actions)`:
  - flop `(1, 1, 1035, A)`;
  - turn `(2070, 5, 1035, A)`;
  - river `(2070, 25, 201, A)`.

  A is 2 or 3, per the template.
- **The average strategy** is `cum / cum.sum(-1)`, or uniform where the sum is 0 (`SampledRangeCFR.average_strategy`). Ignore `sigma_*` (the current strategy) and `regret_*`.
- **Read only the `cum_*` members.** `np.load(npz)[name]` loads one member at a time. An uncompressed member can also be memory-mapped via its zip offset.

**River buckets (`results/m1/<method>/buckets.npz`, `EquityBuckets.load`):**
- `centers` is `(2070, 200, dim)` float64, weakest first: dim 2 for `kmeans_2d`, 1 for `avg_1d`.
- The id of hand h at a river deal is:
  - `argmin_c sum((x_h - centers[turn_deal, c])**2)` (first minimum on ties), if h is possible;
  - otherwise **200**.
- `x_h` is `(eqA, eqB)` for kmeans_2d, and its mean for avg_1d (`river.features`).
- **Equity** (`CardStreet.equity`, `core/removal.py`): on each board separately, `(wins + 0.5 * ties)` against every opponent hand that avoids the hand's cards and all 10 board cards. Hand strength comes from `holdem/evaluator.hand_values` (best 5 of the 2 hole cards plus that board's 5 cards).
- **Floating point:** Python computes it as `(w + 0.5*t) * kappa / (count * kappa)`, not `(w + 0.5*t) / count`. Near-ties between two centers could therefore resolve differently in JS. **The tests below decide whether you need to reproduce the kappa arithmetic exactly.** `kappa = turn.kappa[parent] * river.scale`; export it if needed.

## Size and how to ship it
Store probabilities as `uint8` (p × 255, rounded) and **drop each node's last action**, which the page reconstructs as 1 minus the rest. That leaves 8 bytes per (deal, line, hand) row across the 6 nodes:

| File | Size |
|---|---|
| `meta.json`: cards, hands, `turn_cards`, template (players, actions, histories, children, closes), `contrib`, bet sizes, checkpoint info (method, t, solve hours, git hash), the encoding | small |
| `flop.bin` | about 8 KB |
| `turn/<turnA>.bin`: 46 shards (one per board-A turn card) × 45 deals × 5 × 1,035 × 8 | 1.9 MB each, **86 MB** in all |
| `river/<turnA>.bin`: 46 shards × 45 × 25 × 201 × 8 | 1.8 MB each, **83 MB** in all |
| `centers/<turnA>.bin`: float64, kept exact for the argmin | 3.3 MB (kmeans_2d) in all |

That's about **175 MB and about 140 files per method**. Artifact limits:
- 16 MB per page;
- 15 MB per binary file;
- 255 files and 64 MB per publish;
- 511 files and 256 MB per version.

So:
- **one artifact per method.** Start with kmeans_2d, the thesis method; avg_1d can be a second artifact if the user wants it;
- **several publishes to the same URL:**
  1. the page, `meta.json`, `flop.bin`, the centers and about half the turn shards;
  2. the rest of the turn shards;
  3. the river shards.

The page `fetch()`es shards by relative path, lazily, when the user picks a turn. If `.bin` is refused as a media type, set `contentType: application/octet-stream`; if that fails too, find out why before falling back to base64 JSON.

**Before writing any file, run `Artifact` `action: "quickstart"` with `intent: "other"`.** The result says a plain page fits, and it carries the page contract: no doctype/html/head/body tags; a `<title>`; theme tokens for light and dark; a 16px gutter; working at phone width; CDN allowlist. The previous instance already ran it, but run it again: its text isn't in your context.

## The page
- **Header:** board A and board B as cards: the flop plus turn and river slots. Pickers for turn A, turn B, river A and river B, disabling collisions and cards already on a board. Show the method, the checkpoint (t, hours of solving) and "average strategy, not converged".
- **Action path:** a breadcrumb of the streets and actions taken, such as "Flop: P0 bet 2 · P1 call | Turn: P0 check". The current node's actions are buttons with amounts and the pot. Folds end the hand. A close moves to the next street, and **the next street asks for its cards first.**
- **13×13 grid:** pairs on the diagonal, suited above it, offsuit below.
  - Each cell aggregates its possible combos, weighted by the **acting player's range weight**.
  - The fill is stacked bars of the action mix.
  - The cell's intensity (or bar height) shows the share of the class still in range.
  - Blocked combos are excluded, and a fully blocked cell is greyed.
- **Range weight of hand h for player p:** the product, over p's earlier decisions on this path, of the probability p's strategy gave the action actually taken. Use the turn strategy at `[turn deal, line]`, and the river strategy at `[turn deal, line, id(h)]` with the ids from the chosen river cards. It's 0 if h shares a card with the board. Don't apply opponent card removal; GTO Wizard doesn't either. Optionally, a toggle shows the other player's range.
- **Summary bar:** the range-weighted action frequencies at this node, and the number of combos in range.
- **Combo inspector:** click a cell to list its combos (e.g. `7d7h`), each with its action mix, range weight, and on the river its bucket id and (eqA, eqB).
- **No EVs.** Checkpoints don't store values. Say so in the page's notes rather than inventing a number.
- **River:** when both river cards are set, compute every hand's (eqA, eqB) in JS: 2 × 861 hand evaluations, then the pairwise counts (about 0.74M per board, fast). Then the ids from that turn deal's centers.
- **Hash routing:** optional. Hashes allow only `#name` (ASCII letters, digits, hyphens), so encoding a full spot won't fit. Skip it unless it's easy.

## Steps, each with its checks
1. **Export script** `bombpot/experiments/viewer_export.py`: `--checkpoint P --out results/viewer/<method>_t<t>/`. It builds `Holdem` from the run folder's `config.yaml` (as `m1.build` does), reads the `cum_*` arrays and writes the files above.
   - **Check:** decoding each file gives the Python average strategy within 1/255 at 200 random (street, deal, line, hand, node) points, and every row sums to 1 within 1/255 × (A − 1).
2. **Test fixtures** (`--fixtures`), written as JSON next to the data:
   - (a) `hand_values` for 500 random 7-card sets;
   - (b) equity and **bucket ids for 100 random river deals × both players** (`EquityBuckets.ids` on `river.street(...)`);
   - (c) range weights from `forward()` at 20 random (turn deal, line, node) points, the reference for the page's own reach math.
3. **JS core** as one plain module, without DOM, that both the page and Node run: card parsing, the evaluator, equity, bucket ids, the template walk, decoding and range weights. `/usr/bin/node` is v16.20.2.
   - **Check** (`node viewer/test_core.mjs <fixtures>`):
     - the evaluator equals (a) exactly;
     - the bucket ids equal (b) **with zero mismatches**. If any differ, reproduce the kappa arithmetic and retest; don't accept "close";
     - the range weights match (c) within 1e-6, allowing for uint8 rounding along the path; set the tolerance from the decode error.
4. **The page** (`viewer/index.html`, plus the core module inlined or published alongside). Build it per the quickstart's contract, then take one rendered look, as the quickstart describes.
   - **Check:** a few spots by hand against Python. For example, flop root: player 0's mix for `7d7h` should equal the checkpoint's `flop_hands` line in `solve.log` at that t (7d7h bet 0.0561 at t 1008 for kmeans_2d). Check also one turn node and one river node.
5. **Publish:** private, in batches to one URL, `icon: "cards"`. Then reply to the user with the link, what was checked, and how to refresh.
6. **Refresh path:** `viewer_export.py` on the newest checkpoint (`ls results/m1/m1_kmeans2d/solve/`), then republish all data files to the same URL. Write the commands in the script's docstring. Checkpoints arrive every 2 h. Retention deletes old ones, but the 2 h ones are hard-linked in `results/m1/*/keep/`.

## Questions for the user, if they come up (don't block on them)
- After kmeans_2d, do they also want avg_1d? That's a second artifact, because of the 256 MB limit.
- Should it show both players' ranges, or only the acting player's (GTO Wizard's default)?
