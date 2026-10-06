# CLAUDE.md: double-board bomb-pot solver

Princeton COS senior thesis: the first GTO solver for heads-up double-board PLO bomb pots. The contribution is **joint equity bucketing**: grouping hands by strength on both boards at once.

The work is built in layers. Each layer is a package under `bombpot/` with its own CLAUDE.md for its games, checks, results and build order:

| Layer | Package | Job | Status |
|---|---|---|---|
| Core | `bombpot/core/` | Game protocols, limit-betting engine, tree compiler, solvers, best response, bucketing. Game-agnostic. | in use |
| Toy games | `bombpot/toygames/`, [CLAUDE.md](bombpot/toygames/CLAUDE.md) | Prove the code correct on games small enough to check exactly, and measure what bucketing costs. | done 2026-09-29 |
| Hold'em | `bombpot/holdem/`, [CLAUDE.md](bombpot/holdem/CLAUDE.md) | Real double-board Hold'em: two hole cards, a 52-card deck, and a range solver that scales. | in progress |
| PLO | later | The thesis target. | |

`bombpot/experiments/` runs the experiments of every layer.

## Rules
- **`core/` is game-agnostic.** It never imports a layer or names a specific game. It sees games only through the `Game` and `CardGame` protocols and the compiled tree.
- **Layers build on `core/` only and never import each other.** `experiments/` may use everything. `tests/test_rules.py` enforces both rules.
- **Follow each layer's build order.** Don't start a step while an earlier step's test fails.
- **Don't expand scope.** No neural methods, multiway, or performance work beyond what the current step needs. If something seems necessary, stop and ask.

## Stack
Python 3.11+, numpy, scipy (`linprog` with HiGHS), numba (the range solver's sweeps), POT (Earth Mover's Distance), pyyaml, pytest, matplotlib (plots only, in `experiments/`). `open_spiel` only inside `tests/` for cross-checks.

Use `.venv/bin/python` (3.11, package installed with `pip install -e .`). The system `python3` is 3.9. OpenSpiel's `sequence_form_lp` needs cvxpy, which isn't installed, so tests use its `exploitability`, `expected_game_score` and CFR solvers instead. OpenSpiel's `universal_poker` (the ACPC poker engine) is available and is the outside reference for Hold'em rules.

## Commands
- Tests (about 2.5 min): `.venv/bin/python -m pytest`. Add `--runslow` for the longest cross-checks (about 10 min more). Run one layer with `tests/toygames` or `tests/holdem`.
- Experiments: `.venv/bin/python -m bombpot.experiments.run bombpot/experiments/configs/<layer>/<name>.yaml --workers 6`
  - Each layer's CLAUDE.md lists its configs.
  - `--force` reruns stored runs. `--reuse-any-commit` accepts runs stored by an older clean commit.
- Sizes of the toy games: `.venv/bin/python scripts/count_sizes.py --modes shared independent`

## Solvers

**DCFR** (Brown and Sandholm 2019), alpha = 1.5, beta = 0, gamma = 2, alternating updates. Positive regrets x t^alpha/(t^alpha+1), negative regrets x t^beta/(t^beta+1), average strategy weighted by (t/(t+1))^gamma. Counterfactual values use opponent reach x chance reach. Same class also does vanilla CFR and CFR+.

**Sequence-form LP** (Koller, Megiddo, von Stengel 1994), player 0:
```
maximize f^T v   subject to  E x = e,  F^T v - A^T x <= 0,  x >= 0
```
Symmetric for player 1. Compare values within 1e-6.

**Exploitability** = NashConv / 2, in chips per hand, on the **average** strategy. Matches OpenSpiel.

**Implementation.** `core/tree.py` compiles a game into sequence-form arrays. Each player's sequences are grouped by level. Each terminal is stored as (player 0's last sequence, player 1's last sequence, chance x payoff). A CFR iteration or a best response is then one gather and one bincount over the terminals, plus one pass per level of sequences. This stores every terminal, so it tops out around 10 million terminals.

**Range solver.** `core/public.py` and `core/solvers/range_cfr.py` run the same algorithms on the public tree, with each player's range as a vector over hands. Cost grows with public states x hands, not with hand pairs. It reproduces the tree solver exactly (`tests/test_range_cfr.py`). Big games use card-removal streets (`core/removal.py`): fold and showdown sums by inclusion-exclusion over each hand's card subsets and a sweep in strength order, compiled with numba.

## Conventions
- Player 0 acts first each street. `returns()` sums to zero.
- Infoset keys: own private cards plus everything public, with the full history from every street, and never the opponent's cards. Format `player:hand:boardA|boardB:betting` (`experiments/certify.py` relies on it).
- Chance outcomes are conditioned on cards already dealt.
- Seed everything. Each run writes config, seed, and git hash to `results/<config hash>/` (`config.yaml`, `meta.json`, `log.csv`, `result.json`).
  - A sweep's summary goes to `results/<sweep config hash>/`. Trees and features are cached in `results/cache/`.
  - The git hash is read once per invocation and passed to the workers. "Dirty" means uncommitted changes under `bombpot/` or in `pyproject.toml`.
  - A stored run is reused only when it came from the current clean commit.
- Solver logs are **frozen** as `iteration, wall_time, exploitability, br_value_p0, br_value_p1, game_value`. `wall_time` counts seconds spent iterating and excludes logging.
- Commit every new version, one commit per working step or fix, and push to `origin` (github.com/Tatty2004/Thesis). Run experiments from a clean commit so their results can be reused.
- Saved strategies: `strategy_dcfr.npz` and `strategy_lp.npz` in a run folder hold {infoset key: action probabilities}. `bombpot.core.tree.load_strategy(path)` loads one. `.table()` gives the dict, and `.sigma(tree)` gives the arrays for a compiled tree.

## Gotchas
- Exploitability on the current strategy looks wrong for CFR and misleadingly good for CFR+.
- Swapped DCFR discounts still converge, just slowly. Test against vanilla CFR's value.
- A key missing street-1 history still converges, to the wrong answer. OpenSpiel's Leduc infoset count catches it.
- Raise counters reset each street. Bet size changes each street.
- **HiGHS tolerances are absolute.** Chance-weighted payoffs are tiny, so an unscaled best-response LP stops about 1e-8 short of the optimum. `best_response_lp` scales the costs so the largest entry is 1.
- Pickled caches in `results/cache/` refer to module paths. `features_for` recomputes features whose pickle no longer loads.
- OpenSpiel's `universal_poker` information string sorts all public cards together, so with board cards on two rounds it lacks perfect recall. Compare partitions with a street-blind key (`tests/holdem/test_holdem_reference.py`).

## Layout
```
bombpot/
  core/          game.py (Game and CardGame protocols), limit.py (limit-betting engine, LimitPoker),
                 tree.py (compile a Game to sequence-form arrays, strategy utilities, saved strategies),
                 public.py (public tree for range solvers: betting templates, deals, fold/share operators),
                 removal.py (card-removal fold/showdown sums in O(hands) per deal, compiled)
    solvers/     cfr.py (CFR, CFR+, DCFR on the tree), range_cfr.py (the same on the public tree, ranges as
                 vectors), lp.py (sequence-form LP, best response as an LP)
    eval/        best_response.py, exploitability.py, audit.py (dominated actions)
    abstraction/ features.py, bucketing.py, abstract_game.py
  toygames/      kuhn.py, leduc.py, board_leduc.py, CLAUDE.md
  holdem/        cards.py (any deck size), evaluator.py (best five-card hand), game.py (Holdem),
                 public.py (the public tree with card-removal streets, up to 52 cards), CLAUDE.md
  experiments/   games.py (every game a config can name, and its public tree), run.py (runner and bookkeeping;
                 `solver.backend: range` solves with the range solver),
                 grid.py (bucketing grid, table, plot), certify.py (equilibrium certification),
                 configs/<layer>/*.yaml
scripts/count_sizes.py
tests/           conftest.py (--runslow), helpers.py, game_diff.py, openspiel_game.py (shared tools),
                 test_rules.py (layering rules), test_range_cfr.py (range solver vs tree solver), toygames/, holdem/
results/         gitignored
pyproject.toml
```
