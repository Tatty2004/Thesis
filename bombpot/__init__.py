"""GTO solving for heads-up double-board bomb pots.

core/         game-agnostic: game protocols, limit-betting engine, tree compiler,
              solvers, best response and exploitability, bucketing
toygames/     Kuhn, Leduc and Board-Leduc
holdem/       double-board Hold'em
experiments/  the runner, experiment types and configs; may use every layer

core/ imports no layer, and the layers don't import each other (tests/test_rules.py).
"""
