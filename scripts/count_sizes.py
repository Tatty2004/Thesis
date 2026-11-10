"""Sizes of the toy games, and of the no-limit game's betting trees.

    python scripts/count_sizes.py [--ranks 4 5 6 7] [--modes shared independent]
    python scripts/count_sizes.py --pot [--caps 2 3] [--stack 100] [--ante 1] [--sizes 0.5 1]

Default: infoset and terminal counts for two-street, two-board Board-Leduc (rank-level keys).
--pot: the per-street betting tree of the PotPoker game (Holdem(betting="pot"), flop / turn /
river) for every line that reaches the street, walked through the engine itself: decision
nodes, actions, calm closes (a check or call that ends the street with chips behind), folds
and all-in closes (an all-in that is called), and the lines entering each street.
"""
import argparse
import time
from collections import Counter

from bombpot.core.game import CHANCE
from bombpot.core.tree import compile_game
from bombpot.toygames import BoardLeduc


def board_leduc(args):
    print("| deck | N | Infosets (both players) | Terminals | Compile (s) |")
    print("|---|---|---|---|---|")
    for mode in args.modes:
        for n in args.ranks:
            start = time.perf_counter()
            tree = compile_game(BoardLeduc(num_ranks=n, num_boards=2, streets=2, deck_mode=mode), keep_meta=False)
            secs = time.perf_counter() - start
            print(f"| {mode} | {n} | {sum(tree.n_infosets):,} | {tree.n_terminals:,} | {secs:.1f} |", flush=True)


def street_start(game):
    """The state where the first street's betting begins (cards are irrelevant to betting)."""
    s = game.initial_state()
    while game.current_player(s) == CHANCE:
        s = game.next_state(s, game.chance_outcomes(s)[0][0])
    return s


def walk_street(game, s):
    """Count one street's betting from state s. Returns (counts, calm-closing states)."""
    counts = Counter()
    calm = []
    stack = [s]
    while stack:
        s = stack.pop()
        counts["nodes"] += 1
        for a in game.legal_actions(s):
            counts["actions"] += 1
            c = game.next_state(s, a)
            if c.folder >= 0:
                counts["folds"] += 1
            elif game.all_in(c):
                counts["all-in closes"] += 1
            elif c.step != s.step or game.is_terminal(c):
                counts["closes"] += 1
                calm.append(c)
            else:
                stack.append(c)
    return counts, calm


def advance(game, s):
    """From a closing state to the next street's first decision (deal any board cards)."""
    while not game.is_terminal(s) and game.current_player(s) == CHANCE:
        s = game.next_state(s, game.chance_outcomes(s)[0][0])
    return s


def pot_game(args):
    from bombpot.holdem import Holdem

    for cap in args.caps:
        game = Holdem(num_ranks=13, num_suits=4, board_cards=(3, 1, 1), flops=("Ah7c2d", "KsKd9h"), betting="pot",
                      ante=args.ante, stack=args.stack, sizes=args.sizes, max_aggressive=cap)
        print(f"\ncap {cap}: ante {args.ante}, stack {args.stack}, sizes {args.sizes}")
        print("| street | lines in | nodes | actions | calm closes | folds | all-in closes | per-line tree (nodes, actions, closes) x lines |")
        print("|---|---|---|---|---|---|---|---|")
        lines = [street_start(game)]
        allin_lines = 0
        for t, name in enumerate(("flop", "turn", "river")):
            total, shapes, nxt = Counter(), Counter(), []
            for s in lines:
                counts, calm = walk_street(game, s)
                total.update(counts)
                shapes[(counts["nodes"], counts["actions"], counts["closes"])] += 1
                if t < 2:
                    nxt.extend(advance(game, c) for c in calm)
            shape = ", ".join(f"{k} x {v}" for k, v in sorted(shapes.items(), reverse=True))
            print(f"| {name} | {len(lines):,} | {total['nodes']:,} | {total['actions']:,} | {total['closes']:,} | "
                  f"{total['folds']:,} | {total['all-in closes']:,} | {shape} |")
            if t < 2:
                allin_lines += total["all-in closes"]
                lines = nxt
        print(f"lines entering the river with betting: {len(lines):,}; all-in lines reaching the river with no "
              f"betting (called all-in on the flop or turn): {allin_lines:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--modes", nargs="+", default=["shared"])
    ap.add_argument("--pot", action="store_true", help="count the no-limit game's betting trees instead")
    ap.add_argument("--caps", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--ante", type=float, default=1.0)
    ap.add_argument("--stack", type=float, default=100.0)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.5, 1.0])
    args = ap.parse_args()
    pot_game(args) if args.pot else board_leduc(args)


if __name__ == "__main__":
    main()
