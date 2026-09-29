"""Infoset and terminal counts for two-street, two-board Board-Leduc (rank-level keys).

    python scripts/count_sizes.py [--ranks 4 5 6 7] [--modes shared independent]
"""
import argparse
import time

from toygames.games import BoardLeduc
from toygames.tree import compile_game


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--modes", nargs="+", default=["shared"])
    args = ap.parse_args()
    print("| deck | N | Infosets (both players) | Terminals | Compile (s) |")
    print("|---|---|---|---|---|")
    for mode in args.modes:
        for n in args.ranks:
            start = time.perf_counter()
            tree = compile_game(BoardLeduc(num_ranks=n, num_boards=2, streets=2, deck_mode=mode), keep_meta=False)
            secs = time.perf_counter() - start
            print(f"| {mode} | {n} | {sum(tree.n_infosets):,} | {tree.n_terminals:,} | {secs:.1f} |", flush=True)


if __name__ == "__main__":
    main()
