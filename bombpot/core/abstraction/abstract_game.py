"""The bucketed game: the full tree with every infoset merged into its bucket's infoset.

An abstract infoset key is (player, bucket on every street so far, public key).
Earlier streets' buckets stay in the key, so the abstract game has perfect recall,
and it is solved and evaluated with the same code as the full game. The terminals
are unchanged (only the players' information gets coarser), so exploitability
inside the abstract game is exact. A strategy maps back to the full game by giving
each infoset its bucket's strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bombpot.core.abstraction.bucketing import Bucketing
from bombpot.core.tree import PerfectRecallError, Tree


@dataclass
class AbstractGame:
    tree: Tree        # the bucketed game
    infoset_map: list  # [p] -> abstract infoset of every full infoset
    seq_map: list      # [p] -> abstract sequence of every full sequence

    def lift(self, sigma_abstract) -> list[np.ndarray]:
        """The full-game strategy that plays each infoset like its bucket."""
        return [sigma_abstract[p][self.seq_map[p]] for p in (0, 1)]


def build_abstract_game(full: Tree, bucketing: Bucketing) -> AbstractGame:
    if full.meta is None:
        raise ValueError("abstraction needs a tree compiled from a CardGame")
    fields, infoset_maps, seq_maps = {}, [], []
    for p in (0, 1):
        meta = full.meta[p]
        full_level = np.repeat(np.arange(len(full.level_infosets[p]) - 1), np.diff(full.level_infosets[p]))
        index, reps, keys = {}, [], []
        imap = np.empty(full.n_infosets[p], dtype=np.int64)
        for i in range(full.n_infosets[p]):
            street, board, hand = meta["street"][i], meta["board"][i], meta["hand"][i]
            buckets = ".".join(str(bucketing.bucket(t, p, board, hand)) for t in range(street + 1))
            key = f"{p}:{buckets}:{meta['public'][i]}"
            j = index.get(key)
            if j is None:
                j = index[key] = len(reps)
                reps.append(i)
                keys.append(key)
            imap[i] = j
        reps = np.array(reps, dtype=np.int64)
        na = full.num_actions[p][reps]
        if not np.array_equal(full.num_actions[p], na[imap]):
            raise ValueError("merged infosets have different actions")
        first = 1 + np.concatenate(([0], np.cumsum(na)[:-1]))
        n_seqs = 1 + int(na.sum())
        seq_map = np.zeros(full.n_seqs[p], dtype=np.int64)
        owner = full.seq_infoset[p][1:]
        seq_map[1:] = first[imap[owner]] + (np.arange(1, full.n_seqs[p]) - full.first_seq[p][owner])
        parent = seq_map[full.parent_seq[p][reps]]
        if not np.array_equal(parent[imap], seq_map[full.parent_seq[p]]):
            raise PerfectRecallError("bucketing merges infosets with different own histories")
        level = full_level[reps]
        if np.any(np.diff(level) < 0):
            raise AssertionError("abstract infosets out of level order")
        n_levels = int(level.max()) + 1
        level_inf = np.searchsorted(level, np.arange(n_levels + 1))
        seq_infoset = np.full(n_seqs, -1, dtype=np.int64)
        seq_infoset[1:] = np.repeat(np.arange(len(na)), na)
        seq_parent = np.zeros(n_seqs, dtype=np.int64)
        seq_parent[1:] = parent[seq_infoset[1:]]
        fields[p] = dict(n_infosets=len(na), n_seqs=n_seqs, parent_seq=parent, first_seq=first, num_actions=na,
                         level_infosets=level_inf, level_seqs=np.append(first, n_seqs)[level_inf],
                         seq_infoset=seq_infoset, seq_parent=seq_parent, keys=keys,
                         actions=[full.actions[p][i] for i in reps])
        infoset_maps.append(imap)
        seq_maps.append(seq_map)

    # Terminals with the same pair of abstract sequences add up (A is bilinear).
    s0 = seq_maps[0][full.term_seq[0]]
    s1 = seq_maps[1][full.term_seq[1]]
    pair = s0 * fields[1]["n_seqs"] + s1
    uniq, inv = np.unique(pair, return_inverse=True)
    if len(uniq) < len(pair):
        s0, s1 = uniq // fields[1]["n_seqs"], uniq % fields[1]["n_seqs"]
        cu0 = np.bincount(inv.ravel(), weights=full.term_cu0, minlength=len(uniq))
    else:  # nothing merged (lossless): keep the full game's terminal order exactly
        cu0 = full.term_cu0.copy()
    name = f"{full.name}|{bucketing.method},k={bucketing.k},seed={bucketing.seed}"
    tree = Tree(name=name, **{k: [fields[0][k], fields[1][k]] for k in fields[0]}, term_seq=[s0, s1], term_cu0=cu0,
                stats={"full_infosets": full.n_infosets, "full_terminals": full.n_terminals})
    return AbstractGame(tree, infoset_maps, seq_maps)
