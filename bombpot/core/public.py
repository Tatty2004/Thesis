"""The public tree of a poker game, for solvers that work on ranges.

A public node is (street t, deal d, line class c, entering line l, betting node k):
  deal d       the public cards so far (one entry of `deals(t)`'s boards)
  line l       the betting on earlier streets. Street t's lines are grouped into classes
               (`pt.classes[t]`, a list of LineClass) whose betting has the same shape: the
               same nodes, players and actions. l indexes the lines of class c.
  node k       a node of class c's betting template, the same for every deal (betting ignores
               the cards) and every line of the class (only the chips differ between them)
Every (private hand, public node) pair is one infoset of the full game.

Limit betting has one class per street, whose lines are the mixed-radix index over each
street's closing lines (lines of street t + 1 = lines of t x closes of t). With pot-fraction
sizes and stacks (core/betting.py) the chips differ per line and an all-in truncates the
betting, so a street can have several classes; a class whose lines are all-in has a template
with zero nodes and one close (no betting; on the last street that close is the showdown).
Street t + 1's lines are ordered by (parent class, parent line, close), and its classes by
their first line in that order, so a one-class game keeps the mixed-radix order.

Chance enters only through W[t][d, i, j] = P(hand0 = i, hand1 = j, public cards = deal d),
so a counterfactual value is a matrix-vector product: player 0's value at a fold is
payoff x (W @ reach1), and at a showdown pot x (S @ reach1) - contribution x (W @ reach1)
with S = W x player 0's average board share. Each Street supplies these two products
(`fold`, `share`). build_public_tree makes them dense, from `CardGame.deals` and
`showdown`; a big game supplies a faster Street with the same methods.

The templates are read off the game's betting engine itself (LimitPoker or PotPoker), by
walking every line.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bombpot.core.game import CHANCE


@dataclass
class Template:
    """One class's betting on one street. Chips are relative to the street's start, when both
    players have the same amount in, and are (L_c,) arrays, one entry per line of the class
    (the shape is shared, the chips are not)."""
    players: list   # [k] acting player at decision node k (nodes in creation order: parents first)
    actions: list   # [k] legal actions at node k
    hist: list      # [k] the street's betting string before node k acts
    children: list  # [k][a] ("node", k2) | ("fold", folder, chips the folder added) | ("close", j)
    closes: list    # [j] (betting string, chips each player added) for each line that ends the street
                    # without a fold (a call, a check behind, or an all-in and its call)

    @property
    def num_nodes(self) -> int:
        return len(self.players)

    @property
    def num_closes(self) -> int:
        return len(self.closes)


class Street:
    """One street's deals and the chance-weighted operators the range solver needs.

    For player 0's reach r1 over player 1's hands (and symmetrically for player 1):
      fold(r1, 0)[d, l, i]  = sum_j W[d, i, j] r1[d, l, j]
      share(r1, 0)[d, l, i] = sum_j S[d, i, j] r1[d, l, j]
    with W = P(hand0, hand1, deal) and S = W x player 0's average share of the boards
    (only needed on the last street)."""
    deals: list          # [d] public cards (the game's board tuple) at this street
    parent: np.ndarray   # [d] index of the deal's prefix on the previous street (-1 on street 0)
    index: dict          # deal -> d

    def fold(self, reach: np.ndarray, player: int) -> np.ndarray:
        raise NotImplementedError

    def share(self, reach: np.ndarray, player: int) -> np.ndarray:
        raise NotImplementedError

    def possible(self, player: int) -> np.ndarray:
        """(D, H): whether the player can hold each hand at each deal."""
        raise NotImplementedError

    @property
    def num_deals(self) -> int:
        return len(self.deals)


class DenseStreet(Street):
    """W and S as dense (D, H, H) arrays: exact and simple, for small games."""

    def __init__(self, deals, parent, W, S=None):
        self.deals, self.parent, self.W, self.S = deals, parent, W, S
        self.index = {b: i for i, b in enumerate(deals)}

    @staticmethod
    def _apply(M, reach, player):
        if player == 0:
            return np.einsum("dij,dlj->dli", M, reach)
        return np.einsum("dij,dli->dlj", M, reach)

    def fold(self, reach, player):
        return self._apply(self.W, reach, player)

    def share(self, reach, player):
        return self._apply(self.S, reach, player)

    def possible(self, player):
        return (self.W.sum(axis=2) if player == 0 else self.W.sum(axis=1)) > 0


@dataclass
class LineClass:
    """The lines of one street that share a betting shape.

    lines         (L_c,) the lines' indices among all of the street's lines (in street order)
    contrib       (L_c,) chips each player has in when the street's betting starts
    stack         (L_c,) chips each player has behind then
    parent_line   (L_c,) each line's index within its parent class on the previous street
    parent_close  (L_c,) the parent template's close that leads to it
    parent_class  (L_c,) the parent class (all -1, like the two above, on street 0)
    """
    template: Template
    lines: np.ndarray
    contrib: np.ndarray
    stack: np.ndarray
    parent_line: np.ndarray
    parent_close: np.ndarray
    parent_class: np.ndarray

    @property
    def num_lines(self) -> int:
        return len(self.lines)


@dataclass
class PublicTree:
    name: str
    hands: list           # private hands (one list for both players)
    streets: list         # [t] Street (DenseStreet for small games)
    classes: list         # [t][c] LineClass; may cover more streets than `streets` (sampled_cfr)

    @property
    def num_streets(self) -> int:
        return len(self.streets)

    def num_lines(self, t: int) -> int:
        return sum(lc.num_lines for lc in self.classes[t])

    def shape(self, t: int, c: int) -> tuple:
        """(D_t, L_c, H): the arrays of class c on street t."""
        return (self.streets[t].num_deals, self.classes[t][c].num_lines, len(self.hands))

    def histories(self, t: int) -> list:
        """[c][l] the betting strings of the earlier streets that lead to line l of class c."""
        out = [[()] * lc.num_lines for lc in self.classes[0]]
        for u in range(t):
            prev = out
            out = [[prev[pc][pl] + (self.classes[u][pc].template.closes[pj][0],)
                    for pc, pl, pj in zip(lc.parent_class, lc.parent_line, lc.parent_close)]
                   for lc in self.classes[u + 1]]
        return out


def _advance(game, s):
    """Through chance (any outcome: betting ignores the cards) to the next decision or the end."""
    while not game.is_terminal(s) and game.current_player(s) == CHANCE:
        s = game.next_state(s, game.chance_outcomes(s)[0][0])
    return s


def _walk(game, start):
    """The betting from `start` (a street's first decision) to the end of the street: the
    shape key, the template with this line's chips (scalars), and each close's end state."""
    t = game.street(start)
    base = start.contrib[0]
    players, actions, hist, children, closes, ends = [], [], [], [], [], []
    queue = [start]
    while queue:
        s = queue.pop(0)
        k = len(players)
        players.append(game.current_player(s))
        acts = tuple(game.legal_actions(s))
        actions.append(acts)
        hist.append(s.hist[t])
        row = []
        for a in acts:
            c = game.next_state(s, a)
            if c.folder >= 0:
                row.append(("fold", c.folder, c.contrib[c.folder] - base))
            elif game.is_terminal(c) or c.step != s.step:  # showdown, or the street is over
                row.append(("close", len(closes)))
                closes.append((c.hist[t], c.contrib[0] - base))
                ends.append(c)
            else:
                row.append(("node", k + 1 + len(queue)))
                queue.append(c)
        children.append(row)
    key = (tuple(hist), tuple(actions))
    return key, Template(players, actions, hist, children, closes), ends


def _class_template(templates: list) -> Template:
    """One template for lines of the same shape: their chips stacked into (L_c,) arrays."""
    first = templates[0]
    children = []
    for k, row in enumerate(first.children):
        out = []
        for a, child in enumerate(row):
            if child[0] == "fold":
                out.append(("fold", child[1], np.array([tp.children[k][a][2] for tp in templates], dtype=float)))
            else:
                out.append(child)
        children.append(out)
    closes = [(h, np.array([tp.closes[j][1] for tp in templates], dtype=float)) for j, (h, _) in enumerate(first.closes)]
    return Template(first.players, first.actions, first.hist, children, closes)


ALL_IN = ("all-in",)  # the shape key of a line with no betting left


def betting(game) -> list:
    """Every street's line classes (see LineClass), walking the game's betting engine line by
    line: the betting a line has depends on its pot and stacks."""
    total = getattr(game, "total", None)  # chips each player starts with (stacked games)
    T = game.num_streets
    # Lines entering the current street: (state at its first decision, or None if there is no
    # betting left; chips each player has in; parent class, parent line, parent close).
    s0 = _advance(game, game.initial_state())
    entering = [(s0, s0.contrib[0], -1, -1, -1)]
    classes = []
    for t in range(T):
        groups: dict = {}
        for n, (s, chips, pc, pl, pj) in enumerate(entering):
            if s is None:
                key, tp, ends = ALL_IN, Template([], [], [], [], [("", 0.0)]), [None]
            else:
                key, tp, ends = _walk(game, s)
            groups.setdefault(key, []).append((n, chips, pc, pl, pj, tp, ends))
        street, nxt = [], []
        for c, members in enumerate(groups.values()):
            n, chips, pc, pl, pj, tps, ends = (list(x) for x in zip(*members))
            chips = np.array(chips, dtype=float)
            stack = (total - chips) if total is not None else np.full(len(chips), np.inf)
            street.append(LineClass(_class_template(tps), np.array(n, dtype=np.int64), chips, stack,
                                    np.array(pl, dtype=np.int64), np.array(pj, dtype=np.int64),
                                    np.array(pc, dtype=np.int64)))
        classes.append(street)
        if t + 1 == T:
            break
        for c, lc in enumerate(street):  # street t + 1's lines: by (parent class, parent line, close)
            members = groups[list(groups)[c]]
            for l, (_, chips, _, _, _, tp, ends) in enumerate(members):
                for j, end in enumerate(ends):
                    after = None if end is None else _advance(game, end)
                    added = tp.closes[j][1]
                    if after is not None and (game.is_terminal(after) or game.street(after) != t + 1):
                        after = None  # all-in: the rest is dealt with no betting
                    nxt.append((after, chips + added, c, l, j))
        entering = nxt
    return classes


def build_public_tree(game) -> PublicTree:
    """The public tree of a LimitPoker game with dense chance matrices (small games only)."""
    T = game.num_streets
    joint = []
    hands = set()
    for t in range(T):
        d = defaultdict(float)
        for board, h0, h1, q in game.deals(t):
            d[(board, h0, h1)] += q
            hands.update((h0, h1))
        joint.append(d)
    hands = sorted(hands)
    hidx = {h: i for i, h in enumerate(hands)}
    H = len(hands)

    streets = []
    for t in range(T):
        deals = sorted({b for b, _, _ in joint[t]})
        index = {b: i for i, b in enumerate(deals)}
        W = np.zeros((len(deals), H, H))
        for (b, h0, h1), q in joint[t].items():
            W[index[b], hidx[h0], hidx[h1]] += q
        if t == 0:
            parent = np.full(len(deals), -1, dtype=np.int64)
        else:
            prev = streets[t - 1]
            n = len(prev.deals[0])
            parent = np.array([prev.index[b[:n]] for b in deals], dtype=np.int64)
        streets.append(DenseStreet(deals, parent, W))
    last = streets[-1]
    S = np.zeros_like(last.W)
    for (b, h0, h1), q in joint[-1].items():
        S[last.index[b], hidx[h0], hidx[h1]] += q * float(np.mean(game.showdown(b, h0, h1)))
    last.S = S

    return PublicTree(game.name, hands, streets, betting(game))


# Coordinates of the compiled tree's infosets -----------------------------------------------


def tree_coordinates(pt: PublicTree, tree, player: int) -> np.ndarray:
    """(t, d, c, l, k, i) of every infoset of `player` in a compiled Tree of the same game:
    street, deal, line class, line within the class, template node, hand.

    Uses the tree's card-game metadata, and the key convention that the public key ends with
    ":" followed by each street's betting joined by "/"."""
    meta = tree.meta[player]
    hidx = {h: i for i, h in enumerate(pt.hands)}
    line_of = []  # [t] earlier streets' betting -> (class, line)
    for t in range(len(pt.classes)):
        line_of.append({h: (c, l) for c, row in enumerate(pt.histories(t)) for l, h in enumerate(row)})
    node_of = [[{h: k for k, h in enumerate(lc.template.hist)} for lc in row] for row in pt.classes]
    out = np.empty((tree.n_infosets[player], 6), dtype=np.int64)
    for n in range(tree.n_infosets[player]):
        t = meta["street"][n]
        bets = meta["public"][n].rsplit(":", 1)[1].split("/")
        c, line = line_of[t][tuple(bets[:t])]
        out[n] = (t, pt.streets[t].index[meta["board"][n]], c, line, node_of[t][c][bets[t]], hidx[meta["hand"][n]])
    return out


def to_tree_sigma(pt: PublicTree, tree, hand_sigma) -> list:
    """A per-hand range strategy (hand_sigma[t][c][k] shaped (D, L_c, H, actions)) as the tree
    solver's per-sequence arrays."""
    out = []
    for p in (0, 1):
        co = tree_coordinates(pt, tree, p)
        s = np.ones(tree.n_seqs[p])
        for n, (t, d, c, l, k, i) in enumerate(co):
            f = tree.first_seq[p][n]
            s[f:f + tree.num_actions[p][n]] = hand_sigma[t][c][k][d, l, i]
        out.append(s)
    return out


def from_tree_sigma(pt: PublicTree, tree, sigma) -> list:
    """The tree solver's per-sequence strategy as per-hand range arrays (uniform at hands the
    tree has no infoset for, which can't be held there)."""
    out = [[[np.full(pt.shape(t, c) + (len(a),), 1.0 / len(a)) for a in lc.template.actions]
            for c, lc in enumerate(row)] for t, row in enumerate(pt.classes[:pt.num_streets])]
    for p in (0, 1):
        co = tree_coordinates(pt, tree, p)
        for n, (t, d, c, l, k, i) in enumerate(co):
            f = tree.first_seq[p][n]
            out[t][c][k][d, l, i] = sigma[p][f:f + tree.num_actions[p][n]]
    return out
