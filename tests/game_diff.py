"""Line up two implementations of the same card game and list every difference.

Each game is walked in full through the Game protocol, and `describe(state)` turns
every decision node and terminal into a canonical (hands, boards, history) record
in ranks. That lets a card-level implementation (real suited cards, every card
equally likely) be compared with our rank-level one (ranks weighted by how many
cards are left): their chance nodes never line up, but their records must.

For each record the two games must agree on the chance probability of reaching a
terminal (summed over suits), its payoff, the player to act, the legal actions, and
which records share an information set. Keys are compared as a partition, so the
two implementations can spell them differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bombpot.core.game import CHANCE

CATEGORIES = (
    "internal",       # a game disagrees with itself: chance sums, zero-sum, suits changing the outcome
    "histories",      # a terminal history exists in only one game
    "probabilities",  # chance probability of a shared terminal history
    "payoffs",        # payoff of a shared terminal history
    "decisions",      # a decision node exists in only one game
    "players",        # a different player acts
    "actions",        # different legal actions
    "infosets",       # records grouped into information sets differently
)


def poker_record(state) -> tuple:
    """describe() for our LimitPoker games (Kuhn, Leduc, Board-Leduc)."""
    return (state.hands, state.boards, state.hist)


@dataclass
class Walk:
    terminals: dict  # record -> [chance probability, payoff to player 0]
    decisions: dict  # record -> (player, sorted legal actions, infoset key)
    problems: list   # (kind, record, details) inconsistencies inside this game
    nodes: int


def walk(game, describe=poker_record, tol: float = 1e-12) -> Walk:
    """Visit every history of `game` and aggregate it by canonical record."""
    terminals, decisions, problems = {}, {}, []
    nodes = 0
    stack = [(game.initial_state(), 1.0)]
    while stack:
        s, prob = stack.pop()
        nodes += 1
        if game.is_terminal(s):
            rec = describe(s)
            u0, u1 = game.returns(s)
            if u0 + u1 != 0:
                problems.append(("not zero-sum", rec, (u0, u1)))
            entry = terminals.get(rec)
            if entry is None:
                terminals[rec] = [prob, u0]
            else:
                entry[0] += prob
                if entry[1] != u0:
                    problems.append(("payoff differs between suits", rec, (entry[1], u0)))
            continue
        p = game.current_player(s)
        if p == CHANCE:
            outs = list(game.chance_outcomes(s))
            if abs(sum(q for _, q in outs) - 1.0) > tol or any(q <= 0 for _, q in outs):
                problems.append(("chance probabilities", describe_safe(describe, s), outs))
            for o, q in outs:
                stack.append((game.next_state(s, o), prob * q))
            continue
        rec = describe(s)
        acts = tuple(game.legal_actions(s))
        info = (p, tuple(sorted(acts)), game.infoset_key(s, p))
        old = decisions.setdefault(rec, info)
        if old != info:
            problems.append(("decision differs between suits", rec, (old, info)))
        if not acts:
            problems.append(("no legal actions", rec, None))
        for a in acts:
            stack.append((game.next_state(s, a), prob))
    return Walk(terminals, decisions, problems, nodes)


def describe_safe(describe, s):
    try:
        return describe(s)
    except Exception:  # chance nodes may be mid-deal
        return repr(s)


@dataclass
class Diff:
    counts: dict = field(default_factory=lambda: {c: 0 for c in CATEGORIES})
    examples: dict = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    sizes: dict = field(default_factory=dict)

    def add(self, category: str, example, keep: int = 5) -> None:
        self.counts[category] += 1
        if len(self.examples[category]) < keep:
            self.examples[category].append(example)

    @property
    def ok(self) -> bool:
        return not any(self.counts.values())

    @property
    def caught(self) -> set:
        return {c for c, n in self.counts.items() if n}

    def __str__(self) -> str:
        lines = [f"sizes {self.sizes}"]
        for c in CATEGORIES:
            if self.counts[c]:
                lines.append(f"{c}: {self.counts[c]} differences, e.g.")
                lines.extend(f"    {e}" for e in self.examples[c])
        return "\n".join(lines) if not self.ok else f"no differences; {lines[0]}"


def diff(ours: Walk, ref: Walk, tol: float = 1e-12) -> Diff:
    """Every difference between two walks (`ours` first in each example)."""
    d = Diff()
    d.sizes = {"terminal records": (len(ours.terminals), len(ref.terminals)),
               "decision records": (len(ours.decisions), len(ref.decisions)),
               "nodes walked": (ours.nodes, ref.nodes)}
    for side, w in (("ours", ours), ("reference", ref)):
        for problem in w.problems:
            d.add("internal", (side,) + problem)

    for rec in sorted(ours.terminals.keys() - ref.terminals.keys()):
        d.add("histories", ("only ours", rec))
    for rec in sorted(ref.terminals.keys() - ours.terminals.keys()):
        d.add("histories", ("only reference", rec))
    for rec in ours.terminals.keys() & ref.terminals.keys():
        (pa, ua), (pb, ub) = ours.terminals[rec], ref.terminals[rec]
        if abs(pa - pb) > tol:
            d.add("probabilities", (rec, pa, pb))
        if ua != ub:
            d.add("payoffs", (rec, ua, ub))

    for rec in sorted(ours.decisions.keys() - ref.decisions.keys()):
        d.add("decisions", ("only ours", rec))
    for rec in sorted(ref.decisions.keys() - ours.decisions.keys()):
        d.add("decisions", ("only reference", rec))
    forward, backward = {}, {}
    for rec in ours.decisions.keys() & ref.decisions.keys():
        (pa, aa, ka), (pb, ab, kb) = ours.decisions[rec], ref.decisions[rec]
        if pa != pb:
            d.add("players", (rec, pa, pb))
        if aa != ab:
            d.add("actions", (rec, aa, ab))
        # Same partition: our infoset -> their infoset must be one-to-one.
        if forward.setdefault((pa, ka), (pb, kb)) != (pb, kb):
            d.add("infosets", ("we merge what the reference separates", rec, ka, kb))
        if backward.setdefault((pb, kb), (pa, ka)) != (pa, ka):
            d.add("infosets", ("the reference merges what we separate", rec, ka, kb))
    d.sizes["infosets"] = (len(forward), len(backward))
    return d
