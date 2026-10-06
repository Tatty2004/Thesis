"""Double-board Hold'em: a reference implementation for cross-checking the main one.

Deliberately naive and separate from bombpot/. Cards are dealt one at a time like a
real dealer: player 0's two cards, player 1's two, then each street's cards to board A
and then board B. Chips and turns are worked out from the action strings, and every
showdown is judged by ACPC's evaluator (universal_poker), not ours.

A state is (hole, boards, history):
    hole     hole cards dealt so far, in deal order (player 0's two, then player 1's two)
    boards   one tuple of cards per board, in deal order
    history  one action string per betting round begun
Cards are ints numbered like ACPC: rank * num_suits + suit.
"""
from acpc import acpc_winner

CHANCE = -1
TERMINAL = -2


class ReferenceHoldem:

    def __init__(self, num_ranks=4, num_suits=2, board_cards=(1, 1), num_boards=2, deck_mode="shared",
                 ante=1.0, bet_sizes=(2.0, 4.0), max_raises=2):
        if deck_mode not in ("shared", "identical"):
            raise ValueError(f"unknown deck_mode {deck_mode!r}")
        self.num_ranks, self.num_suits = num_ranks, num_suits
        self.board_cards = tuple(board_cards)
        self.streets = len(self.board_cards)
        self.num_boards, self.deck_mode = num_boards, deck_mode
        self.ante = float(ante)
        self.bet_sizes = tuple(float(b) for b in bet_sizes[:self.streets])
        if isinstance(max_raises, int):
            max_raises = [max_raises] * self.streets
        self.max_raises = tuple(int(m) for m in max_raises[:self.streets])
        self.deck = tuple(range(num_ranks * num_suits))
        self.name = (f"ReferenceHoldem({num_ranks}x{num_suits}, boards={num_boards}, cards={self.board_cards}, "
                     f"{deck_mode})")
        self._judged = {}

    def initial_state(self):
        return ((), ((),) * self.num_boards, ())

    def _cards_on_each_board(self, street):
        """How many cards each board holds once `street` (1, 2, ...) has been dealt."""
        return sum(self.board_cards[:street])

    def current_player(self, state):
        hole, boards, history = state
        if len(hole) < 4 or not history:
            return CHANCE                  # hole cards or street-1 board cards still to come
        actions = history[-1]
        if actions.endswith("f"):
            return TERMINAL
        if actions == "cc" or actions.endswith("rc"):    # both checked, or a bet was called
            return TERMINAL if len(history) == self.streets else CHANCE
        return len(actions) % 2            # player 0 acts first every street

    def is_terminal(self, state):
        return self.current_player(state) == TERMINAL

    def legal_actions(self, state):
        actions = state[2][-1]
        legal = ("f", "c") if actions.endswith("r") else ("c",)   # facing a bet exactly after an "r"
        if actions.count("r") < self.max_raises[len(state[2]) - 1]:
            legal += ("r",)
        return legal

    def _cards_for_next_deal(self, state):
        hole, boards, history = state
        if len(hole) < 4:
            gone = hole
        elif self.deck_mode == "shared":
            gone = hole + tuple(card for board in boards for card in board)
        else:
            gone = hole + boards[0]
        return [card for card in self.deck if card not in gone]

    def chance_outcomes(self, state):
        cards = self._cards_for_next_deal(state)
        return [(card, 1.0 / len(cards)) for card in cards]

    def next_state(self, state, move):
        hole, boards, history = state
        if self.current_player(state) != CHANCE:
            if move not in self.legal_actions(state):
                raise ValueError(f"illegal action {move!r}")
            return (hole, boards, history[:-1] + (history[-1] + move,))
        if move not in self._cards_for_next_deal(state):
            raise ValueError(f"card {move!r} cannot be dealt here")
        if len(hole) < 4:
            return (hole + (move,), boards, history)
        need = self._cards_on_each_board(len(history) + 1)
        if self.deck_mode == "identical":
            boards = tuple(board + (move,) for board in boards)
        else:
            i = next(i for i, board in enumerate(boards) if len(board) < need)
            boards = boards[:i] + (boards[i] + (move,),) + boards[i + 1:]
        if all(len(board) == need for board in boards):
            history += ("",)               # this street's cards are out: betting begins
        return (hole, boards, history)

    def _put_in(self, history):
        put_in = [self.ante, self.ante]
        for street, actions in enumerate(history):
            for i, action in enumerate(actions):
                me, other = i % 2, 1 - i % 2
                if action == "r":
                    put_in[me] = put_in[other] + self.bet_sizes[street]
                elif action == "c":
                    put_in[me] = put_in[other]
        return put_in

    def _winner(self, hole0, hole1, board):
        key = (tuple(sorted(hole0)), tuple(sorted(hole1)), tuple(sorted(board)))
        if key not in self._judged:
            self._judged[key] = acpc_winner(self.num_ranks, self.num_suits, hole0, hole1, board)
        return self._judged[key]

    def returns(self, state):
        hole, boards, history = state
        put_in = self._put_in(history)
        if history[-1].endswith("f"):
            folder = (len(history[-1]) - 1) % 2
            lost = put_in[folder]
            return (-lost, lost) if folder == 0 else (lost, -lost)
        share = (put_in[0] + put_in[1]) / self.num_boards
        received = [0.0, 0.0]
        for board in boards:
            w = self._winner(hole[:2], hole[2:], board)
            if w > 0:
                received[0] += share
            elif w < 0:
                received[1] += share
            else:
                received[0] += share / 2
                received[1] += share / 2
        return (received[0] - put_in[0], received[1] - put_in[1])

    def _streets(self, board):
        """A board's cards grouped by street, each street's cards sorted (their order within
        a street tells nobody anything)."""
        out, start = [], 0
        for k in self.board_cards:
            if start >= len(board):
                break
            out.append(tuple(sorted(board[start:start + k])))
            start += k
        return out

    def infoset_key(self, state, player):
        hole, boards, history = state
        own = tuple(sorted(hole[2 * player:2 * player + 2]))
        shown = "|".join("/".join(",".join(map(str, s)) for s in self._streets(b)) for b in boards)
        return f"P{player} own={own} boards={shown} bets={'/'.join(history)}"

    def describe(self, state):
        hole, boards, history = state
        return ((tuple(sorted(hole[:2])), tuple(sorted(hole[2:]))),
                tuple(tuple(c for s in self._streets(b) for c in s) for b in boards),
                history)
