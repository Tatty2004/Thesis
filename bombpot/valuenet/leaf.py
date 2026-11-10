"""NetLeaf: the value network as the turn-end value provider of range_cfr.backward
(PLAN_valuenet.md, section C):

    leaf.values(p, t, closing) -> agg

closing = (reach0, reach1), each (D, L, closes, H): both players' reach at every turn deal d,
entering line l and closing line c. agg is (D, L, closes, H): player p's counterfactual value
of each hand in the solver's units (chance weight x opponent reach x payoff, as a showdown
terminal). With line classes, `closing` is a list of such tuples (one per class), pot_by_line
and stack_by_line are lists of per-class arrays, and the result is a list.

For every (d, l, c):
  1. Mask both reaches with possible_by_deal[d] (hands that can be held at that deal) and
     normalize each to sum 1. A reach with no mass becomes uniform over the possible hands:
     the network still needs a valid input range, and the result is multiplied by the
     opponent's mass anyway (0 when the opponent can't be here; player p's own mass doesn't
     enter, so p's values stay defined at hands it never plays, as CFR needs).
  2. Histogram both normalized ranges over deal d's equity bins (equity_by_deal[d], (H, 2)).
  3. Run the network (all (d, l, c) in one batch, split into chunks of `batch_size`) and map
     player p's per-bin output (pots) back to hands.
  4. Rescale:  agg[d, l, c, i] = cfv[i] x pot[l, c] x sum_j reach_opp[d, l, c, j] x kappa[d],
     the sum over the opponent's possible hands, unnormalized. Hands impossible at d get 0.
Counterfactual values are linear in the opponent's reach, which is why step 4 is exact for
an exact network. kappa_by_deal[d] is the chance weight per hand pair at deal d (the caller's
W[d, i, j] for compatible hands); the leaf returns equilibrium values in every mode.

The caller (experiments/) supplies the equities, boards and possible hands from the game's
code; this layer never imports a game.
"""
from __future__ import annotations

import numpy as np
import torch

from bombpot.valuenet.features import bin_index, encode_boards, pot_features


class NetLeaf:

    def __init__(self, model, equity_by_deal, pot_by_line, stack_by_line, kappa_by_deal, device="cpu", *,
                 boards_by_deal=None, possible_by_deal=None, batch_size: int = 32768):
        """model: a ValueNet (anything with .features and forward(board, scalars, h0, h1) ->
        (v0, v1) per bin, in pots). equity_by_deal (D, H, 2). pot_by_line, stack_by_line
        (L, closes) chips (pot, and chips behind each) at each turn-closing line, or a list of
        such per line class. kappa_by_deal (D,). boards_by_deal (D, 2, k) card ids, required
        unless the model's board encoding is "none". possible_by_deal (D, H) bool, default all
        hands (then the caller must give impossible hands zero reach)."""
        self.model = model.to(device).eval()
        self.cfg, self.device, self.batch_size = model.features, torch.device(device), batch_size
        params = list(model.parameters())
        self.dtype = params[0].dtype if params else torch.float32  # the network's inputs
        equity = np.asarray(equity_by_deal, dtype=np.float64)
        D, H, _ = equity.shape
        self.D, self.H = D, H
        self.bins = torch.as_tensor(bin_index(equity, self.cfg), device=self.device)
        possible = np.ones((D, H), bool) if possible_by_deal is None else np.asarray(possible_by_deal, bool)
        self.possible = torch.as_tensor(possible, device=self.device)
        if self.cfg.board == "none":
            boards_by_deal = np.zeros((D, 2, self.cfg.board_cards), np.int64)
        elif boards_by_deal is None:
            raise ValueError("boards_by_deal is required for board encoding " + repr(self.cfg.board))
        self.board = torch.as_tensor(encode_boards(np.asarray(boards_by_deal), self.cfg), device=self.device)
        self.kappa = np.asarray(kappa_by_deal, dtype=np.float64)
        self.pot, self.stack = pot_by_line, stack_by_line

    def values(self, p: int, t: int, closing):
        if isinstance(closing, list):
            return [self._values(p, c, np.asarray(pot, float), np.asarray(st, float))
                    for c, pot, st in zip(closing, self.pot, self.stack)]
        return self._values(p, closing, np.asarray(self.pot, float), np.asarray(self.stack, float))

    @torch.no_grad()
    def _values(self, p: int, closing, pot: np.ndarray, stack: np.ndarray) -> np.ndarray:
        r0, r1 = (np.asarray(r, dtype=np.float64) for r in closing)
        D, L, C, H = r0.shape
        assert (D, H) == (self.D, self.H) and pot.shape == stack.shape == (L, C), "shapes disagree"
        B = self.cfg.num_bins
        scal = torch.as_tensor(pot_features(pot, stack).reshape(L * C, -1), device=self.device)
        out = np.empty((D * L * C, H))
        flat = [torch.as_tensor(r.reshape(-1, H), device=self.device) for r in (r0, r1)]
        for a in range(0, D * L * C, self.batch_size):
            n = torch.arange(a, min(a + self.batch_size, D * L * C), device=self.device)
            d, lc = n // (L * C), n % (L * C)
            poss = self.possible[d]
            bins = self.bins[d]
            hists, mass = [], []
            for r in flat:
                x = torch.where(poss, r[n], 0.0)
                m = x.sum(-1, keepdim=True)
                uni = poss.double() / poss.double().sum(-1, keepdim=True).clamp_min(1.0)
                x = torch.where(m > 0, x / torch.where(m > 0, m, 1.0), uni)
                h = torch.zeros(len(n), B, dtype=torch.float64, device=self.device).scatter_add_(1, bins, x)
                hists.append(h.to(self.dtype))
                mass.append(m[:, 0])
            board = self.board[d] if self.cfg.board == "ids" else self.board[d].to(self.dtype)
            v = self.model(board, scal[lc].to(self.dtype), hists[0], hists[1])[p]
            cfv = torch.gather(v.double(), 1, bins)
            scale = torch.as_tensor(pot.reshape(-1), device=self.device)[lc] * mass[1 - p] \
                * torch.as_tensor(self.kappa, device=self.device)[d]
            out[a:a + len(n)] = torch.where(poss, cfv * scale[:, None], 0.0).cpu().numpy()
        return out.reshape(D, L, C, H)
