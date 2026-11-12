"""Last-street subgames solved one deal at a time in a single compiled kernel.

In a last-street subgame (solvers/subgame.py) the root ranges are fixed and every deal has
its own strategy, so the deals don't interact: an iteration of CFR on the subgame is one
independent iteration per deal. `SubgameCFR` still runs it as the range solver runs a
whole tree, one numpy-sized operation at a time over every deal (about 70 kernel calls per
iteration on arrays of D x H, most of them memory-bound). Here one kernel call runs n whole
iterations (forward, backward and update for both players, alternating) for each deal, in
parallel over (sample, deal), with that deal's arrays in cache.

The arithmetic is the range solver's, step by step and in the same order (the fold and
showdown sweeps of core/removal.py, with the bookkeeping changes described at the per-deal
pieces below, which leave every sum's terms and order as they are; `_expectation`,
`_combine` and `_update` of range_cfr.py), so in float64 the iterates (regrets, average sums, strategies, free-running),
root values, exploitability and the iteration a tolerance stops at equal SubgameCFR's bit
for bit (tests/test_fused.py). Root values are summed over deals by the same segment sum.
The arrays can also be float32 (state and sweeps): then one iteration agrees to rounding
(a few 1e-6 of chips x chance weight), but a regret that is an exact 0 in float64 can be
rounding in float32, so trajectories part ways at such ties and a solve to a tolerance is
another solve of the same quality (the same tolerance on its own exploitability, measured
in float32), not the same numbers.

Several subgames on the same street (the same deals: e.g. several samples at one turn
deal) are solved in one batch, each with its own ranges, template and chips. Each stops
on its own tolerance.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from bombpot.core.jit import kernel
from bombpot.core.solvers.range_cfr import VARIANTS, _segment_sum
from bombpot.core.solvers.subgame import Subgame

NODE, FOLD, CLOSE = 0, 1, 2


# Per-deal pieces (inlined into the kernels) --------------------------------------------------
# The sums of core/removal.py with two of each hand's subsets kept out of the scatter arrays:
# the empty set (column 0 of HandSet.subs: every hand has it, so its total is a running
# scalar) and the whole hand (the last column: no other hand has it, so its total is the
# hand's own reach). Only the subsets in between (`mid`, numbered compactly) are gathered.
# The showdown sweep visits only the deal's valid hands (`order`, `group_end` compacted: an
# invalid hand adds nothing and its value is 0). Every sum is still accumulated in the same
# order from the same terms, so the results are the same floats.


@njit(cache=True, inline="always")
def _fold1(ro, valid, kappa, mid, signs, zero, total, out):
    """core/removal._fold_kernel for one deal and line."""
    H = ro.shape[0]
    M = mid.shape[1]
    total[:] = zero
    all_ = zero
    for i in range(H):
        r = ro[i]
        if valid[i] and r != 0.0:
            all_ += r
            for k in range(M):
                total[mid[i, k]] += r
    for i in range(H):
        v = zero
        if valid[i]:
            v += signs[0] * all_
            for k in range(M):
                v += signs[k + 1] * total[mid[i, k]]
            r = ro[i]
            v += signs[M + 1] * (r if r != 0.0 else zero)
        out[i] = v * kappa


@njit(cache=True, inline="always")
def _beats1(ro, valid, n_valid, kappa, order, group_end, mid, signs, consts, weaker, tied, out):
    """core/removal._beats_kernel for one deal and line (order, group_end: (B, H), the deal's
    n_valid valid hands in strength order and each one's tie group end)."""
    H = ro.shape[0]
    B = order.shape[0]
    M = mid.shape[1]
    zero, half, boards = consts[0], consts[1], consts[3]
    for i in range(H):
        out[i] = zero
    for b in range(B):
        weaker[:] = zero
        weaker_all = zero
        q = 0
        while q < n_valid:
            e = group_end[b, q]
            tied_all = zero
            for u in range(q, e):
                i = order[b, u]
                r = ro[i]
                if r != 0.0:
                    tied_all += r
                    for k in range(M):
                        tied[mid[i, k]] += r
            for u in range(q, e):
                i = order[b, u]
                w = zero
                t = zero
                w += signs[0] * weaker_all
                t += signs[0] * tied_all
                for k in range(M):
                    s = mid[i, k]
                    w += signs[k + 1] * weaker[s]
                    t += signs[k + 1] * tied[s]
                r = ro[i]
                t += signs[M + 1] * (r if r != 0.0 else zero)
                out[i] += w + half * t
            for u in range(q, e):
                i = order[b, u]
                for k in range(M):
                    s = mid[i, k]
                    weaker[s] += tied[s]
                    tied[s] = zero
            weaker_all += tied_all
            q = e
    for i in range(H):
        out[i] = out[i] / boards * kappa if valid[i] else zero


@njit(cache=True, inline="always")
def _from_sums(p, kd, c, two, fbuf, bbuf, out):
    """Player p's value of a fold (kd == FOLD) or showdown child with coefficient c from the
    fold and showdown sums of the opponent's reach (range_cfr.class_values: _scaled / _showdown)."""
    H = out.shape[0]
    if kd == FOLD:
        for i in range(H):
            out[i] = c * fbuf[i]
    elif p == 0:
        for i in range(H):
            out[i] = c * (two * bbuf[i] - fbuf[i])
    else:
        for i in range(H):
            out[i] = c * (two * (fbuf[i] - bbuf[i]) - fbuf[i])


@njit(cache=True, inline="always")
def _terminals(p, k, K, player, nact, kind, coef, sig, b, reach, valid, n_valid, kappa, order, group_end, mid, signs,
               consts, weaker, tied, total, rbuf, fbuf, bbuf, cv):
    """cv[a] for every fold and showdown child a of node k: player p's value against the
    opponent's reach after a. At p's own node the opponent's reach is the node's for every
    child, so its sums are taken once."""
    H = reach.shape[2]
    q = player[k]
    o = 1 - p
    A = nact[k]
    if q == p:
        have_f = False
        have_b = False
        for a in range(A):
            kd = kind[k, a]
            if kd == NODE:
                continue
            if not have_f:
                _fold1(reach[o, k], valid, kappa, mid, signs, consts[0], total, fbuf)
                have_f = True
            if kd == CLOSE and not have_b:
                _beats1(reach[o, k], valid, n_valid, kappa, order, group_end, mid, signs, consts, weaker, tied, bbuf)
                have_b = True
            _from_sums(p, kd, coef[p, k, a], consts[2], fbuf, bbuf, cv[a])
    else:
        for a in range(A):
            kd = kind[k, a]
            if kd == NODE:
                continue
            for i in range(H):
                rbuf[i] = reach[o, k, i] * sig[b + i * A + a]
            _fold1(rbuf, valid, kappa, mid, signs, consts[0], total, fbuf)
            if kd == CLOSE:
                _beats1(rbuf, valid, n_valid, kappa, order, group_end, mid, signs, consts, weaker, tied, bbuf)
            _from_sums(p, kd, coef[p, k, a], consts[2], fbuf, bbuf, cv[a])


@njit(cache=True, inline="always")
def _forward1(K, player, nact, kind, target, sig, off, base, reach):
    """reach[q, k, :] at every node from reach[:, 0, :] (range_cfr.class_reach)."""
    H = reach.shape[2]
    for k in range(K):
        q = player[k]
        A = nact[k]
        o = 1 - q
        b = base + off[k]
        for a in range(A):
            if kind[k, a] == NODE:
                j = target[k, a]
                for i in range(H):
                    reach[q, j, i] = reach[q, k, i] * sig[b + i * A + a]
                    reach[o, j, i] = reach[o, k, i]


@njit(cache=True, inline="always")
def V_of(kd, j, V, tv, a, i):
    return V[j, i] if kd == NODE else tv[a, i]


# Kernels ----------------------------------------------------------------------------------


@kernel
def _iterate_kernel(valid, n_valid, kappa, order, group_end, mid, signs, n_mid, consts, root, player, nact, kind, target,
                    coef, n_nodes, off, size, base, unif, regret, cum, sigma, weight, pos, neg, variant, n_iter,
                    active):
    """n_iter alternating CFR iterations of every active sample s at every deal d.

    root (S, 2, H); template arrays per sample (S, Kmax[, Amax]); coef (S, 2, Kmax, Amax);
    regret, cum, sigma flat, sample s's deal d at base[s] + d size[s], node k at off[s, k]
    within it, laid out (H, A_k); weight, pos, neg (S, n_iter) per iteration; consts = (0, 1/2,
    2, boards, flush) and unif (S, Kmax) = 1 / actions in the state's dtype."""
    S = root.shape[0]
    D, H = valid.shape
    Kmax = player.shape[1]
    Amax = kind.shape[2]
    for m in prange(S * D):
        s = m // D
        d = m - s * D
        if active[s]:
            K = n_nodes[s]
            b0 = base[s] + d * size[s]
            reach = np.empty((2, Kmax, H), root.dtype)
            V = np.empty((Kmax, H), root.dtype)
            cv = np.empty((Amax, H), root.dtype)
            inst = np.empty(size[s], root.dtype)
            rbuf = np.empty(H, root.dtype)
            fbuf = np.empty(H, root.dtype)
            bbuf = np.empty(H, root.dtype)
            weaker = np.zeros(n_mid, root.dtype)
            tied = np.zeros(n_mid, root.dtype)
            total = np.zeros(n_mid, root.dtype)
            vd, od, gd = valid[d], order[d], group_end[d]
            nv = n_valid[d]
            kap = kappa[d]
            zero, flush = consts[0], consts[4]
            for it in range(n_iter):
                for p in range(2):
                    for i in range(H):
                        reach[0, 0, i] = root[s, 0, i]
                        reach[1, 0, i] = root[s, 1, i]
                    _forward1(K, player[s], nact[s], kind[s], target[s], sigma, off[s], b0, reach)
                    for k in range(K - 1, -1, -1):
                        q = player[s, k]
                        A = nact[s, k]
                        bk = b0 + off[s, k]
                        _terminals(p, k, K, player[s], nact[s], kind[s], coef[s], sigma, bk, reach, vd, nv, kap, od,
                                   gd, mid, signs, consts, weaker, tied, total, rbuf, fbuf, bbuf, cv)
                        for a in range(A):
                            if kind[s, k, a] == NODE:
                                j = target[s, k, a]
                                for i in range(H):
                                    cv[a, i] = V[j, i]
                        if q != p:
                            for i in range(H):
                                v = cv[0, i]
                                for a in range(1, A):
                                    v = v + cv[a, i]
                                V[k, i] = v
                        else:
                            ik = off[s, k]
                            for i in range(H):
                                v = zero
                                for a in range(A):
                                    v += sigma[bk + i * A + a] * cv[a, i]
                                V[k, i] = v
                                for a in range(A):
                                    inst[ik + i * A + a] = cv[a, i] - v
                    w0, ps, ng = weight[s, it], pos[s, it], neg[s, it]
                    for k in range(K):
                        if player[s, k] != p:
                            continue
                        A = nact[s, k]
                        bk = b0 + off[s, k]
                        ik = off[s, k]
                        uniform = unif[s, k]
                        for i in range(H):
                            w = w0 * reach[p, k, i]
                            tot = zero
                            for a in range(A):
                                x = bk + i * A + a
                                cum[x] += w * sigma[x]
                                r = regret[x] + inst[ik + i * A + a]
                                if variant == 1 and r < 0.0:
                                    r = zero
                                elif variant == 2:
                                    r = r * (ps if r > 0.0 else ng)
                                    if r < 0.0 and r > flush:  # float32 only (see FusedSubgames)
                                        r = zero
                                regret[x] = r
                                if r > 0.0:
                                    tot += r
                            for a in range(A):
                                x = bk + i * A + a
                                sigma[x] = (max(regret[x], zero) / tot) if tot > 0.0 else uniform


@kernel
def _evaluate_kernel(valid, n_valid, kappa, order, group_end, mid, signs, n_mid, consts, root, player, nact, kind, target,
                     coef, n_nodes, off, size, base, unif, cum, active, values, br):
    """Root values of the average strategy (mode "cfr") and of a best response (mode "br")
    for both players at every deal: values, br (S, 2, D, H)."""
    S = root.shape[0]
    D, H = valid.shape
    Kmax = player.shape[1]
    Amax = kind.shape[2]
    for m in prange(S * D):
        s = m // D
        d = m - s * D
        if active[s]:
            K = n_nodes[s]
            b0 = base[s] + d * size[s]
            avg = np.empty(size[s], root.dtype)
            reach = np.empty((2, Kmax, H), root.dtype)
            Vc = np.empty((Kmax, H), root.dtype)
            Vb = np.empty((Kmax, H), root.dtype)
            tv = np.empty((Amax, H), root.dtype)
            rbuf = np.empty(H, root.dtype)
            fbuf = np.empty(H, root.dtype)
            bbuf = np.empty(H, root.dtype)
            weaker = np.zeros(n_mid, root.dtype)
            tied = np.zeros(n_mid, root.dtype)
            total = np.zeros(n_mid, root.dtype)
            vd, od, gd = valid[d], order[d], group_end[d]
            nv = n_valid[d]
            kap = kappa[d]
            zero = consts[0]
            for k in range(K):  # RangeCFR.average_strategy
                A = nact[s, k]
                uniform = unif[s, k]
                for i in range(H):
                    x0 = b0 + off[s, k] + i * A
                    t = zero
                    for a in range(A):
                        t += cum[x0 + a]
                    for a in range(A):
                        avg[off[s, k] + i * A + a] = cum[x0 + a] / t if t > 0 else uniform
            for i in range(H):
                reach[0, 0, i] = root[s, 0, i]
                reach[1, 0, i] = root[s, 1, i]
            _forward1(K, player[s], nact[s], kind[s], target[s], avg, off[s], 0, reach)
            for p in range(2):
                for k in range(K - 1, -1, -1):
                    q = player[s, k]
                    A = nact[s, k]
                    bk = off[s, k]
                    _terminals(p, k, K, player[s], nact[s], kind[s], coef[s], avg, bk, reach, vd, nv, kap, od, gd,
                               mid, signs, consts, weaker, tied, total, rbuf, fbuf, bbuf, tv)
                    for i in range(H):
                        # mode "cfr": p plays the average (_expectation), o's actions add (_combine)
                        if q == p:
                            v = zero
                            for a in range(A):
                                x = V_of(kind[s, k, a], target[s, k, a], Vc, tv, a, i)
                                v += avg[bk + i * A + a] * x
                        else:
                            v = V_of(kind[s, k, 0], target[s, k, 0], Vc, tv, 0, i)
                            for a in range(1, A):
                                v = v + V_of(kind[s, k, a], target[s, k, a], Vc, tv, a, i)
                        # mode "br": p takes the best action (ties keep the earlier one)
                        u = V_of(kind[s, k, 0], target[s, k, 0], Vb, tv, 0, i)
                        for a in range(1, A):
                            x = V_of(kind[s, k, a], target[s, k, a], Vb, tv, a, i)
                            if q != p:
                                u = u + x
                            elif x > u:
                                u = x
                        Vc[k, i] = v
                        Vb[k, i] = u
                for i in range(H):
                    values[s, p, d, i] = Vc[0, i]
                    br[s, p, d, i] = Vb[0, i]


# Driver -----------------------------------------------------------------------------------


def compile_template(tp, contrib: float):
    """A one-line template as arrays: player (K,), nact (K,), kind and target (K, Amax),
    coef (2, K, Amax): player p's chips factor at each fold or showdown child, as
    range_cfr.class_values computes it (sign x chips)."""
    K = tp.num_nodes
    Amax = max(len(a) for a in tp.actions)
    player = np.array(tp.players, dtype=np.int64)
    nact = np.array([len(a) for a in tp.actions], dtype=np.int64)
    kind = np.zeros((K, Amax), dtype=np.int64)
    target = np.zeros((K, Amax), dtype=np.int64)
    coef = np.zeros((2, K, Amax))
    contrib = np.atleast_1d(np.asarray(contrib, dtype=float))
    for k, row in enumerate(tp.children):
        for a, ch in enumerate(row):
            if ch[0] == "node":
                kind[k, a], target[k, a] = NODE, ch[1]
                continue
            if ch[0] == "fold":
                _, folder, extra = ch
                lost = contrib + np.atleast_1d(np.asarray(extra, dtype=float))
                u0 = -lost if folder == 0 else lost
                kind[k, a] = FOLD
            else:
                u0 = contrib + np.atleast_1d(np.asarray(tp.closes[ch[1]][1], dtype=float))
                kind[k, a], target[k, a] = CLOSE, ch[1]
            for p, sign in ((0, 1.0), (1, -1.0)):
                coef[p, k, a] = (sign * u0)[0]
    return player, nact, kind, target, coef


def _discounts(variant: str, t: np.ndarray, alpha: float, beta: float, gamma: float):
    """weight, pos, neg of iterations t (RangeCFR.iteration), as Python floats do it."""
    w, p, n = [], [], []
    for t_ in (int(x) for x in t):
        w.append({"cfr": 1.0, "cfr+": float(t_), "dcfr": float(t_) ** gamma}[variant])
        p.append(t_ ** alpha / (t_ ** alpha + 1.0))
        n.append(t_ ** beta / (t_ ** beta + 1.0))
    return np.array(w), np.array(p), np.array(n)


def _compact_ids(subs: np.ndarray):
    """Subset ids renumbered 0..n-1 over the ones that occur: (ids int32, n)."""
    if subs.size == 0:
        return np.zeros(subs.shape, np.int32), 1
    u, inv = np.unique(subs, return_inverse=True)
    return np.ascontiguousarray(inv.reshape(subs.shape).astype(np.int32)), len(u)


def _valid_sweep(valid: np.ndarray, order: np.ndarray, group_end: np.ndarray):
    """A street's strength sweep restricted to each deal's valid hands: order (D, B, H) int32
    with the deal's valid hands first, in strength order; group_end (D, B, H) int32 the end of
    each one's tie group among them; n_valid (D,). The tie groups are the street's, minus the
    invalid hands (a group of only invalid hands is gone)."""
    D, B, H = order.shape
    pos = np.arange(H)
    gid = np.zeros((D, B, H), np.int64)  # tie group of each position: a new one after each group's end
    gid[..., 1:] = np.cumsum(group_end[..., :-1] == pos[1:], axis=2)
    keep = np.take_along_axis(np.broadcast_to(valid[:, None, :], (D, B, H)), order, axis=2)
    perm = np.argsort(~keep, axis=2, kind="stable")  # kept positions first, order preserved
    out = np.take_along_axis(order, perm, axis=2)
    gid = np.take_along_axis(gid, perm, axis=2)
    n_valid = valid.sum(axis=1).astype(np.int64)
    last = np.ones((D, B, H), bool)
    last[..., :-1] = gid[..., 1:] != gid[..., :-1]
    last |= (pos[None, None, :] == (n_valid[:, None, None] - 1))
    end = np.minimum.accumulate(np.where(last, pos + 1, H)[..., ::-1], axis=2)[..., ::-1]
    return np.ascontiguousarray(out.astype(np.int32)), np.ascontiguousarray(end.astype(np.int32)), n_valid


class FusedSubgames:
    """A batch of one-line subgames on one street (the same deals), solved per deal.

    street     a CardStreet (its kappa are the subgames' chance weights)
    templates  [s] a one-line Template per subgame (line_template(class template, l))
    contrib    [s] chips each player has in at the root
    roots      [s] (reach0, reach1), each (H,)
    dtype      np.float64 (bit-identical to SubgameCFR) or np.float32
    """

    def __init__(self, street, templates, contrib, roots, variant: str = "dcfr", alpha: float = 1.5,
                 beta: float = 0.0, gamma: float = 2.0, dtype=np.float64):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.st, self.variant, self.dtype = street, variant, np.dtype(dtype)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        S = len(templates)
        if any(tp.num_nodes == 0 for tp in templates):
            raise ValueError("a template without betting has nothing to solve")
        D, H = street.valid.shape
        self.S, self.D, self.H = S, D, H
        comp = [compile_template(tp, c) for tp, c in zip(templates, contrib)]
        Kmax = max(len(c[0]) for c in comp)
        Amax = max(c[2].shape[1] for c in comp)
        self.player = np.zeros((S, Kmax), np.int64)
        self.nact = np.ones((S, Kmax), np.int64)
        self.kind = np.zeros((S, Kmax, Amax), np.int64)
        self.target = np.zeros((S, Kmax, Amax), np.int64)
        self.coef = np.zeros((S, 2, Kmax, Amax), self.dtype)
        self.n_nodes = np.array([len(c[0]) for c in comp], np.int64)
        self.off = np.zeros((S, Kmax), np.int64)
        for s, (pl, na, kd, tg, cf) in enumerate(comp):
            K, A = kd.shape
            self.player[s, :K], self.nact[s, :K] = pl, na
            self.kind[s, :K, :A], self.target[s, :K, :A], self.coef[s, :, :K, :A] = kd, tg, cf
            self.off[s, :K] = np.concatenate([[0], np.cumsum(na * H)[:-1]])
        self.size = (self.nact * H * (np.arange(Kmax)[None] < self.n_nodes[:, None])).sum(1).astype(np.int64)
        self.base = np.concatenate([[0], np.cumsum(self.size * D)[:-1]]).astype(np.int64)
        n = int((self.size * D).sum())
        self.regret = np.zeros(n, self.dtype)
        self.cum = np.zeros(n, self.dtype)
        self.sigma = np.empty(n, self.dtype)
        self.unif = (1.0 / self.nact).astype(self.dtype)
        for s in range(S):  # uniform, as RangeCFR._strategy of zero regrets
            block = self.sigma[self.base[s]:self.base[s] + D * self.size[s]].reshape(D, self.size[s])
            for k in range(self.n_nodes[s]):
                block[:, self.off[s, k]:self.off[s, k] + H * self.nact[s, k]] = self.unif[s, k]
        self.roots = [tuple(np.asarray(r, dtype=float) for r in rt) for rt in roots]
        self.root = np.ascontiguousarray(np.stack([np.stack(r) for r in self.roots]), dtype=self.dtype)  # (S, 2, H)
        self.t = np.zeros(S, np.int64)
        self.contrib = [float(np.asarray(c).reshape(-1)[0]) for c in contrib]
        h = street.hands
        if (h.subs[:, 0] != h.subs[0, 0]).any() or len(np.unique(h.subs[:, -1])) != H:
            raise ValueError("hands must be distinct card sets, their subsets listed from empty to whole")
        mid, n_mid = _compact_ids(h.subs[:, 1:-1])
        order, group_end, n_valid = _valid_sweep(street.valid, street.order, street.group_end)
        # DCFR halves a negative regret every iteration (beta = 0), which in float32 reaches the
        # denormals (slow arithmetic) after about a hundred iterations: there, negative regrets
        # above -1e-30 are set to 0. In float64 the threshold is -0.0, which nothing is above.
        flush = -1e-30 if self.dtype == np.float32 else -0.0
        consts = np.array([0.0, 0.5, 2.0, order.shape[1], flush], self.dtype)
        self._street = (np.ascontiguousarray(street.valid), n_valid, np.ascontiguousarray(street.kappa, dtype=self.dtype),
                        order, group_end, mid, h.signs.astype(self.dtype), n_mid, consts)
        self.seconds = 0.0

    def _template(self):
        return (self.player, self.nact, self.kind, self.target, self.coef, self.n_nodes, self.off, self.size,
                self.base, self.unif)

    def work(self, active) -> int:
        st = self.st
        return int(active.sum()) * self.D * self.H * st.hands.subs.shape[1] * 40

    def iterate(self, n: int, active=None) -> None:
        """n iterations of every active subgame."""
        active = np.ones(self.S, bool) if active is None else np.asarray(active, bool)
        start = time.perf_counter()
        W = np.zeros((self.S, n)), np.zeros((self.S, n)), np.zeros((self.S, n))
        for s in np.flatnonzero(active):
            for x, y in zip(W, _discounts(self.variant, self.t[s] + 1 + np.arange(n), self.alpha, self.beta,
                                          self.gamma)):
                x[s] = y
        _iterate_kernel(self.work(active) * n, *self._street, self.root, *self._template(), self.regret, self.cum,
                        self.sigma, *(x.astype(self.dtype) for x in W), VARIANTS[self.variant], n, active)
        self.t[active] += n
        self.seconds += time.perf_counter() - start

    def evaluate(self, active=None) -> dict:
        """Per subgame (lists over s, None where inactive): values[p], br_values[p] (H,) summed
        over deals, exploitability (chips per unit of root reach mass), exploitability_raw,
        mass; as SubgameCFR.evaluate."""
        active = np.ones(self.S, bool) if active is None else np.asarray(active, bool)
        start = time.perf_counter()
        vals = np.zeros((self.S, 2, self.D, self.H), self.dtype)
        brs = np.zeros_like(vals)
        _evaluate_kernel(self.work(active), *self._street, self.root, *self._template(), self.cum, active, vals, brs)
        self.seconds += time.perf_counter() - start
        zero = np.zeros(self.D, dtype=np.int64)
        out = {"values": [], "br_values": [], "exploitability": [], "exploitability_raw": [], "mass": []}
        for s in range(self.S):
            if not active[s]:
                for v in out.values():
                    v.append(None)
                continue
            v = [_segment_sum(vals[s, p][:, None].astype(float), zero, 1)[0, 0] for p in (0, 1)]
            b = [_segment_sum(brs[s, p][:, None].astype(float), zero, 1)[0, 0] for p in (0, 1)]
            r0, r1 = self.roots[s]
            raw = ((b[0] * r0).sum(-1) + (b[1] * r1).sum(-1)) / 2
            mass = self.mass(s)
            out["values"].append(v)
            out["br_values"].append(b)
            out["exploitability_raw"].append(float(raw))
            out["mass"].append(mass)
            out["exploitability"].append(float(raw / mass) if mass > 0 else 0.0)
        return out

    def mass(self, s: int) -> float:
        if not hasattr(self, "_mass"):
            self._mass = {}
        if s not in self._mass:  # SubgameCFR.mass
            r0, r1 = self.roots[s]
            f = self.st.fold(np.ascontiguousarray(np.broadcast_to(r1, (self.D, 1, self.H))), 0)
            zero = np.zeros(self.D, dtype=np.int64)
            self._mass[s] = float((_segment_sum(f, zero, 1) * r0).sum(-1)[0, 0])
        return self._mass[s]

    def node_array(self, x: np.ndarray, s: int, k: int) -> np.ndarray:
        """Subgame s's node k of a state array (regret, cum or sigma) as (D, H, A), a copy."""
        A = self.nact[s, k]
        idx = self.base[s] + np.arange(self.D)[:, None] * self.size[s] + self.off[s, k] + np.arange(self.H * A)
        return x[idx].reshape(self.D, self.H, A)

    def average_strategy(self, s: int) -> list:
        """[k] (D, H, A) the average strategy of subgame s (RangeCFR.average_strategy)."""
        out = []
        for k in range(self.n_nodes[s]):
            c = self.node_array(self.cum, s, k).astype(float)
            total = c.sum(-1, keepdims=True)
            out.append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
        return out

    def solve(self, iters: int, tol=None, check_every: int = 10) -> dict:
        """Iterate each subgame until its exploitability is <= its tolerance (tol: one number
        or one per subgame; checked every `check_every` iterations) or `iters` iterations, as
        SubgameCFR.solve with criterion "max". A subgame that has stopped is left alone while
        the others go on. Returns evaluate() of every subgame's final average strategy, with
        `iterations` added."""
        tol = None if tol is None else np.broadcast_to(np.asarray(tol, dtype=float), (self.S,))
        active = self.t < iters
        ev = self.evaluate(~active)  # nothing to iterate (iters reached already)
        final = {k: list(v) for k, v in ev.items()}
        while active.any():
            # the subgames start together, so the active ones are on the same iteration
            at = int(self.t[active][0])
            n = iters - at if tol is None else min(check_every - at % check_every, iters - at)
            self.iterate(n, active)
            at += n
            ev = self.evaluate(active)
            for s in np.flatnonzero(active):
                if at >= iters or ev["exploitability"][s] <= tol[s]:
                    for k in final:
                        final[k][s] = ev[k][s]
                    active[s] = False
        final["iterations"] = [int(x) for x in self.t]
        return final


def solve_subgame_fused(street, template, contrib, reach0, reach1, variant: str = "dcfr", iters: int = 1000,
                        tol: float | None = None, check_every: int = 10, dtype=np.float64,
                        keep_strategy: bool = True) -> Subgame:
    """subgame.solve_subgame through FusedSubgames: the same Subgame (values, br_values (H,),
    strategy[k] (D, H, A), exploitability), bit for bit in float64. keep_strategy=False
    leaves `strategy` empty (building it costs seconds on a big street)."""
    f = FusedSubgames(street, [template], [contrib], [(reach0, reach1)], variant, dtype=dtype)
    ev = f.solve(iters, tol, check_every)
    return Subgame(ev["values"][0], ev["br_values"][0], f.average_strategy(0) if keep_strategy else [],
                   ev["iterations"][0], ev["exploitability"][0], ev["exploitability_raw"][0], ev["mass"][0], f.seconds, None)
