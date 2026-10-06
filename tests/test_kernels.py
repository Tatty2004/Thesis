"""Every compiled kernel's parallel and serial variants (core/jit.py) give identical
results. Small games only reach the serial variants and the real deck the parallel ones,
so this is what lets small-game tests vouch for real-deck runs."""
from itertools import combinations

import numpy as np
import pytest

from bombpot.core.jit import Kernel
from bombpot.core.removal import HandSet, _beats_kernel, _fold_kernel
from bombpot.core.solvers.range_cfr import _expectation
from bombpot.core.solvers.range_cfr import _update as range_update
from bombpot.core.solvers.sampled_cfr import _update as sampled_update

rng = np.random.default_rng(0)
HANDS = HandSet(np.array(list(combinations(range(14), 2))), 14)
D, L, H, B, N, A = 9, 4, len(HANDS), 2, 500, 3
REACH, VALID, KAPPA = rng.random((D, L, H)), rng.random((D, H)) > 0.2, rng.random(D)
STRENGTH = rng.integers(0, 30, size=(D, B, H))  # many ties
ORDER = np.argsort(STRENGTH, axis=2, kind="stable")
_s = np.take_along_axis(STRENGTH, ORDER, axis=2)
_last = np.ones(_s.shape, bool)
_last[..., :-1] = _s[..., 1:] != _s[..., :-1]
GROUP_END = np.minimum.accumulate(np.where(_last, np.arange(H) + 1, H)[..., ::-1], axis=2)[..., ::-1].copy()
SIG = rng.dirichlet(np.ones(A), size=N)

CASES = {
    "fold": (_fold_kernel, lambda: (REACH, VALID, KAPPA, HANDS.subs, HANDS.signs, HANDS.n_subsets, np.empty_like(REACH))),
    "beats": (_beats_kernel, lambda: (REACH, VALID, KAPPA, ORDER, GROUP_END, HANDS.subs, HANDS.signs, HANDS.n_subsets,
                                      np.empty_like(REACH))),
    "expectation": (_expectation, lambda: (SIG, rng.normal(size=(A, N)), np.empty(N), np.empty((N, A)))),
    **{f"range_update_{v}": (range_update, lambda v=v: (rng.normal(size=(N, A)), rng.normal(size=(N, A)), SIG.copy(),
                                                        rng.random((N, A)), rng.random(N), 4.0, 0.7, 0.5, v))
       for v in (0, 1, 2)},
    "sampled_update": (sampled_update, lambda: (rng.normal(size=(N, A)), rng.normal(size=(N, A)), SIG.copy(),
                                                rng.random((N, A)), rng.random(N), 3.0)),
}


@pytest.mark.parametrize("name", CASES)
def test_parallel_equals_serial(name):
    kernel, make = CASES[name]
    args = make()
    outs = []
    for variant in (kernel.serial, kernel.parallel):
        a = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        variant(*a)
        outs.append(a)
    for x, y in zip(*outs):
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)


def test_variants_are_cached_separately():
    # numba's cache ignores the parallel flag, so the variants must differ in name.
    k = Kernel(lambda x: x)
    assert k.serial.py_func.__qualname__ != k.parallel.py_func.__qualname__
