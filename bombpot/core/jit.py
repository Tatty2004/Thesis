"""Compile a numba kernel twice, in parallel and serially, and pick by problem size.

Starting numba's threads costs a few milliseconds per call (with the default workqueue
threading layer), more than the work itself on the small games used in tests, so small
calls run serially. Large calls (the real deck) run in parallel.

numba's on-disk cache keys a function by its name, source line, signature and bytecode,
not by the `parallel` flag, so the serial variant is compiled from a renamed copy of the
function: otherwise the two variants would share, and overwrite, one cache entry.
"""
from __future__ import annotations

import types

from numba import njit

PARALLEL_FROM = 4_000_000  # operations above which a call runs in parallel


def _renamed(fn, suffix: str):
    """The same function under another name, so numba caches it separately."""
    out = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__ + suffix, fn.__defaults__, fn.__closure__)
    out.__qualname__ = fn.__qualname__ + suffix
    out.__module__ = fn.__module__
    out.__doc__ = fn.__doc__
    return out


class Kernel:
    def __init__(self, fn):
        self.parallel = njit(parallel=True, cache=True)(fn)
        self.serial = njit(cache=True)(_renamed(fn, "_serial"))

    def __call__(self, work: int, *args):
        return (self.parallel if work >= PARALLEL_FROM else self.serial)(*args)


def kernel(fn) -> Kernel:
    return Kernel(fn)
