"""
Local lightweight stub for `numba`.

Why this exists:
- Your environment's installed `numba` fails to import on Windows (DLL load error).
- SHAP imports `numba` during import-time, even if we don't need JIT compilation for our use case.

This stub provides a minimal API surface (`njit`, `prange`, `typed.List`) so `import shap` works.
It does *not* provide real JIT compilation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def njit(*args: Any, **kwargs: Any):
    """
    Identity decorator replacement for `numba.njit`.

    Supports both usages:
    - @njit
    - @njit(...)
    """

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator


def jit(*args: Any, **kwargs: Any):
    """Alias for njit identity decorator."""

    return njit(*args, **kwargs)


def prange(*args: Any, **kwargs: Any):
    """Best-effort replacement for `numba.prange`."""

    return range(*args)


class _TypedModule:
    class List(list):
        """Replacement for `numba.typed.List`."""


# Provide `numba.typed` namespace
typed = _TypedModule()

