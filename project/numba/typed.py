"""
Local lightweight stub for `numba.typed`.

Only implements `List`, used by SHAP during import-time for image masking utilities.
"""

from __future__ import annotations


class List(list):
    """Replacement for `numba.typed.List`."""

