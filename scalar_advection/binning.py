"""
Shared helpers for radial/binning operations.
"""

from __future__ import annotations

import numpy as np

def find_ell_bin_edges(r_min: float, r_max: float, n_ell_bins: int) -> np.ndarray:
    """
    Compute integer-valued bin edges for isotropic shell averages.
    """
    r_min = max(1.0, float(r_min))
    r_max = max(r_min + 1.0, float(r_max))
    lo, hi = n_ell_bins + 1, 3 * n_ell_bins
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        edges = np.unique(np.rint(np.geomspace(r_min, r_max, mid)).astype(int))
        if len(edges) == n_ell_bins + 1:
            best = edges
            break
        if len(edges) < n_ell_bins + 1:
            lo = mid + 1
        else:
            hi = mid - 1
            best = edges
    if best is None:
        best = edges  # type: ignore[name-defined]
    return best


__all__ = ["find_ell_bin_edges"]
