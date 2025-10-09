"""
Utilities for fitting power laws to spectra/structure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PowerLawFit:
    """Container for a best-fit power law segment."""

    i: int
    j: int
    m: float
    A: float
    xseg: np.ndarray
    yfit: np.ndarray


def best_powerlaw_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 4,
    min_decades: float = 0.5,
    x_range: Tuple[float, float] | None = None,
) -> Optional[PowerLawFit]:
    """
    Find the log-log segment with the highest R^2 for a power law fit.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < min_points:
        return None

    if x_range is not None:
        lo, hi = x_range
        mask = (x >= lo) & (x <= hi)
        x = x[mask]
        y = y[mask]
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        if x.size < min_points:
            return None

    lx = np.log10(x)
    ly = np.log10(y)

    best = None
    best_r2 = -np.inf

    for i in range(0, x.size - min_points + 1):
        for j in range(i + min_points, x.size + 1):
            span = lx[j - 1] - lx[i]
            if span < min_decades:
                continue
            xi = lx[i:j]
            yi = ly[i:j]
            A = np.vstack([xi, np.ones_like(xi)]).T
            m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
            yi_fit = m * xi + c
            ss_res = np.sum((yi - yi_fit) ** 2)
            ss_tot = np.sum((yi - yi.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)
            if r2 > best_r2:
                best_r2 = r2
                xseg = x[i:j]
                yfit = 10 ** (m * np.log10(xseg) + c)
                best = PowerLawFit(i=i, j=j, m=m, A=10**c, xseg=xseg, yfit=yfit)

    return best


__all__ = ["PowerLawFit", "best_powerlaw_fit"]
