"""
Fractal-dimension utilities (box-counting) for binary masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .fitting import best_powerlaw_fit


@dataclass
class BoxCountingResult:
    box_sizes: np.ndarray
    inv_box_sizes: np.ndarray
    counts: np.ndarray
    slope: Optional[float]
    intercept: Optional[float]
    fit_indices: Tuple[int, int] | None


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("box_counting expects a 2D array")
    return m


def box_counting(
    mask: np.ndarray,
    *,
    box_sizes: Sequence[int] | None = None,
    min_box: int = 1,
    max_box: Optional[int] = None,
    fit_min_size: Optional[int] = None,
    fit_max_size: Optional[int] = None,
    fit_min_points: int = 3,
    fit_min_decades: float = 0.3,
) -> BoxCountingResult:
    """
    Perform standard box-counting on a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (non-zero values treated as filled).
    box_sizes : sequence[int], optional
        Explicit list of box sizes in pixels. If None, powers of two between
        ``min_box`` and ``max_box`` are used.
    min_box : int
        Minimum box size (pixels) when auto-generating sizes.
    max_box : int, optional
        Maximum box size (defaults to min(mask.shape)).
    fit_min_size, fit_max_size : int, optional
        Box-size bounds (inclusive) for the log-log fit.
    """
    mask_bool = _prepare_mask(mask)
    h, w = mask_bool.shape
    max_dim = min(h, w)
    if max_box is None:
        max_box = max_dim

    if box_sizes is None:
        sizes = []
        s = max(1, min_box)
        while s <= max_box:
            sizes.append(s)
            s *= 2
        if sizes[-1] < max_box:
            sizes.append(max_box)
        box_sizes_arr = np.array(sizes, dtype=int)
    else:
        box_sizes_arr = np.array(sorted(set(int(s) for s in box_sizes if s >= 1)), dtype=int)
        box_sizes_arr = box_sizes_arr[(box_sizes_arr >= min_box) & (box_sizes_arr <= max_box)]

    counts = []
    for size in box_sizes_arr:
        nrows = int(np.ceil(h / size))
        ncols = int(np.ceil(w / size))
        padded = np.zeros((nrows * size, ncols * size), dtype=bool)
        padded[:h, :w] = mask_bool
        reshaped = padded.reshape(nrows, size, ncols, size)
        occupancy = reshaped.any(axis=(1, 3))
        counts.append(int(np.count_nonzero(occupancy)))

    counts_arr = np.array(counts, dtype=float)

    # Fit to log-log data: log(N) vs log(1/size)
    sizes_for_fit = box_sizes_arr.astype(float)
    inv_sizes = 1.0 / sizes_for_fit

    mask_fit = np.ones_like(sizes_for_fit, dtype=bool)
    if fit_min_size is not None:
        mask_fit &= sizes_for_fit >= fit_min_size
    if fit_max_size is not None:
        mask_fit &= sizes_for_fit <= fit_max_size

    fit = None
    fit_indices = None
    if np.any(mask_fit):
        inv_subset = inv_sizes[mask_fit]
        counts_subset = counts_arr[mask_fit]
        fit = best_powerlaw_fit(
            inv_subset,
            counts_subset,
            min_points=fit_min_points,
            min_decades=fit_min_decades,
        )
        if fit is not None:
            sel_indices = np.flatnonzero(mask_fit)
            subset = sel_indices[fit.i:fit.j]
            if subset.size:
                fit_indices = (int(subset.min()), int(subset.max()) + 1)
    if fit is not None:
        slope = fit.m
        intercept = np.log10(fit.A)
    else:
        slope = None
        intercept = None

    return BoxCountingResult(
        box_sizes=box_sizes_arr,
        inv_box_sizes=inv_sizes,
        counts=counts_arr,
        slope=slope,
        intercept=intercept,
        fit_indices=fit_indices,
    )


__all__ = ["BoxCountingResult", "box_counting"]
