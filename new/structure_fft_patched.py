"""
FFT-accelerated Structure Function calculation for Scalar Fields.

This module computes isotropic structure functions for 2D scalar fields.

For even integer orders ``p``, we compute

    ``S_p(r) = < |u(x+r) - u(x)|^p > = < (u(x+r) - u(x))^p >``

exactly via a binomial expansion and FFT-based cross-correlations.

For odd integer orders, ``|u(x+r) - u(x)|^p`` is not a polynomial in the pair
``(u(x+r), u(x))``, so we fall back to a displacement-sampling estimator that
matches the absolute-increment convention used elsewhere in this repository.

Method:
    S_p(r) = < (u(x+r) - u(x))^p >
           = Sum_{k=0 to p} [ binomial(p, k) * (-1)^(p-k) * < u(x+r)^k * u(x)^(p-k) > ]

    The term <...> is the Cross-Correlation of (u^k) and (u^(p-k)), calculated efficiently via FFT.
"""

from __future__ import annotations

from math import comb
from typing import Dict, Sequence, Tuple

import numpy as np

from .binning import find_ell_bin_edges

Array = np.ndarray
Field = Array  # Explicitly scalar for this implementation


def radial_profile_from_map(arr2d: Array, ell_edges: Array) -> Tuple[Array, Array]:
    """
    Compute the isotropic radial profile of a 2D map.
    """
    ny, nx = arr2d.shape
    y, x = np.indices((ny, nx))

    # Center frequencies for FFT shift (0 is at center)
    cy, cx = ny // 2, nx // 2

    r = np.hypot(x - cx, y - cy)
    n_bins = len(ell_edges) - 1

    # Digitize
    bin_idx = np.searchsorted(ell_edges, r.ravel(), side='right') - 1

    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    weights = arr2d.ravel()

    sums = np.bincount(bin_idx[valid], weights=weights[valid], minlength=n_bins)
    counts = np.bincount(bin_idx[valid], minlength=n_bins)

    prof = sums / np.maximum(1, counts)

    centers = np.sqrt(ell_edges[:-1] * ell_edges[1:])
    return centers, prof


def _fft_correlate(field1: Array, field2: Array) -> Array:
    """
    Compute Cross-Correlation of two 2D fields using FFTs.
    Returns the centered correlation map.
    """
    # 1. FFT
    f1 = np.fft.rfft2(field1)
    f2 = np.fft.rfft2(field2)

    # 2. Multiply (Complex Conjugate for Correlation)
    # Corr(A, B) = IFFT( FFT(A) * Conj(FFT(B)) )
    # Note: We compute <u(x+r)u(x)>. The standard convolution theorem
    # usually gives A(x) * B(r-x). Correlation is A(x+r) * B(x).
    spec = f1 * np.conj(f2)

    # 3. Inverse FFT
    corr = np.fft.irfft2(spec, s=field1.shape)

    # 4. Normalize
    corr /= field1.size

    # 5. Shift so zero-lag is at the center
    return np.fft.fftshift(corr)


def _corr_from_rffts(f1: Array, f2: Array, shape: Tuple[int, int], norm: float) -> Array:
    """Centered correlation map from precomputed real FFTs."""
    corr = np.fft.irfft2(f1 * np.conj(f2), s=shape)
    corr /= norm
    return np.fft.fftshift(corr)


def _slice_overlap(a: Array, dx: int, dy: int) -> Tuple[Array, Array]:
    """
    Return views over the overlapping region for a non-periodic shift.

    The second view corresponds to ``a`` shifted by (dy, dx):
    ``a1[y, x] = a[y+dy, x+dx]``.
    """
    ny, nx = a.shape
    y0 = max(0, -dy)
    y1 = min(ny, ny - dy)
    x0 = max(0, -dx)
    x1 = min(nx, nx - dx)
    if (y1 <= y0) or (x1 <= x0):
        empty = a[:0, :0]
        return empty, empty
    return a[y0:y1, x0:x1], a[y0 + dy : y1 + dy, x0 + dx : x1 + dx]


def structure_functions_fft(
    field: Field,
    orders: Sequence[int] = (2, 3, 4),
    *,
    r_min: float = 1.0,
    r_max: float | None = None,
    n_ell_bins: int = 40,
    n_disp_total: int = 2048,
    seed: int | None = None,
    pad: bool = False,
) -> Dict[str, Array]:
    """
    Compute isotropic Structure Functions for a Scalar field using FFTs.

    Args:
        field: 2D numpy array (scalar field).
        orders: Sequence of integer orders.
        n_disp_total: Total displacements to sample for odd orders.
        seed: Random seed for displacement sampling.
        pad: If True, pads input with zeros to avoid periodic wrap-around effects.
             (Resulting calculation is effectively on a non-periodic domain).

    Returns:
        Dictionary with keys 'r' (radii) and 'S' (Structure Functions shape (n_orders, n_bins)).
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("FFT Structure Functions currently implemented for 2D Scalar fields only.")

    # Validation
    orders_int: list[int] = []
    for p in orders:
        if not float(p).is_integer():
            raise ValueError(f"structure_functions_fft requires integer orders; got {p!r}")
        p_int = int(p)
        if p_int < 1:
            raise ValueError(f"structure_functions_fft requires orders >= 1; got {p_int}")
        orders_int.append(p_int)

    ny, nx = field.shape
    if r_max is None:
        r_max = min(ny, nx) // 2

    ell_edges = find_ell_bin_edges(r_min, r_max, n_ell_bins)
    n_bins = len(ell_edges) - 1
    r_centers = np.sqrt(ell_edges[:-1] * ell_edges[1:])
    n_orders = len(orders_int)
    S_profiles = np.zeros((n_orders, n_bins), dtype=np.float64)

    even_orders = sorted({p for p in orders_int if (p % 2) == 0})
    odd_orders = sorted({p for p in orders_int if (p % 2) == 1})

    # ------------------------------------------------------------------
    # Even orders: exact FFT binomial method (|diff|^p == diff^p for even p)
    # ------------------------------------------------------------------
    if even_orders:
        if pad:
            work_shape = (2 * ny, 2 * nx)
            work_field = np.zeros(work_shape, dtype=np.float64)
            work_field[:ny, :nx] = field.astype(np.float64, copy=False)
            mask = np.zeros(work_shape, dtype=np.float64)
            mask[:ny, :nx] = 1.0
        else:
            work_field = field.astype(np.float64, copy=False)
            work_shape = work_field.shape
            mask = np.ones_like(work_field, dtype=np.float64)

        norm = float(work_field.size)

        needed_powers: set[int] = set()
        for p in even_orders:
            needed_powers.update(range(p + 1))

        rffts: dict[int, Array] = {}
        for k in sorted(needed_powers):
            if k == 0:
                base = mask
            elif k == 1:
                base = work_field
            else:
                base = np.power(work_field, k)
            rffts[k] = np.fft.rfft2(base)

        weight_map = None
        if pad:
            weight_map = _corr_from_rffts(rffts[0], rffts[0], work_shape, norm)

        for p in even_orders:
            Sp_map = np.zeros(work_shape, dtype=np.float64)
            for k in range(p + 1):
                # binomial(p,k) * (-1)^(p-k)
                coeff = float(comb(p, k)) * (1.0 if ((p - k) % 2) == 0 else -1.0)
                Sp_map += coeff * _corr_from_rffts(rffts[k], rffts[p - k], work_shape, norm)

            if pad:
                assert weight_map is not None
                Sp_map = np.divide(Sp_map, weight_map, out=np.zeros_like(Sp_map), where=weight_map > 0.0)

            _, prof = radial_profile_from_map(Sp_map, ell_edges)
            for i, p_req in enumerate(orders_int):
                if p_req == p:
                    S_profiles[i, :] = prof

    # ------------------------------------------------------------------
    # Odd orders: displacement sampling using absolute increments
    # ------------------------------------------------------------------
    if odd_orders:
        if n_disp_total <= 0:
            raise ValueError("Odd orders require n_disp_total > 0 for sampling.")
        if pad:
            from .structure import generate_displacements

            n_per_bin = max(1, int(n_disp_total) // int(n_ell_bins))
            disps = generate_displacements(ell_edges, n_per_bin, seed=seed, ndim=2)
            sums = {p: np.zeros(n_bins, dtype=np.float64) for p in odd_orders}
            counts = np.zeros(n_bins, dtype=np.int64)

            for dx, dy in disps:
                r = float(np.hypot(dx, dy))
                b = int(np.searchsorted(ell_edges, r, side="right") - 1)
                if b < 0 or b >= n_bins or r == 0.0:
                    continue
                a0, a1 = _slice_overlap(field, int(dx), int(dy))
                if a0.size == 0:
                    continue
                diff = a1.astype(np.float64, copy=False) - a0.astype(np.float64, copy=False)
                adiff = np.abs(diff)
                for p in odd_orders:
                    sums[p][b] += float(np.mean(np.power(adiff, p)))
                counts[b] += 1

            for p in odd_orders:
                prof = sums[p] / np.maximum(1, counts)
                for i, p_req in enumerate(orders_int):
                    if p_req == p:
                        S_profiles[i, :] = prof
        else:
            # Periodic: reuse the numba-accelerated estimator in structure.py
            from .structure import structure_functions as _sf_sample

            sampled = _sf_sample(
                field,
                orders=tuple(odd_orders),
                r_min=r_min,
                r_max=r_max,
                n_ell_bins=n_ell_bins,
                n_disp_total=n_disp_total,
                use_fft_for_p2=False,
                seed=seed,
            )
            for j, p in enumerate(odd_orders):
                prof = sampled["S"][j]
                for i, p_req in enumerate(orders_int):
                    if p_req == p:
                        S_profiles[i, :] = prof

    return {
        "r": r_centers,
        "S": S_profiles,
        "ell_edges": ell_edges,
        "orders": np.array(orders_int, dtype=np.float64),
    }
