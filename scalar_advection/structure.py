"""
Structure function and increment statistics for 2-D fields.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .binning import find_ell_bin_edges
from .fitting import best_powerlaw_fit

# Optional imports ---------------------------------------------------------
try:  # pragma: no cover - optional acceleration
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_NUMBA = False

Array = np.ndarray
Field = Union[Array, Tuple[Array, Array]]

# -------------------------------------------------------------------------
# Displacement sampling
# -------------------------------------------------------------------------


def generate_displacements(ell_bin_edges: Array, n_per_bin: int, seed: int | None = None) -> Array:
    rng = np.random.default_rng(seed)
    starts = ell_bin_edges[:-1, None]
    stops = ell_bin_edges[1:, None]
    r_vals = np.geomspace(starts, stops, n_per_bin, axis=1).flatten()
    ang = rng.uniform(0.0, 2*np.pi, r_vals.size)
    dx = np.rint(r_vals * np.cos(ang)).astype(np.int32)
    dy = np.rint(r_vals * np.sin(ang)).astype(np.int32)
    disp = np.stack([dx, dy], axis=1)
    r = np.hypot(disp[:, 0], disp[:, 1])
    return disp[np.argsort(r)]


# -------------------------------------------------------------------------
# Radial profile utilities
# -------------------------------------------------------------------------


def radial_profile_from_map(arr2d: Array, ell_edges: Array) -> Tuple[Array, Array]:
    ny, nx = arr2d.shape
    y, x = np.indices((ny, nx))
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = np.hypot(x - cx, y - cy)
    n_bins = len(ell_edges) - 1
    bin_idx = np.digitize(r.ravel(), ell_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    sums = np.bincount(bin_idx[valid], weights=arr2d.ravel()[valid], minlength=n_bins)
    counts = np.bincount(bin_idx[valid], minlength=n_bins)
    prof = sums / np.maximum(1, counts)
    centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    return centers, prof


def s2_fft_scalar(field: Array, ell_edges: Array) -> Tuple[Array, Array]:
    F = np.fft.fft2(field)
    corr = np.fft.ifft2(np.abs(F) ** 2).real
    corr = np.fft.fftshift(corr) / field.size
    mu2 = np.mean(field**2)
    S2_map = 2.0 * (mu2 - corr)
    return radial_profile_from_map(S2_map, ell_edges)


def s2_fft_vector(ux: Array, uy: Array, ell_edges: Array) -> Tuple[Array, Array]:
    Fx = np.fft.fft2(ux)
    Fy = np.fft.fft2(uy)
    corr = np.fft.ifft2(np.abs(Fx) ** 2 + np.abs(Fy) ** 2).real
    corr = np.fft.fftshift(corr) / ux.size
    mu2 = np.mean(ux**2 + uy**2)
    S2_map = 2.0 * (mu2 - corr)
    return radial_profile_from_map(S2_map, ell_edges)


# -------------------------------------------------------------------------
# NumPy reference implementations (fallback)
# -------------------------------------------------------------------------


def _bin_index_for_radius(r: float, edges: Array) -> int:
    return int(np.searchsorted(edges, r, side="right") - 1)


def _sf_scalar_via_rolls(field: Array, orders: Array, ell_edges: Array, displacements: Array) -> Dict[str, Array]:
    n_bins = len(ell_edges) - 1
    n_orders = len(orders)
    sums = np.zeros((n_orders, n_bins), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for dx, dy in displacements:
        r = float(np.hypot(dx, dy))
        b = _bin_index_for_radius(r, ell_edges)
        if b < 0 or b >= n_bins:
            continue
        diff = np.roll(field, shift=(dy, dx), axis=(0, 1)) - field
        adiff = np.abs(diff)
        for j, p in enumerate(orders):
            sums[j, b] += np.mean(adiff**p)
        counts[b] += 1
    S = sums / np.maximum(1, counts)
    centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    return {"r": centers, "S": S, "counts": counts}


def _sf_vector_via_rolls(
    ux: Array,
    uy: Array,
    orders: Array,
    ell_edges: Array,
    displacements: Array,
    signed_longitudinal: bool,
) -> Dict[str, Array]:
    n_bins = len(ell_edges) - 1
    n_orders = len(orders)
    sums_mag = np.zeros((n_orders, n_bins), dtype=np.float64)
    sums_long = np.zeros((n_orders, n_bins), dtype=np.float64)
    sums_tran = np.zeros((n_orders, n_bins), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for dx, dy in displacements:
        r = float(np.hypot(dx, dy))
        b = _bin_index_for_radius(r, ell_edges)
        if b < 0 or b >= n_bins or r == 0.0:
            continue
        dux = np.roll(ux, shift=(dy, dx), axis=(0, 1)) - ux
        duy = np.roll(uy, shift=(dy, dx), axis=(0, 1)) - uy
        ex, ey = dx / r, dy / r
        long_ = dux * ex + duy * ey
        tran_ = -dux * ey + duy * ex
        mag_ = np.hypot(dux, duy)
        for j, p in enumerate(orders):
            sums_mag[j, b] += np.mean(np.abs(mag_) ** p)
            if signed_longitudinal:
                if abs(p - int(round(p))) > 1e-12:
                    raise ValueError("signed_longitudinal=True requires integer orders.")
                sums_long[j, b] += np.mean(long_**p)
            else:
                sums_long[j, b] += np.mean(np.abs(long_) ** p)
            sums_tran[j, b] += np.mean(np.abs(tran_) ** p)
        counts[b] += 1
    centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    return {
        "r": centers,
        "mag": sums_mag / np.maximum(1, counts),
        "long": sums_long / np.maximum(1, counts),
        "tran": sums_tran / np.maximum(1, counts),
        "counts": counts,
    }


# -------------------------------------------------------------------------
# Numba kernels
# -------------------------------------------------------------------------

if HAVE_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _scalar_means_per_disp(field, dxs, dys, orders):
        ny, nx = field.shape
        ndisp = dxs.size
        norders = orders.size
        out = np.zeros((ndisp, norders), dtype=np.float64)
        inv = 1.0 / (ny * nx)
        for m in prange(ndisp):
            dx = int(dxs[m])
            dy = int(dys[m])
            acc = np.zeros(norders, dtype=np.float64)
            for j in range(ny):
                jp = (j + dy) % ny
                for i in range(nx):
                    ip = (i + dx) % nx
                    d = field[jp, ip] - field[j, i]
                    ad = d if d >= 0.0 else -d
                    for k in range(norders):
                        p = orders[k]
                        acc[k] += ad**p
            for k in range(norders):
                out[m, k] = acc[k] * inv
        return out

    @njit(parallel=True, fastmath=True)
    def _vector_means_per_disp(ux, uy, dxs, dys, exs, eys, orders, signed_longitudinal):
        ny, nx = ux.shape
        ndisp = dxs.size
        norders = orders.size
        out_mag = np.zeros((ndisp, norders), dtype=np.float64)
        out_long = np.zeros((ndisp, norders), dtype=np.float64)
        out_tran = np.zeros((ndisp, norders), dtype=np.float64)
        inv = 1.0 / (ny * nx)
        for m in prange(ndisp):
            dx = int(dxs[m])
            dy = int(dys[m])
            ex = exs[m]
            ey = eys[m]
            acc_mag = np.zeros(norders, dtype=np.float64)
            acc_long = np.zeros(norders, dtype=np.float64)
            acc_tran = np.zeros(norders, dtype=np.float64)
            for j in range(ny):
                jp = (j + dy) % ny
                for i in range(nx):
                    ip = (i + dx) % nx
                    dux = ux[jp, ip] - ux[j, i]
                    duy = uy[jp, ip] - uy[j, i]
                    long_ = dux * ex + duy * ey
                    tran_ = -dux * ey + duy * ex
                    mag_ = (dux * dux + duy * duy) ** 0.5
                    al = long_ if signed_longitudinal else (long_ if long_ >= 0.0 else -long_)
                    at = tran_ if tran_ >= 0.0 else -tran_
                    am = mag_
                    for k in range(norders):
                        p = orders[k]
                        acc_mag[k] += am**p
                        acc_long[k] += (long_**p) if signed_longitudinal else (al**p)
                        acc_tran[k] += at**p
            for k in range(norders):
                out_mag[m, k] = acc_mag[k] * inv
                out_long[m, k] = acc_long[k] * inv
                out_tran[m, k] = acc_tran[k] * inv
        return out_mag, out_long, out_tran


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def structure_functions(
    field: Field,
    orders: Sequence[float] = (1, 2, 3),
    *,
    r_min: float = 1.0,
    r_max: float | None = None,
    n_ell_bins: int = 40,
    n_disp_total: int = 2048,
    use_fft_for_p2: bool = True,
    signed_longitudinal: bool = False,
    seed: int | None = None,
) -> Dict[str, Array]:
    """
    Compute isotropic structure functions for scalar or vector fields.
    """
    if isinstance(field, (tuple, list)):
        ux, uy = field
        ny, nx = ux.shape
    else:
        ny, nx = field.shape

    if r_max is None:
        r_max = min(ny, nx) // 2

    ell_edges = find_ell_bin_edges(r_min, r_max, n_ell_bins)
    n_per_bin = max(1, n_disp_total // n_ell_bins)
    disps = generate_displacements(ell_edges, n_per_bin, seed=seed)
    orders_arr = np.asarray(orders, dtype=np.float64)

    r = np.hypot(disps[:, 0], disps[:, 1]).astype(np.float64)
    b = np.searchsorted(ell_edges, r, side="right") - 1
    valid = (b >= 0) & (b < (len(ell_edges) - 1)) & (r > 0.0)
    dxs = disps[valid, 0].astype(np.int32)
    dys = disps[valid, 1].astype(np.int32)
    bins = b[valid].astype(np.int32)
    exs = (dxs / r[valid]).astype(np.float64) if isinstance(field, (tuple, list)) else None
    eys = (dys / r[valid]).astype(np.float64) if isinstance(field, (tuple, list)) else None

    n_bins = len(ell_edges) - 1
    n_orders = len(orders_arr)

    if isinstance(field, (tuple, list)):
        ux, uy = field
        if HAVE_NUMBA:
            out_mag_d, out_long_d, out_tran_d = _vector_means_per_disp(
                ux, uy, dxs, dys, exs, eys, orders_arr, signed_longitudinal
            )
            counts = np.bincount(bins, minlength=n_bins)
            mag = np.vstack(
                [np.bincount(bins, weights=out_mag_d[:, j], minlength=n_bins) for j in range(n_orders)]
            ) / np.maximum(1, counts)
            long = np.vstack(
                [np.bincount(bins, weights=out_long_d[:, j], minlength=n_bins) for j in range(n_orders)]
            ) / np.maximum(1, counts)
            tran = np.vstack(
                [np.bincount(bins, weights=out_tran_d[:, j], minlength=n_bins) for j in range(n_orders)]
            ) / np.maximum(1, counts)
            centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
            out = {"r": centers, "mag": mag, "long": long, "tran": tran, "counts": counts}
        else:
            out = _sf_vector_via_rolls(ux, uy, orders_arr, ell_edges, disps, signed_longitudinal)
        if use_fft_for_p2 and np.any(np.isclose(orders_arr, 2.0)):
            r_fft, s2_fft = s2_fft_vector(ux, uy, ell_edges)
            j2 = np.where(np.isclose(orders_arr, 2.0))[0]
            out["mag"][j2, :] = s2_fft
    else:
        field = np.asarray(field)
        if HAVE_NUMBA:
            out_d = _scalar_means_per_disp(field, dxs, dys, orders_arr)
            counts = np.bincount(bins, minlength=n_bins)
            S = np.vstack(
                [np.bincount(bins, weights=out_d[:, j], minlength=n_bins) for j in range(n_orders)]
            ) / np.maximum(1, counts)
            centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
            out = {"r": centers, "S": S, "counts": counts}
        else:
            out = _sf_scalar_via_rolls(field, orders_arr, ell_edges, disps)
        if use_fft_for_p2 and np.any(np.isclose(orders_arr, 2.0)):
            j2 = np.where(np.isclose(orders_arr, 2.0))[0]
            r_fft, s2_fft = s2_fft_scalar(field, ell_edges)
            out["S"][j2, :] = s2_fft

    out["orders"] = orders_arr
    out["ell_edges"] = ell_edges
    return out


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------


def _pth_root_curves(curves: np.ndarray, orders: np.ndarray) -> np.ndarray:
    out = np.empty_like(curves)
    for j, p in enumerate(orders):
        out[j] = np.power(np.maximum(curves[j], 0.0), 1.0 / p)
    return out


def plot_structure_functions(
    result: Dict[str, Array],
    *,
    fname: str | None = None,
    title: str | None = None,
    annotate_fits: bool = False,
    fit_min_points: int = 6,
    fit_min_decades: float = 0.6,
    plot_long_and_tran: bool = True,
    fit_min_r: float | None = None,
    fit_max_r: float | None = None,
) -> None:
    """
    Plot structure functions and overlay best-fit power-law segments.
    """
    r = result["r"]
    orders = result["orders"]
    is_vector = "mag" in result

    plt.figure(figsize=(8.0, 5.2), dpi=140)
    ax = plt.gca()

    def _plot_set(curves: np.ndarray, label_prefix: str, style: str = "-", z: int = 0):
        colors = plt.cm.tab10.colors
        for j, p in enumerate(orders):
            y = curves[j]
            color = colors[j % len(colors)]
            ax.loglog(r, y, style, color=color, lw=1.8, alpha=0.9, label=fr"{label_prefix} p={p:g}")
            x_range = None
            if fit_min_r is not None or fit_max_r is not None:
                lo = fit_min_r if fit_min_r is not None else r.min()
                hi = fit_max_r if fit_max_r is not None else r.max()
                x_range = (lo, hi)

            fit = best_powerlaw_fit(
                r,
                y,
                min_points=fit_min_points,
                min_decades=fit_min_decades,
                x_range=x_range,
            )
            if fit is None:
                continue
            ax.loglog(
                fit.xseg,
                fit.yfit,
                color=color,
                lw=4,
                alpha=0.5,
                solid_capstyle="round",
                zorder=3 + z,
                label=fr"$\propto r^{{{fit.m:.3f}}}$",
            )
            if annotate_fits:
                mid = (fit.i + fit.j) // 2
                ax.text(
                    r[mid],
                    y[mid],
                    fr"$m={fit.m:.3f}$",
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

    if is_vector:
        mag_rt = _pth_root_curves(result["mag"], orders)
        _plot_set(mag_rt, "|δu|,", "-")
        if plot_long_and_tran:
            long_rt = _pth_root_curves(result["long"], orders)
            tran_rt = _pth_root_curves(result["tran"], orders)
            _plot_set(long_rt, "long,", "--", z=1)
            _plot_set(tran_rt, "tran,", ":", z=2)
        ylabel = r"$(S_p)^{1/p}$"
    else:
        S_rt = _pth_root_curves(result["S"], orders)
        _plot_set(S_rt, "p=")
        ylabel = r"$(S_p)^{1/p}$"

    ax.set_xlabel(r"separation $r$ [pixels]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(ncol=1, fontsize=9, frameon=False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------
# Pair-increment PDF
# -------------------------------------------------------------------------


def _edges_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _uniform_edges(edges: np.ndarray, rtol=1e-10, atol=1e-12) -> bool:
    d = np.diff(edges)
    return np.allclose(d, d[0], rtol=rtol, atol=atol)


try:  # pragma: no cover - optional dependency
    from numba import njit as _njit, prange as _prange

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

if _HAVE_NUMBA:

    @_njit(parallel=True, fastmath=True)
    def _pdf_vector_uniform(
        ux,
        uy,
        dxs,
        dys,
        exs,
        eys,
        bins,
        kind_id,
        signed_longitudinal,
        du_min,
        inv_ddu,
        n_du_bins,
        n_ell_bins,
    ):
        ny, nx = ux.shape
        counts = np.zeros((n_ell_bins, n_du_bins), dtype=np.int64)
        for m in _prange(dxs.size):
            dx = int(dxs[m])
            dy = int(dys[m])
            b = int(bins[m])
            if b < 0 or b >= n_ell_bins:
                continue
            ex = exs[m]
            ey = eys[m]
            for j in range(ny):
                jp = (j + dy) % ny
                for i in range(nx):
                    ip = (i + dx) % nx
                    dux = ux[jp, ip] - ux[j, i]
                    duy = uy[jp, ip] - uy[j, i]
                    if kind_id == 0:
                        val = (dux * dux + duy * duy) ** 0.5
                    elif kind_id == 1:
                        val = dux * ex + duy * ey
                        if not signed_longitudinal and val < 0.0:
                            val = -val
                    else:
                        val = -dux * ey + duy * ex
                        if val < 0.0:
                            val = -val
                    k = int(np.floor((val - du_min) * inv_ddu))
                    if 0 <= k < n_du_bins:
                        counts[b, k] += 1
        return counts


def pair_increment_pdf(
    field,
    *,
    kind: str = "mag",
    signed_longitudinal: bool = False,
    r_min: float = 1.0,
    r_max: float | None = None,
    n_ell_bins: int = 40,
    n_disp_total: int = 2048,
    du_edges: np.ndarray | None = None,
    n_du_bins: int = 128,
    du_max: float | None = None,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Build a 2D histogram of increment PDFs versus separation.
    """
    if isinstance(field, (tuple, list)):
        ux, uy = field
        ny, nx = ux.shape
        is_vector = True
    else:
        raise ValueError("pair_increment_pdf expects a vector field (ux, uy).")

    if r_max is None:
        r_max = min(ny, nx) // 2
    ell_edges = find_ell_bin_edges(r_min, r_max, n_ell_bins)
    n_per_bin = max(1, n_disp_total // n_ell_bins)
    disps = generate_displacements(ell_edges, n_per_bin, seed=seed)

    r = np.hypot(disps[:, 0], disps[:, 1]).astype(np.float64)
    bins = np.searchsorted(ell_edges, r, side="right") - 1
    valid = (bins >= 0) & (bins < (len(ell_edges) - 1)) & (r > 0.0)
    dxs = disps[valid, 0].astype(np.int32)
    dys = disps[valid, 1].astype(np.int32)
    bins = bins[valid].astype(np.int32)
    exs = (dxs / r[valid]).astype(np.float64)
    eys = (dys / r[valid]).astype(np.float64)
    n_ell = len(ell_edges) - 1

    spd_max = 2.0 * float(np.max(np.hypot(ux, uy))) + 1e-12
    if du_edges is None:
        if du_max is None:
            du_max = spd_max
        du_min = max(du_max * 1e-3, 1e-6)
        if kind == "long" and signed_longitudinal:
            n_pos = n_du_bins // 2
            pos = np.geomspace(du_min, du_max, n_pos + 1)
            neg = -pos[::-1]
            du_edges = np.concatenate([neg[:-1], [0.0], pos])
        else:
            du_edges = np.geomspace(du_min, du_max, n_du_bins + 1)

    du_edges = np.asarray(du_edges, dtype=np.float64)
    du_centers = _edges_centers(du_edges)
    n_du = len(du_centers)

    counts_disp = np.bincount(bins, minlength=n_ell)
    pairs_per_bin = counts_disp * (nx * ny)
    counts2d = np.zeros((n_ell, n_du), dtype=np.int64)

    if _HAVE_NUMBA and _uniform_edges(du_edges):
        kind_id = 0 if kind == "mag" else (1 if kind == "long" else 2)
        ddu = du_edges[1] - du_edges[0]
        inv_ddu = 1.0 / ddu
        counts2d = _pdf_vector_uniform(
            ux,
            uy,
            dxs,
            dys,
            exs,
            eys,
            bins,
            kind_id,
            signed_longitudinal,
            du_edges[0],
            inv_ddu,
            n_du,
            n_ell,
        )
    else:
        for dx, dy, b, ex, ey in zip(dxs, dys, bins, exs, eys):
            dux = np.roll(ux, shift=(dy, dx), axis=(0, 1)) - ux
            duy = np.roll(uy, shift=(dy, dx), axis=(0, 1)) - uy
            if kind == "mag":
                vals = np.hypot(dux, duy)
            elif kind == "long":
                vals = dux * ex + duy * ey
                if not signed_longitudinal:
                    vals = np.abs(vals)
            elif kind == "tran":
                vals = -dux * ey + duy * ex
                vals = np.abs(vals)
            else:
                raise ValueError("kind must be 'mag', 'long', or 'tran'")
            h, _ = np.histogram(vals.ravel(), bins=du_edges)
            counts2d[b, :] += h.astype(np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        pdf = counts2d / np.maximum(pairs_per_bin[:, None], 1)

    mass_captured = pdf.sum(axis=1)

    return {
        "r": _edges_centers(ell_edges),
        "ell_edges": ell_edges,
        "du_edges": du_edges,
        "pdf": pdf,
        "mass_captured": mass_captured,
        "counts_disp": counts_disp,
        "pairs_per_bin": pairs_per_bin,
        "kind": kind,
        "signed_longitudinal": signed_longitudinal,
    }


def plot_pair_increment_pdf(
    pdf_result: Dict[str, np.ndarray],
    *,
    ax: plt.Axes | None = None,
    fname: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ell_on_x: bool = True,
    log_axes: bool = True,
    log_norm: bool = True,
    norm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot the pair-increment PDF as a heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=140)
    else:
        fig = ax.figure

    pdf = pdf_result["pdf"]
    r = pdf_result["r"]
    du = _edges_centers(pdf_result["du_edges"])

    if ell_on_x:
        x = r
        y = du
        Z = pdf.T
        xlabel = r"separation $\ell$"
        ylabel = r"increment $|\delta u|$"
    else:
        x = du
        y = r
        Z = pdf
        xlabel = r"increment $|\delta u|$"
        ylabel = r"separation $\ell$"

    X, Y = np.meshgrid(x, y, indexing="xy")

    norm = None
    vmin_arg = vmin
    vmax_arg = vmax
    if log_norm:
        params: Dict[str, Any] = dict(norm_kwargs or {})
        if vmin is not None:
            params.setdefault("vmin", vmin)
        if vmax is not None:
            params.setdefault("vmax", vmax)
        positive = Z[Z > 0]
        if not positive.size:
            positive = np.array([1e-12])
        params.setdefault("vmin", max(np.min(positive), 1e-12))
        params.setdefault("vmax", np.max(positive))
        norm = LogNorm(**params)
        vmin_arg = None
        vmax_arg = None

    Z = np.asarray(Z, dtype=float)

    if log_norm:
        Z_plot = np.where(Z > 0.0, Z, np.nan)
    else:
        Z_plot = Z

    pcm = ax.pcolormesh(
        X,
        Y,
        Z_plot,
        shading="auto",
        cmap=cmap,
        vmin=vmin_arg,
        vmax=vmax_arg,
        norm=norm,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("fraction of pairs")

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if fname:
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    elif ax is None:
        plt.show()


def structure_functions_from_pair_pdf(
    pdf_result: Dict[str, np.ndarray],
    orders: Sequence[float] = (1, 2, 3),
) -> Dict[str, np.ndarray]:
    """
    Build structure functions by taking moments over the pair-increment PDF.
    """
    du_edges = pdf_result["du_edges"]
    du_c = _edges_centers(du_edges)
    pdf = pdf_result["pdf"]
    r = pdf_result["r"]
    kind = pdf_result["kind"]
    signedL = pdf_result["signed_longitudinal"]

    orders = np.asarray(orders, dtype=float)
    use_abs = not (kind == "long" and signedL)
    base = np.abs(du_c) if use_abs else du_c

    S = np.empty((orders.size, r.size), dtype=float)
    for j, p in enumerate(orders):
        weights = base**p
        S[j, :] = pdf @ weights

    out = {"r": r, "orders": orders, "bins": pdf_result["ell_edges"]}
    if kind == "mag":
        out["mag"] = S
    elif kind == "long":
        out["long"] = S
    else:
        out["tran"] = S
    out["counts"] = pdf_result["counts_disp"]
    return out


__all__ = [
    "structure_functions",
    "plot_structure_functions",
    "pair_increment_pdf",
    "plot_pair_increment_pdf",
    "structure_functions_from_pair_pdf",
    "radial_profile_from_map",
    "s2_fft_scalar",
    "s2_fft_vector",
]
