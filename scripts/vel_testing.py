#!/usr/bin/env python3
"""
Compute and plot velocity structure functions (FFT-based).

This is a drop-in companion to `scripts/compute_velocity_sf_1024.py`, but it uses
`scalar_advection.structure_functions_fft` for the structure functions.

Notes
-----
- `structure_functions_fft` is implemented for 2D *scalar* fields. For a 2D
  velocity field (ux, uy), we compute per-component structure functions and
  then average:  S_p = 0.5 * (S_p[ux] + S_p[uy]).
- Even orders are computed exactly via FFT/binomial correlations.
- Odd orders use an absolute-increment displacement sampler under the hood.

Generates velocity fields with the same parameters as the alpha-kappa 1024 campaign (scaled down for speed):
- N=512, lam_min=2, lam_max=512, wavelet=mexh, seed=1000
- alphas = [1/6, 1/3, 1/2, 2/3, 5/6]
- orders = (1, 2, 3, 4, 6, 8, 10)

Produces per-alpha plots:
- Velocity field map (|u| with downsampled quiver overlay)
- Velocity energy spectrum E(k) with power-law fits (free + fixed slope)
- Top panel: (S_p)^{1/p} vs r with power-law fits
- Bottom panel: zeta_p(r) = local log-slope vs r
- ESS plots (compensated + exponent ratios)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cmasher as cmr

ORDER_CMAP = cmr.ember

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import structure_functions_fft
from scalar_advection.fitting import best_powerlaw_fit
from scalar_advection.spectra import kinetic_energy_spectrum
from scalar_advection.velocity import generate_divfree_field

# Parameters (scaled down from the 1024 campaign for faster iteration)
N = 4096
LAM_MIN = 2
LAM_MAX = 2048
WAVELET = "mexh"
SEED = 14
ALPHAS = [1 / 3, 2/3]#, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
ORDERS = (1, 2, 3, 4, 6, 8, 10)

# Match the original script’s binning/sampling defaults
N_ELL_BINS = 96
N_DISP_TOTAL = 6400

# Velocity spectrum binning
N_K_BINS = 96

OUTPUT_DIR = REPO_ROOT / "velocity_structure_functions_fft_testing"

# Fraction labels for display
ALPHA_LABELS = {
    1 / 6: r"1/6",
    1 / 3: r"1/3",
    1 / 2: r"1/2",
    2 / 3: r"2/3",
    5 / 6: r"5/6",
}


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local log-log slope using a sliding window."""
    mask = (r > 0) & (y > 0)
    r = r[mask]
    y = y[mask]

    if len(r) < window:
        return np.array([]), np.array([])

    log_r = np.log(r)
    log_y = np.log(np.maximum(y, 1e-30))

    slopes = []
    centers = []
    for i in range(len(r) - window + 1):
        idx = slice(i, i + window)
        m, _ = np.polyfit(log_r[idx], log_y[idx], 1)
        slopes.append(m)
        centers.append(np.exp(np.mean(log_r[idx])))

    return np.array(centers), np.array(slopes)


def plot_velocity_sf(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    alpha: float,
    fname: Path,
) -> None:
    """
    Plot velocity structure functions with zeta subpanel.

    Top panel: (S_p)^{1/p} vs r
    Bottom panel: zeta_p(r) = d log(S_p^{1/p}) / d log r
    """
    fig, (ax_main, ax_zeta) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
    )

    n_orders = len(orders)
    colors = ORDER_CMAP(np.linspace(0.15, 0.85, n_orders))

    # Fit range (inertial range)
    fit_lo = LAM_MIN
    fit_hi = LAM_MAX

    alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / p)
        color = colors[j]

        # Main curve
        ax_main.loglog(r, y, "o-", ms=4, lw=1.8, color=color, label=rf"$p={int(p)}$")

        # Power-law fit
        fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.5, x_range=(fit_lo, fit_hi))
        if fit is not None:
            ax_main.loglog(fit.xseg, fit.yfit, "--", lw=2.0, color=color, alpha=0.6)

        # Local slope for zeta subpanel
        centers, slopes = sliding_log_slope(r, y, window=8)
        if len(centers) > 0:
            ax_zeta.semilogx(centers, slopes, "-", lw=1.4, color=color)

    # Reference line at alpha (expected scaling)
    ax_zeta.axhline(alpha, color="k", ls="--", lw=1.5, alpha=0.7, label=rf"$\alpha = {alpha_label}$")

    # Mark fit region
    ax_main.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.5)
    ax_main.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.5)
    ax_zeta.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.5)
    ax_zeta.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.5)

    ax_main.set_ylabel(r"$(S_p)^{1/p}$", fontsize=13)
    ax_main.legend(frameon=False, fontsize=10, ncol=2, loc="lower right")
    ax_main.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)

    ax_zeta.set_xlabel(r"separation $r$ [pixels]", fontsize=13)
    ax_zeta.set_ylabel(r"$\zeta_p(r)$", fontsize=13)
    ax_zeta.legend(frameon=False, fontsize=10, loc="upper right")
    ax_zeta.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax_zeta.set_ylim(0, 1.2)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_field_map(
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    alpha: float,
    fname: Path,
    quiver_stride: int | None = None,
) -> None:
    """Plot a map of the velocity field: speed magnitude + downsampled quiver."""
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError(f"Expected ux, uy to be same-shaped 2D arrays, got {ux.shape=} {uy.shape=}")

    N_local = int(ux.shape[0])
    speed = np.hypot(ux, uy)
    finite = np.isfinite(speed)
    vmax = float(np.percentile(speed[finite], 99.0)) if np.any(finite) else float(np.max(speed))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(speed)) if np.size(speed) else 1.0

    if quiver_stride is None:
        quiver_stride = max(1, N_local // 32)
    quiver_stride = int(max(1, quiver_stride))

    alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=160, constrained_layout=True)
    im = ax.imshow(speed, origin="lower", cmap=cmr.neutral, vmin=0.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(rf"$|\mathbf{{u}}|$ map: $\alpha={alpha_label}$, $N={N_local}$", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")

    ys = np.arange(0, N_local, quiver_stride)
    xs = np.arange(0, N_local, quiver_stride)
    X, Y = np.meshgrid(xs, ys)
    u_sub = ux[::quiver_stride, ::quiver_stride]
    v_sub = uy[::quiver_stride, ::quiver_stride]

    spd_sub = np.hypot(u_sub, v_sub)
    finite_sub = np.isfinite(spd_sub)
    ref = float(np.percentile(spd_sub[finite_sub], 90.0)) if np.any(finite_sub) else float(np.max(spd_sub))
    target_len = 0.7 * float(quiver_stride)
    scale_fac = (target_len / (ref + 1e-12)) if (np.isfinite(ref) and ref > 0) else 1.0

    ax.quiver(
        X,
        Y,
        u_sub * scale_fac,
        v_sub * scale_fac,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="k",
        alpha=0.65,
        width=0.0022,
        headwidth=3.0,
        headlength=4.2,
        headaxislength=3.6,
    )

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_energy_spectrum(
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    alpha: float,
    fname: Path,
    n_k_bins: int = N_K_BINS,
) -> dict[str, float] | None:
    """Plot isotropized kinetic energy spectrum E(k) with power-law fits in the LAM_MIN..LAM_MAX band."""
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError(f"Expected ux, uy to be same-shaped 2D arrays, got {ux.shape=} {uy.shape=}")

    n = int(ux.shape[0])
    spec = kinetic_energy_spectrum(ux, uy, n_bins=int(n_k_bins))
    edges = np.asarray(spec["edges"], dtype=float)
    E = np.asarray(spec["E"], dtype=float)
    k = np.sqrt(edges[:-1] * edges[1:])

    # Convert wavelength band [LAM_MIN, LAM_MAX] to wavenumber-index band [n/LAM_MAX, n/LAM_MIN]
    k_fit_lo = n / float(LAM_MAX)
    k_fit_hi = n / float(LAM_MIN)
    if k_fit_lo > k_fit_hi:
        k_fit_lo, k_fit_hi = k_fit_hi, k_fit_lo

    fit_mask = (k >= k_fit_lo) & (k <= k_fit_hi) & (E > 0) & np.isfinite(k) & np.isfinite(E)
    n_fit = int(np.sum(fit_mask))

    alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160)
    ax.loglog(k, E, lw=1.8, alpha=0.9, label=r"$E(k)$")

    results: dict[str, float] = {}
    if n_fit >= 6:
        k_fit = k[fit_mask]
        E_fit = E[fit_mask]
        logk = np.log10(k_fit)
        logE = np.log10(E_fit)

        # Unconstrained least-squares fit over the full k-band.
        m_free, c_free = np.polyfit(logk, logE, 1)
        A_free = float(10**c_free)
        results["m_free"] = float(m_free)
        results["A_free"] = A_free

        pred = m_free * logk + c_free
        ss_res = float(np.sum((logE - pred) ** 2))
        ss_tot = float(np.sum((logE - float(np.mean(logE))) ** 2))
        results["r2_free"] = float(1.0 - ss_res / (ss_tot + 1e-30))

        # Fixed-slope fit: m = -(2*alpha + 1), only optimize normalization A.
        m_fixed = -(2.0 * float(alpha) + 1.0)
        c_fixed = float(np.mean(logE - m_fixed * logk))
        A_fixed = float(10**c_fixed)
        results["m_fixed"] = float(m_fixed)
        results["A_fixed"] = A_fixed

        pred_fixed = m_fixed * logk + c_fixed
        ss_res_fixed = float(np.sum((logE - pred_fixed) ** 2))
        results["r2_fixed"] = float(1.0 - ss_res_fixed / (ss_tot + 1e-30))

        k_line = np.geomspace(k_fit_lo, k_fit_hi, 200)
        ax.loglog(
            k_line,
            A_free * (k_line**m_free),
            "k--",
            lw=2.4,
            alpha=0.75,
            label=fr"free fit: $m={m_free:.3f}$",
        )
        ax.loglog(
            k_line,
            A_fixed * (k_line**m_fixed),
            color="tab:red",
            ls="--",
            lw=2.0,
            alpha=0.8,
            label=fr"fixed $m=-(2\alpha+1)={m_fixed:.3f}$",
        )

    ax.axvline(k_fit_lo, color="0.5", ls=":", lw=1.2, alpha=0.7)
    ax.axvline(k_fit_hi, color="0.5", ls=":", lw=1.2, alpha=0.7)

    ax.set_title(rf"Velocity energy spectrum: $\alpha={alpha_label}$, $N={n}$", fontsize=13)
    ax.set_xlabel(r"$k$   (mode number; $k{=}1\equiv 2\pi/L$)", fontsize=12)
    ax.set_ylabel(r"$E(k)$", fontsize=12)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return results or None


def _choose_ess_ref_order(orders: np.ndarray) -> int | None:
    """Prefer p=3, fall back to p=2 if p=3 is unavailable."""
    vals = set(int(p) for p in np.asarray(orders).astype(int))
    if 3 in vals:
        return 3
    if 2 in vals:
        return 2
    return None


def fit_ess_slopes(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    *,
    p_ref: int,
    fit_lo: float,
    fit_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ESS slopes: log10 S_p vs log10 S_pref over a restricted r window.

    Returns:
        slopes: slope for each p (NaN if insufficient data)
        slope_err: standard error of slope (NaN if unavailable)
        r2: R^2 of the log-log fit (NaN if unavailable)
    """
    orders_int = np.asarray(orders).astype(int)
    jref = np.where(orders_int == int(p_ref))[0]
    if not jref.size:
        raise ValueError(f"Reference order p_ref={p_ref} not present in orders={orders_int.tolist()}")
    jref = int(jref[0])

    Sref = np.asarray(S[jref], dtype=float)
    base_mask = (r >= fit_lo) & (r <= fit_hi) & (Sref > 0) & np.isfinite(Sref)

    slopes = np.full(len(orders_int), np.nan, dtype=float)
    slope_err = np.full(len(orders_int), np.nan, dtype=float)
    r2 = np.full(len(orders_int), np.nan, dtype=float)

    for j, _p in enumerate(orders_int):
        Sp = np.asarray(S[j], dtype=float)
        valid = base_mask & (Sp > 0) & np.isfinite(Sp)
        if np.sum(valid) < 4:
            continue

        x = np.log10(Sref[valid])
        y = np.log10(Sp[valid])
        m, c = np.polyfit(x, y, 1)
        slopes[j] = float(m)

        y_pred = m * x + c
        resid = y - y_pred
        dof = x.size - 2
        if dof > 0:
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            r2[j] = 1.0 - ss_res / (ss_tot + 1e-30)
            s2 = ss_res / dof
            sxx = float(np.sum((x - float(np.mean(x))) ** 2))
            if sxx > 0:
                slope_err[j] = float(np.sqrt(s2 / sxx))

    return slopes, slope_err, r2


def plot_ess_compensated(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    *,
    p_ref: int,
    slopes: np.ndarray,
    fit_lo: float,
    fit_hi: float,
    title: str,
    fname: Path,
) -> None:
    """
    Compensated ESS: plot S_p / S_pref^{zeta_p} vs S_pref, where zeta_p is the ESS slope.
    """
    orders_int = np.asarray(orders).astype(int)
    jref = int(np.where(orders_int == int(p_ref))[0][0])
    Sref = np.asarray(S[jref], dtype=float)
    base_mask = (r >= fit_lo) & (r <= fit_hi) & (Sref > 0) & np.isfinite(Sref)

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=160)
    colors = ORDER_CMAP(np.linspace(0.15, 0.85, len(orders_int)))
    comp_values: List[np.ndarray] = []

    for j, p in enumerate(orders_int):
        zeta = float(slopes[j])
        if not np.isfinite(zeta):
            continue
        Sp = np.asarray(S[j], dtype=float)
        valid = base_mask & (Sp > 0) & np.isfinite(Sp)
        if np.sum(valid) < 3:
            continue
        x = Sref[valid]
        comp = Sp[valid] / np.power(x, zeta)
        comp_values.append(comp)
        ax.loglog(x, comp, "D:", ms=4.0, lw=1.2, color=colors[j], alpha=0.85, label=rf"$p={p}$")

        med = float(np.median(comp))
        ax.hlines(med, xmin=float(x.min()), xmax=float(x.max()), colors=[colors[j]], linestyles=":", lw=1.0, alpha=0.5)

    if comp_values:
        all_comp = np.concatenate(comp_values)
        valid_comp = np.isfinite(all_comp) & (all_comp > 0)
        if np.any(valid_comp):
            logc = np.log10(all_comp[valid_comp])
            q_lo, q_hi = np.quantile(logc, [0.05, 0.95])
            span = float(q_hi - q_lo)
            pad = max(0.15, 0.05 * span)  # decades
            y_lo = float(10 ** (q_lo - pad))
            y_hi = float(10 ** (q_hi + pad))
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo > 0 and y_hi > y_lo:
                ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(rf"$S_{{{p_ref}}}$", fontsize=12)
    ax.set_ylabel(rf"$S_p / S_{{{p_ref}}}^{{\zeta_p}}$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_ess_exponent_ratios(
    orders: np.ndarray,
    *,
    p_ref: int,
    slopes: np.ndarray,
    slope_err: np.ndarray,
    title: str,
    fname: Path,
) -> None:
    """
    Plot normalized ESS exponent ratios:
      zeta_p / (p/p_ref)
    where zeta_p is the fitted ESS slope (log S_p vs log S_pref).
    """
    orders_int = np.asarray(orders).astype(int)
    p_over = orders_int.astype(float) / float(p_ref)
    ratio = slopes / p_over
    ratio_err = slope_err / p_over

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=160)

    valid = np.isfinite(ratio)
    if np.any(valid):
        ax.errorbar(
            orders_int[valid],
            ratio[valid],
            yerr=ratio_err[valid],
            fmt="o-",
            ms=5,
            lw=1.8,
            capsize=3,
            color="k",
        )

    ax.axhline(1.0, color="0.4", ls=":", lw=1.4, label=rf"$\zeta_p = p/{p_ref}$")
    ax.set_xlabel(r"$p$", fontsize=12)
    ax.set_ylabel(rf"$\zeta_p / (p/{p_ref})$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)
    ax.set_xlim(0.5, float(np.max(orders_int)) + 0.5)
    ax.legend(frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing velocity structure functions for N={N} campaign (FFT-based)")
    print(f"Parameters: lam_min={LAM_MIN}, lam_max={LAM_MAX}, wavelet={WAVELET}, seed={SEED}")
    print(f"Alphas: {[ALPHA_LABELS[a] for a in ALPHAS]}")
    print(f"Orders: {ORDERS}")
    print(f"Bins: n_ell_bins={N_ELL_BINS}, n_k_bins={N_K_BINS}, odd-order samples n_disp_total={N_DISP_TOTAL}")
    print()

    for alpha in ALPHAS:
        alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")
        print(f"Processing alpha = {alpha_label}...")
        alpha_str = f"{alpha:.4f}".replace(".", "p")

        # Generate velocity field
        ux, uy, _ = generate_divfree_field(
            N=N,
            lam_min=LAM_MIN,
            lam_max=LAM_MAX,
            alpha=alpha,
            wavelet=WAVELET,
            sparsity=0.0,
            seed=SEED,
            scales_per_octave=30,     # 4–8 is a good range
            taper_frac=0.0,         # 0.1–0.2 usually works well
        )
        ux = ux.astype(np.float32, copy=False)
        uy = uy.astype(np.float32, copy=False)

        vel_map_path = OUTPUT_DIR / f"velocity_field_alpha_{alpha_str}.png"
        plot_velocity_field_map(ux, uy, alpha=alpha, fname=vel_map_path)
        print(f"  Saved: {vel_map_path}")

        spec_path = OUTPUT_DIR / f"velocity_spectrum_alpha_{alpha_str}.png"
        spec_fit = plot_velocity_energy_spectrum(ux, uy, alpha=alpha, fname=spec_path)
        print(f"  Saved: {spec_path}")
        if spec_fit is not None:
            m_free = spec_fit.get("m_free")
            r2_free = spec_fit.get("r2_free")
            m_fixed = spec_fit.get("m_fixed")
            A_fixed = spec_fit.get("A_fixed")
            r2_fixed = spec_fit.get("r2_fixed")
            print(f"  Spectrum fit (free): m={m_free:.4f}, R^2={r2_free:.4f} over lambda=[{LAM_MIN}, {LAM_MAX}]")
            print(f"  Spectrum fit (fixed): m={m_fixed:.4f}, A={A_fixed:.4e}, R^2={r2_fixed:.4f}")

        # Compute per-component structure functions via FFT method, then average.
        sf_x = structure_functions_fft(
            ux,
            orders=ORDERS,
            n_ell_bins=N_ELL_BINS,
            n_disp_total=N_DISP_TOTAL,
            seed=SEED,
            pad=False,
        )
        sf_y = structure_functions_fft(
            uy,
            orders=ORDERS,
            n_ell_bins=N_ELL_BINS,
            n_disp_total=N_DISP_TOTAL,
            seed=SEED,
            pad=False,
        )

        if not np.all(sf_x["ell_edges"] == sf_y["ell_edges"]):
            raise RuntimeError("Component bin edges disagree; cannot combine results.")

        orders_arr = sf_x["orders"]
        r = sf_x["r"]
        S = 0.5 * (sf_x["S"] + sf_y["S"])

        # Plot
        fname = OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.png"
        plot_velocity_sf(r, S, orders_arr, alpha, fname)
        print(f"  Saved: {fname}")

        # ESS plots (use the same inertial-range window as the main plot)
        fit_lo = 8.0
        fit_hi = N / 4.0
        p_ref = _choose_ess_ref_order(orders_arr)
        if p_ref is None:
            print("  [warn] ESS skipped: need p=3 or p=2 in ORDERS")
            ess_slopes = np.full_like(orders_arr, np.nan, dtype=float)
            ess_slope_err = np.full_like(orders_arr, np.nan, dtype=float)
            ess_r2 = np.full_like(orders_arr, np.nan, dtype=float)
        else:
            ess_slopes, ess_slope_err, ess_r2 = fit_ess_slopes(
                r,
                S,
                orders_arr,
                p_ref=p_ref,
                fit_lo=fit_lo,
                fit_hi=fit_hi,
            )
            ess_comp_path = OUTPUT_DIR / f"velocity_ess_comp_alpha_{alpha_str}.png"
            ess_ratio_path = OUTPUT_DIR / f"velocity_ess_ratio_alpha_{alpha_str}.png"
            ess_title = rf"Compensated ESS (velocity): $\alpha={alpha_label}$, N={N} [FFT]"
            plot_ess_compensated(
                r,
                S,
                orders_arr,
                p_ref=p_ref,
                slopes=ess_slopes,
                fit_lo=fit_lo,
                fit_hi=fit_hi,
                title=ess_title,
                fname=ess_comp_path,
            )
            ratio_title = rf"ESS exponent ratios (velocity): $\alpha={alpha_label}$, N={N} [FFT]"
            plot_ess_exponent_ratios(
                orders_arr,
                p_ref=p_ref,
                slopes=ess_slopes,
                slope_err=ess_slope_err,
                title=ratio_title,
                fname=ess_ratio_path,
            )
            print(f"  Saved: {ess_comp_path}")
            print(f"  Saved: {ess_ratio_path}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
