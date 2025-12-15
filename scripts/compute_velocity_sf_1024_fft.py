#!/usr/bin/env python3
"""
Compute and plot velocity structure functions for the 1024 campaign (FFT-based).

This is a drop-in companion to `scripts/compute_velocity_sf_1024.py`, but it uses
`scalar_advection.structure_functions_fft` for the structure functions.

Notes
-----
- `structure_functions_fft` is implemented for 2D *scalar* fields. For a 2D
  velocity field (ux, uy), we compute per-component structure functions and
  then average:  S_p = 0.5 * (S_p[ux] + S_p[uy]).
- Even orders are computed exactly via FFT/binomial correlations.
- Odd orders use an absolute-increment displacement sampler under the hood.

Generates velocity fields with the same parameters as the alpha-kappa 1024 campaign:
- N=1024, lam_min=2, lam_max=1024, wavelet=mexh, seed=1000
- alphas = [1/6, 1/3, 1/2, 2/3, 5/6]
- orders = (1, 2, 3, 4, 6, 8, 10)

Produces per-alpha plots:
- Top panel: (S_p)^{1/p} vs r with power-law fits
- Bottom panel: zeta_p(r) = local log-slope vs r
- ESS plots (compensated + exponent ratios)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cmasher as cmr

    ORDER_CMAP = cmr.ember
except ImportError:
    from matplotlib import cm as mpl_cm

    ORDER_CMAP = mpl_cm.viridis

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import structure_functions_fft
from scalar_advection.fitting import best_powerlaw_fit
from scalar_advection.velocity import generate_divfree_field

# Parameters matching the 1024 campaign
N = 1024
LAM_MIN = 2.0
LAM_MAX = 1024.0
WAVELET = "mexh"
SEED = 1000
ALPHAS = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
ORDERS = (1, 2, 3, 4, 6, 8, 10)

# Match the original script’s binning/sampling defaults
N_ELL_BINS = 50
N_DISP_TOTAL = 8192

OUTPUT_DIR = REPO_ROOT / "campaigns" / "alpha_kappa_1024" / "velocity_structure_functions_fft"

# Fraction labels for display
ALPHA_LABELS = {
    1 / 6: r"1/6",
    1 / 3: r"1/3",
    1 / 2: r"1/2",
    2 / 3: r"2/3",
    5 / 6: r"5/6",
}


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
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
    fit_lo = 8.0
    fit_hi = N / 4.0

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
    ax_main.set_title(rf"Velocity Structure Functions ($\alpha = {alpha_label}$, N=1024) [FFT]", fontsize=14)
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
    print(f"Bins: n_ell_bins={N_ELL_BINS}, odd-order samples n_disp_total={N_DISP_TOTAL}")
    print()

    for alpha in ALPHAS:
        alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")
        print(f"Processing alpha = {alpha_label}...")

        # Generate velocity field
        ux, uy, _ = generate_divfree_field(
            N=N,
            lam_min=LAM_MIN,
            lam_max=LAM_MAX,
            alpha=alpha,
            wavelet=WAVELET,
            sparsity=0.0,
            seed=SEED,
        )
        ux = ux.astype(np.float32, copy=False)
        uy = uy.astype(np.float32, copy=False)

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
        alpha_str = f"{alpha:.4f}".replace(".", "p")
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
            ess_title = rf"Compensated ESS (velocity): $\alpha={alpha_label}$, N=1024 [FFT]"
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
            ratio_title = rf"ESS exponent ratios (velocity): $\alpha={alpha_label}$, N=1024 [FFT]"
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

        # Save raw data
        np.savez_compressed(
            OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.npz",
            r=r,
            S=S,
            orders=orders_arr,
            alpha=alpha,
            component_Sx=sf_x["S"],
            component_Sy=sf_y["S"],
            ell_edges=sf_x["ell_edges"],
            fit_lo=fit_lo,
            fit_hi=fit_hi,
            ess_ref_order=(int(p_ref) if p_ref is not None else -1),
            ess_slopes=ess_slopes,
            ess_slope_err=ess_slope_err,
            ess_r2=ess_r2,
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
