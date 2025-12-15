#!/usr/bin/env python3
"""
Compute and plot velocity structure functions for the 1024 campaign.

Generates velocity fields with the same parameters as the alpha-kappa 1024 campaign:
- N=1024, lam_min=2, lam_max=1024, wavelet=mexh, seed=1000
- alphas = [1/6, 1/3, 1/2, 2/3, 5/6]
- orders = (1, 2, 3, 4, 6, 8, 10)

Produces per-alpha plots:
- Top panel: (S_p)^{1/p} vs r with power-law fits
- Bottom panel: zeta_p(r) = local log-slope vs r
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

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

from scalar_advection.velocity import generate_divfree_field
from scalar_advection.structure import structure_functions, s2_fft_vector
from scalar_advection.fitting import best_powerlaw_fit

# Parameters matching the 1024 campaign
N = 1024
LAM_MIN = 2.0
LAM_MAX = 1024.0
WAVELET = "mexh"
SEED = 1000
ALPHAS = [1/6, 1/3, 1/2, 2/3, 5/6]
ORDERS = (1, 2, 3, 4, 6, 8, 10)

OUTPUT_DIR = REPO_ROOT / "campaigns" / "alpha_kappa_1024" / "velocity_structure_functions"

# Fraction labels for display
ALPHA_LABELS = {
    1/6: r"1/6",
    1/3: r"1/3",
    1/2: r"1/2",
    2/3: r"2/3",
    5/6: r"5/6",
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
        2, 1, figsize=(9, 7), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05}
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
    ax_main.set_title(rf"Velocity Structure Functions ($\alpha = {alpha_label}$, N=1024)", fontsize=14)
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing velocity structure functions for N={N} campaign")
    print(f"Parameters: lam_min={LAM_MIN}, lam_max={LAM_MAX}, wavelet={WAVELET}, seed={SEED}")
    print(f"Alphas: {[ALPHA_LABELS[a] for a in ALPHAS]}")
    print(f"Orders: {ORDERS}")
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

        # Compute structure functions
        sf = structure_functions(
            (ux, uy),
            orders=ORDERS,
            n_ell_bins=50,
            n_disp_total=8192,
            seed=SEED,
            use_fft_for_p2=True,
            signed_longitudinal=False,
        )

        # Use FFT-based S2 for better accuracy
        orders_arr = sf["orders"]
        j2 = np.where(np.isclose(orders_arr, 2.0))[0]
        if j2.size:
            r2, S2_fft = s2_fft_vector(ux, uy, sf["ell_edges"])
            sf["mag"][j2[0], :] = S2_fft

        r = sf["r"]
        S = sf["mag"]

        # Plot
        alpha_str = f"{alpha:.4f}".replace(".", "p")
        fname = OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.png"
        plot_velocity_sf(r, S, orders_arr, alpha, fname)
        print(f"  Saved: {fname}")

        # Save raw data
        np.savez_compressed(
            OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.npz",
            r=r,
            S=S,
            orders=orders_arr,
            alpha=alpha,
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
