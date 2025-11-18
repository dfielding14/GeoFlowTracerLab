#!/usr/bin/env python3
"""
Compute and plot velocity structure functions for the synthetic velocity field
used in the edot_vs_mu scaling experiments.

Defaults mirror experiments/edot_vs_mu_scaling.py:
- N=1024, wavelet=mexh, lam_min=8, lam_max=512, seed=1
- alphas = [1/3, 1/2, 2/3]
- orders = (1, 2, 3, 4, 6, 8, 10)

Produces per-alpha plots in the same style as theta structure functions:
top panel: (S_p)^{1/p} with power-law fit; bottom: local log-slope vs scale.
Also saves raw results to .npz.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import ScalarAdvectionAPI  # noqa: E402
from scalar_advection.velocity import generate_divfree_field  # noqa: E402
from scalar_advection.structure import structure_functions, s2_fft_vector  # noqa: E402
from scalar_advection.fitting import best_powerlaw_fit  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Velocity structure functions for wavelet velocity fields")
    p.add_argument("--N", type=int, default=1024, help="Grid size (default: 1024)")
    p.add_argument("--alpha", type=float, nargs="+", default=[1.0/3.0, 1.0/2.0, 2.0/3.0])
    p.add_argument("--lam-min", type=float, default=8.0)
    p.add_argument("--lam-max", type=float, default=512.0)
    p.add_argument("--wavelet", choices=("mexh", "haar"), default="mexh")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10])
    p.add_argument("--n-ell-bins", type=int, default=40)
    p.add_argument("--n-disp-total", type=int, default=4096)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--output-root", type=Path, default=Path("experimental_results") / "velocity_structure_functions")
    return p.parse_args(argv)


def sliding_log_slope(rvals: np.ndarray, yvals: np.ndarray, window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if rvals.size < window:
        return np.array([]), np.array([])
    lr = np.log(rvals)
    ly = np.log(np.maximum(yvals, 1e-30))
    slopes, centers = [], []
    for i in range(rvals.size - window + 1):
        idx = slice(i, i + window)
        m, c = np.polyfit(lr[idx], ly[idx], 1)
        slopes.append(m)
        centers.append(np.exp(np.mean(lr[idx])))
    return np.array(centers), np.array(slopes)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    N = int(args.N)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    out_dir = args.output_root / f"N{N}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dummy API just to adopt dtype/grid if needed downstream
    _ = ScalarAdvectionAPI(N=N, L=1.0, dtype=dtype, warm_cache=False)

    # Combined overlay fig
    fig_all, (axA_top, axA_bot) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), dpi=160, sharex=True,
        constrained_layout=True, gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05}
    )
    colors = plt.cm.tab10.colors

    for j, a in enumerate(args.alpha):
        # Build velocity
        ux, uy, _ = generate_divfree_field(
            N=N,
            lam_min=float(args.lam_min),
            lam_max=float(args.lam_max),
            alpha=float(a),
            wavelet=args.wavelet,
            sparsity=0.0,
            seed=int(args.seed),
        )
        ux = ux.astype(dtype, copy=False)
        uy = uy.astype(dtype, copy=False)

        # Velocity structure functions (magnitude)
        sf = structure_functions(
            (ux, uy),
            orders=tuple(args.orders),
            n_ell_bins=int(args.n_ell_bins),
            n_disp_total=int(args.n_disp_total),
            seed=int(args.seed),
            use_fft_for_p2=True,
            signed_longitudinal=False,
        )
        # Force p=2 magnitude to use FFT-based estimator explicitly
        orders_arr = sf["orders"]
        j2 = np.where(np.isclose(orders_arr, 2.0))[0]
        if j2.size:
            r2, S2_fft = s2_fft_vector(ux, uy, sf["ell_edges"])
            sf["mag"][j2, :] = S2_fft

        # Save raw results
        np.savez_compressed(out_dir / f"velocity_sf_alpha_{a:.3f}.npz", **sf)

        r = sf["r"]
        orders = sf["orders"]
        Smag = sf["mag"]  # shape: [n_orders, n_bins]
        root_curves = np.array([np.power(np.maximum(Smag[k], 1e-30), 1.0 / orders[k]) for k in range(len(orders))])

        # Fit range
        fit_lo = max(16.0, N / 128.0)
        fit_hi = N / 4.0

        # Per-alpha figure
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.2, 6.0), dpi=160, sharex=True,
            constrained_layout=True, gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05}
        )
        col = colors[j % len(colors)]
        for k, p in enumerate(orders):
            y = root_curves[k]
            ax_top.loglog(r, y, 'o-', ms=3, lw=1.6, label=f"p={p:g}")
            fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.5, x_range=(fit_lo, fit_hi))
            if fit is not None:
                ax_top.loglog(fit.xseg, fit.yfit, '--', lw=2.0, alpha=0.8)
            cen, sl = sliding_log_slope(r, y, window=4)
            if cen.size:
                ax_bot.semilogx(cen, sl, '-', lw=1.4)
        ax_top.set_ylabel(r"$(S_p)^{1/p}$")
        ax_top.grid(True, which='both', ls=':', lw=0.6)
        ax_top.legend(frameon=False, ncol=1, loc='lower right')
        ax_bot.set_xlabel(r"$\ell/\Delta x$")
        ax_bot.set_ylabel(r"$d\log(S_p^{1/p})/d\log\ell$")
        ax_bot.grid(True, which='both', ls=':', lw=0.6)
        fig.savefig(out_dir / f"velocity_sf_alpha_{a:.3f}.png", bbox_inches='tight')
        plt.close(fig)

        # Add to combined overlay (use p=2 curve to declutter)
        # Find index for p=2; fall back to p=1 if not present
        try:
            idx2 = list(orders).index(2)
        except ValueError:
            idx2 = 0
        y2 = root_curves[idx2]
        axA_top.loglog(r, y2, '-', lw=1.8, color=col, label=f"alpha={a:g}")
        cen, sl = sliding_log_slope(r, y2, window=4)
        if cen.size:
            axA_bot.semilogx(cen, sl, '-', lw=1.6, color=col)

    axA_top.set_ylabel(r"$(S_2)^{1/2}$")
    axA_top.grid(True, which='both', ls=':', lw=0.6)
    axA_top.legend(frameon=False)
    axA_bot.set_xlabel(r"$\ell/\Delta x$")
    axA_bot.set_ylabel(r"$d\log (S_2^{1/2}) / d\log\ell$")
    axA_bot.grid(True, which='both', ls=':', lw=0.6)
    fig_all.savefig(out_dir / "velocity_sf_overlay.png", bbox_inches='tight')
    plt.close(fig_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
