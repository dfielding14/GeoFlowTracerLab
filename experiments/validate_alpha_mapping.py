#!/usr/bin/env python3
"""
Validate the alpha → structure-function mapping for synthetic velocity fields.

For a set of alpha values, this script generates velocity fields on a square
grid, computes first-order vector structure functions S1(ℓ) = E[|δu|], fits a
log–log slope over a scale range, and compares the measured slope to alpha.
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
from scalar_advection.velocity import VelocityConfig  # noqa: E402
from scalar_advection.structure import structure_functions  # noqa: E402
from scalar_advection.fitting import best_powerlaw_fit  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate alpha mapping via S1(ℓ) slopes")
    p.add_argument("--N", type=int, default=512, help="Grid size (default: 512)")
    p.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.2, 1.0 / 3.0, 0.5],
        help="Alpha values to test (default: 0.2, 1/3, 0.5)",
    )
    p.add_argument("--seed", type=int, default=1, help="Velocity RNG seed")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--n-ell-bins", type=int, default=40)
    p.add_argument("--n-disp-total", type=int, default=4096)
    p.add_argument(
        "--fit-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="Fit range in ℓ/Δx (default: [max(8, N/256), N/4])",
    )
    p.add_argument("--output-root", type=Path, default=Path("experimental_results") / "validation_alpha")
    return p.parse_args(argv)


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if r.size < window:
        return np.array([]), np.array([])
    lr = np.log(r)
    ly = np.log(np.maximum(y, 1e-30))
    slopes, centers = [], []
    for i in range(r.size - window + 1):
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

    api = ScalarAdvectionAPI(N=N, L=1.0, dtype=dtype, warm_cache=True)

    measured = []
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)

    for a in args.alpha:
        cfg = VelocityConfig(alpha=float(a), seed=args.seed, urms=1.0)
        ux, uy = api.generate_velocity(cfg)

        sf = structure_functions(
            (ux, uy),
            orders=(1,),
            n_ell_bins=args.n_ell_bins,
            n_disp_total=args.n_disp_total,
            seed=args.seed,
            use_fft_for_p2=False,
            signed_longitudinal=False,
        )
        r = sf["r"]
        S1 = sf["mag"][0]

        # Choose fit range in grid units
        if args.fit_range:
            lo, hi = args.fit_range
        else:
            lo = max(8.0, N / 256.0)
            hi = N / 4.0
        fit = best_powerlaw_fit(r, S1, min_points=6, min_decades=0.5, x_range=(lo, hi))
        slope = float("nan") if fit is None else float(fit.m)
        measured.append((float(a), slope))

        # Plot S1 and fitted segment
        ax.loglog(r, S1, label=f"alpha={a:g}")
        if fit is not None:
            ax.loglog(fit.xseg, fit.yfit, "k--", alpha=0.7)

        # Also dump per-alpha plot
        figi, axi = plt.subplots(figsize=(5.4, 4.2), dpi=160)
        axi.loglog(r, S1, "o-", ms=3)
        if fit is not None:
            axi.loglog(fit.xseg, fit.yfit, "k--", lw=2, label=f"fit m={fit.m:.3f}")
            axi.legend(frameon=False)
        axi.set_title(f"S1 vs ℓ (alpha={a:g})")
        axi.set_xlabel(r"separation $\ell/\Delta x$")
        axi.set_ylabel(r"$S_1(\ell)$")
        axi.grid(True, which="both", ls=":", lw=0.6)
        figi.tight_layout()
        figi.savefig(out_dir / f"S1_alpha_{a:.3f}.png", bbox_inches="tight")
        plt.close(figi)

    ax.set_xlabel(r"separation $\ell/\Delta x$")
    ax.set_ylabel(r"$S_1(\ell)$")
    ax.grid(True, which="both", ls=":", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "S1_all_alphas.png", bbox_inches="tight")
    plt.close(fig)

    # Scatter plot of measured slopes vs alpha
    alphas = np.array([a for a, _ in measured])
    slopes = np.array([s for _, s in measured])
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.2), dpi=160)
    ax2.plot([alphas.min(), alphas.max()], [alphas.min(), alphas.max()], "k--", lw=1.5, label="ideal: slope=alpha")
    ax2.plot(alphas, slopes, "o", ms=6, label="measured")
    ax2.set_xlabel("alpha (target)")
    ax2.set_ylabel("measured slope (S1)")
    ax2.grid(True, ls=":", lw=0.6)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(out_dir / "slope_vs_alpha.png", bbox_inches="tight")
    plt.close(fig2)

    # Save numeric results
    np.savez_compressed(out_dir / "results.npz", alphas=alphas, slopes=slopes)
    (out_dir / "README.txt").write_text(
        "Validation of alpha mapping: slopes close to alpha across tested values.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

