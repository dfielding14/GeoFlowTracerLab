#!/usr/bin/env python3
"""
Comprehensive theta structure function analysis for the 2048 alpha-kappa campaign.

Produces per-simulation:
1. Structure Functions Plot (S_p vs r) - log-log, p=1..10, Taylor microscale marked
2. Zeta sub-panels - zeta_p(r) vs r for all orders (scale-dependent slopes)
3. ESS Plot - log S_p(r) vs log S_3(r) for p=1..10
4. Power Spectrum Plot - E_theta(k) vs k (log-log)
5. Taylor Microscale - from S2 curvature + gradient method cross-check
6. Integral Scale - from correlation function / S2
7. Best-Fit Exponents - zeta_p for p=1..10, fitted between lambda_T and min(L_int, L_box/4)
8. zeta_p vs p Plot - multipanel (one per alpha), kappas as different colors
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    import cmasher as cmr

    ALPHA_CMAP = cmr.ember
    KAPPA_CMAP = cmr.ocean
except ImportError:
    from matplotlib import cm as mpl_cm

    ALPHA_CMAP = mpl_cm.viridis
    KAPPA_CMAP = mpl_cm.plasma

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection.structure import structure_functions, s2_fft_scalar
from scalar_advection.spectra import scalar_power_spectrum, plot_scalar_spectrum
from scalar_advection.grid import SpectralGrid
from scalar_advection.fitting import best_powerlaw_fit
from scalar_advection.fft import fft2, ifft2

# Path to 2048 campaign results (using symlinked path for consistency)
RESULTS_ROOT = Path("/mnt/home/dfielding/ceph/GeoFlowTracerLab/experimental_results/alpha_kappa_2048")
OUTPUT_DIR = Path("/mnt/home/dfielding/ceph/GeoFlowTracerLab/campaigns/alpha_kappa_2048/theta_structure_functions")
# All orders from 1 to 10 as requested
ORDERS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


@dataclass
class ScaleInfo:
    """Container for computed length scales."""

    lambda_T_gradient: float  # Taylor microscale from variance/gradient
    lambda_T_S2: float  # Taylor microscale from S2 curvature
    L_int: float  # Integral scale from correlation function
    fit_min: float  # Lower bound for power-law fit
    fit_max: float  # Upper bound for power-law fit


@dataclass
class FitExponents:
    """Container for fitted scaling exponents."""

    orders: np.ndarray  # p values
    zeta_p: np.ndarray  # fitted exponents
    zeta_p_err: np.ndarray  # fit uncertainties (placeholder)
    fit_r2: np.ndarray  # R^2 values for fits


def load_final_theta(run_dir: Path) -> Optional[np.ndarray]:
    """Load the final theta snapshot from a run."""
    final_path = run_dir / "fields" / "theta_final.npz"
    if final_path.exists():
        return np.load(final_path)["theta"]

    snapshot_dir = run_dir / "fields" / "snapshots"
    if not snapshot_dir.exists():
        return None
    snapshots = sorted(snapshot_dir.glob("theta_*.npz"))
    if not snapshots:
        return None
    return np.load(snapshots[-1])["theta"]


def compute_taylor_microscale_gradient(theta: np.ndarray, L: float = 1.0) -> float:
    """
    Compute Taylor microscale from variance and gradient variance.

    lambda_T = sqrt(<theta^2> / <|grad theta|^2>)
    """
    N = theta.shape[0]
    dx = L / N

    # Compute gradients via FFT
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    theta_hat = fft2(theta.astype(np.float64))
    theta_x_hat = 1j * KX * theta_hat
    theta_y_hat = 1j * KY * theta_hat

    theta_x = np.real(ifft2(theta_x_hat))
    theta_y = np.real(ifft2(theta_y_hat))

    theta_var = float(np.mean(theta**2))
    grad_sq = float(np.mean(theta_x**2 + theta_y**2))

    if grad_sq < 1e-30:
        return np.nan

    lambda_T = np.sqrt(theta_var / grad_sq)
    return lambda_T


def compute_taylor_microscale_S2(r: np.ndarray, S2: np.ndarray) -> float:
    """
    Compute Taylor microscale from S2 curvature at small r.

    For r << lambda_T: S2(r) ~ r^2 / lambda_T^2
    So lambda_T = r / sqrt(S2(r)) at small r where S2 ~ r^2

    We fit S2 = A * r^2 in the small-r region and extract lambda_T = 1/sqrt(A).
    """
    # Use points where S2 shows r^2 scaling (typically first few bins)
    # Find where local slope is close to 2
    mask = (r > 0) & (S2 > 0)
    r_valid = r[mask]
    S2_valid = S2[mask]

    if len(r_valid) < 4:
        return np.nan

    # Fit in the first ~10 points or up to r where slope deviates from 2
    log_r = np.log(r_valid)
    log_S2 = np.log(S2_valid)

    # Compute local slopes
    slopes = np.diff(log_S2) / np.diff(log_r)

    # Find region where slope is close to 2 (within tolerance)
    tol = 0.5
    good_idx = np.where(np.abs(slopes - 2.0) < tol)[0]

    if len(good_idx) < 3:
        # Fallback: use first 5 points
        n_fit = min(5, len(r_valid))
    else:
        # Use the contiguous region starting from beginning
        n_fit = 1
        for i in range(len(good_idx) - 1):
            if good_idx[i + 1] == good_idx[i] + 1 and good_idx[i] == i:
                n_fit = good_idx[i + 1] + 2
            else:
                break
        n_fit = max(n_fit, 3)

    n_fit = min(n_fit, len(r_valid))

    # Fit S2 = A * r^2 (log S2 = log A + 2 log r)
    # Fix slope to 2, solve for intercept
    log_A = np.mean(log_S2[:n_fit] - 2.0 * log_r[:n_fit])
    A = np.exp(log_A)

    if A <= 0:
        return np.nan

    # S2 = A * r^2 = r^2 / lambda_T^2 => lambda_T = 1/sqrt(A)
    lambda_T = 1.0 / np.sqrt(A)
    return lambda_T


def compute_integral_scale(r: np.ndarray, S2: np.ndarray, theta_var: float) -> float:
    """
    Compute integral scale from the correlation function.

    C(r) = <theta(x) theta(x+r)> = variance - S2(r)/2
    L_int = integral_0^inf C(r)/C(0) dr

    We integrate until C(r) drops to near zero or becomes negative.
    """
    # Correlation function: C(r) = variance - S2/2
    C = theta_var - S2 / 2.0

    # Normalize
    C0 = C[0] if len(C) > 0 and C[0] > 0 else theta_var
    if C0 <= 0:
        return np.nan

    C_norm = C / C0

    # Find where correlation drops to ~0 or becomes negative
    valid = C_norm > 0.01
    if not np.any(valid):
        return np.nan

    r_valid = r[valid]
    C_valid = C_norm[valid]

    # Trapezoidal integration
    L_int = float(np.trapz(C_valid, r_valid))
    return max(L_int, 1.0)  # At least 1 pixel


def compute_scales(
    theta: np.ndarray, r: np.ndarray, S2: np.ndarray, L: float = 1.0, N: int = 2048
) -> ScaleInfo:
    """
    Compute all relevant length scales.

    Uses best_powerlaw_fit to automatically detect the inertial range from S2,
    which is more robust than trying to compute Taylor and integral scales
    from physical definitions that may give inconsistent results.
    """
    dx = L / N

    # Taylor microscale from gradient method (in physical units, then convert to pixels)
    lambda_T_grad_phys = compute_taylor_microscale_gradient(theta, L)
    lambda_T_grad_px = lambda_T_grad_phys / dx if np.isfinite(lambda_T_grad_phys) else np.nan

    # Taylor microscale from S2 (already in pixels since r is in pixels)
    lambda_T_S2_px = compute_taylor_microscale_S2(r, S2)

    # Variance for correlation function
    theta_var = float(np.mean(theta**2))

    # Integral scale (in pixels)
    L_int_px = compute_integral_scale(r, S2, theta_var)

    # Use best_powerlaw_fit to automatically detect the inertial range
    # This is more robust than relying on Taylor/integral scale calculations
    fit = best_powerlaw_fit(r, S2, min_points=8, min_decades=0.8)

    if fit is not None:
        # Use the automatically detected inertial range
        fit_min = float(fit.xseg[0])
        fit_max = float(fit.xseg[-1])
    else:
        # Fallback: use heuristic bounds
        # Taylor microscale estimate (use the smaller of the two if both valid)
        if np.isfinite(lambda_T_grad_px) and np.isfinite(lambda_T_S2_px):
            lambda_T_est = min(lambda_T_grad_px, lambda_T_S2_px)
        elif np.isfinite(lambda_T_S2_px):
            lambda_T_est = lambda_T_S2_px
        elif np.isfinite(lambda_T_grad_px):
            lambda_T_est = lambda_T_grad_px
        else:
            lambda_T_est = 4.0

        fit_min = max(lambda_T_est, 4.0)
        fit_max = min(L_int_px if np.isfinite(L_int_px) else N / 4, N / 4)

    # Sanity check: ensure fit_min < fit_max
    if fit_min >= fit_max:
        # If inertial range detection failed, use conservative defaults
        fit_min = max(4.0, r[r > 0].min() * 2)
        fit_max = min(N / 4, r.max() / 2)

    # Final safety: ensure at least some range
    if fit_min >= fit_max:
        fit_min = 4.0
        fit_max = N / 4

    return ScaleInfo(
        lambda_T_gradient=lambda_T_grad_px,
        lambda_T_S2=lambda_T_S2_px,
        L_int=L_int_px,
        fit_min=fit_min,
        fit_max=fit_max,
    )


def fit_scaling_exponents(
    r: np.ndarray, S: np.ndarray, orders: np.ndarray, fit_min: float, fit_max: float
) -> FitExponents:
    """
    Fit power-law exponents zeta_p for each order in the specified range.

    S_p(r) ~ r^{p * zeta_p} => (S_p)^{1/p} ~ r^{zeta_p}
    """
    n_orders = len(orders)
    zeta_p = np.full(n_orders, np.nan)
    zeta_p_err = np.full(n_orders, np.nan)
    fit_r2 = np.full(n_orders, np.nan)

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / p)
        fit = best_powerlaw_fit(
            r, y, min_points=6, min_decades=0.4, x_range=(fit_min, fit_max)
        )
        if fit is not None:
            zeta_p[j] = fit.m
            # Compute R^2 manually
            mask = (r >= fit_min) & (r <= fit_max) & (r > 0) & (y > 0)
            if np.sum(mask) > 2:
                log_r = np.log10(r[mask])
                log_y = np.log10(y[mask])
                y_pred = fit.m * log_r + np.log10(fit.A)
                ss_res = np.sum((log_y - y_pred) ** 2)
                ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                fit_r2[j] = 1.0 - ss_res / (ss_tot + 1e-30)

    return FitExponents(orders=orders, zeta_p=zeta_p, zeta_p_err=zeta_p_err, fit_r2=fit_r2)


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_structure_functions_with_scales(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    scales: ScaleInfo,
    fname: Path,
    title: str,
) -> None:
    """
    Plot S_p vs r with Taylor microscale marked on each curve.
    Two-panel: top = (S_p)^{1/p}, bottom = local slope zeta_p(r).
    """
    fig, (ax_main, ax_slope) = plt.subplots(
        2, 1, figsize=(10, 8), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05}
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(orders)))

    # Determine lambda_T to mark
    lambda_T = scales.lambda_T_gradient
    if not np.isfinite(lambda_T):
        lambda_T = scales.lambda_T_S2

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / p)
        color = colors[j]

        # Main curve
        ax_main.loglog(r, y, "-", lw=1.6, color=color, label=rf"$p={int(p)}$")

        # Mark Taylor microscale on each curve
        if np.isfinite(lambda_T) and lambda_T > r.min() and lambda_T < r.max():
            # Interpolate to find y at lambda_T
            y_at_lambda = np.interp(lambda_T, r, y)
            ax_main.scatter([lambda_T], [y_at_lambda], s=50, color=color,
                          edgecolor="white", linewidth=1.0, zorder=10)

        # Local slope
        centers, slopes = sliding_log_slope(r, y, window=4)
        if len(centers) > 0:
            ax_slope.semilogx(centers, slopes, "-", lw=1.2, color=color)

    # Mark fit region
    ax_main.axvline(scales.fit_min, color="gray", ls="--", lw=1, alpha=0.7, label=rf"$\lambda_T \approx {scales.fit_min:.0f}$")
    ax_main.axvline(scales.fit_max, color="gray", ls=":", lw=1, alpha=0.7, label=rf"$L_{{\rm int}} \approx {scales.fit_max:.0f}$")
    ax_slope.axvline(scales.fit_min, color="gray", ls="--", lw=1, alpha=0.7)
    ax_slope.axvline(scales.fit_max, color="gray", ls=":", lw=1, alpha=0.7)

    ax_main.set_ylabel(r"$(S_p)^{1/p}$", fontsize=12)
    ax_main.set_title(title, fontsize=13)
    ax_main.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    ax_main.grid(True, which="both", ls=":", lw=0.5)

    ax_slope.set_xlabel(r"separation $r$ [pixels]", fontsize=12)
    ax_slope.set_ylabel(r"$\zeta_p(r) = d\log(S_p^{1/p})/d\log r$", fontsize=11)
    ax_slope.grid(True, which="both", ls=":", lw=0.5)
    ax_slope.set_ylim(-0.1, 1.2)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_ess(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    scales: ScaleInfo,
    fname: Path,
    title: str,
) -> None:
    """
    Extended Self-Similarity (ESS) plot: log S_p vs log S_3.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)

    # Find S_3
    j3 = np.where(np.isclose(orders, 3.0))[0]
    if len(j3) == 0:
        # Fallback to S_2 if no S_3
        j3 = np.where(np.isclose(orders, 2.0))[0]
    if len(j3) == 0:
        plt.close(fig)
        return

    j3 = j3[0]
    S3 = S[j3]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(orders)))

    # Restrict to fit range
    mask = (r >= scales.fit_min) & (r <= scales.fit_max) & (S3 > 0)

    for j, p in enumerate(orders):
        Sp = S[j]
        valid = mask & (Sp > 0)
        if np.sum(valid) < 3:
            continue

        log_S3 = np.log10(S3[valid])
        log_Sp = np.log10(Sp[valid])

        ax.plot(log_S3, log_Sp, "o-", ms=4, lw=1.2, color=colors[j], label=rf"$p={int(p)}$")

        # Fit slope (ESS exponent)
        if np.sum(valid) >= 4:
            m, c = np.polyfit(log_S3, log_Sp, 1)
            x_fit = np.array([log_S3.min(), log_S3.max()])
            y_fit = m * x_fit + c
            ax.plot(x_fit, y_fit, "--", lw=1.5, color=colors[j], alpha=0.7)

    ax.set_xlabel(r"$\log_{10} S_3$", fontsize=12)
    ax.set_ylabel(r"$\log_{10} S_p$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(True, which="both", ls=":", lw=0.5)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectrum(
    theta: np.ndarray,
    L: float,
    fname: Path,
    title: str,
) -> None:
    """Plot scalar power spectrum E_theta(k)."""
    N = theta.shape[0]
    grid = SpectralGrid(N=N, L=L)
    k, E = scalar_power_spectrum(theta, grid, subtract_mean=True)

    plot_scalar_spectrum(
        k, E,
        fname=str(fname),
        title=title,
        fit_min_points=6,
        fit_min_decades=0.5,
        annotate_fit=True,
    )


def plot_zeta_vs_p_multipanel(
    results: List[Dict],
    fname: Path,
) -> None:
    """
    Multipanel plot of zeta_p vs p.
    One panel per alpha, different kappas as different colors.
    Horizontal reference line at alpha in each panel.
    """
    # Group by alpha
    alphas = sorted(set(r["alpha"] for r in results))
    kappas = sorted(set(r["kappa"] for r in results))

    n_alphas = len(alphas)
    ncols = min(3, n_alphas)
    nrows = (n_alphas + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=160, squeeze=False)

    kappa_colors = KAPPA_CMAP(np.linspace(0.15, 0.85, len(kappas)))
    kappa_to_color = {k: kappa_colors[i] for i, k in enumerate(kappas)}

    for idx, alpha in enumerate(alphas):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Get all results for this alpha
        alpha_results = [r for r in results if abs(r["alpha"] - alpha) < 0.01]

        for res in alpha_results:
            exps = res["exponents"]
            kappa = res["kappa"]
            color = kappa_to_color.get(kappa, "gray")

            valid = np.isfinite(exps.zeta_p)
            if np.any(valid):
                ax.plot(
                    exps.orders[valid], exps.zeta_p[valid],
                    "o-", ms=6, lw=1.8, color=color,
                    label=rf"$\kappa={kappa:.1e}$"
                )

        # Reference line at alpha
        ax.axhline(alpha, color="k", ls="--", lw=1.5, alpha=0.7, label=rf"$\alpha={alpha:.3f}$")

        # Obukhov-Corrsin reference (zeta_p = p/3 for passive scalar)
        p_ref = np.array([1, 2, 3, 4, 6, 8, 10])
        ax.plot(p_ref, p_ref / 3, ":", color="gray", lw=1.2, alpha=0.6, label=r"$p/3$ (O-C)")

        ax.set_xlabel(r"$p$", fontsize=11)
        ax.set_ylabel(r"$\zeta_p$", fontsize=11)
        ax.set_title(rf"$\alpha = {alpha:.3f}$", fontsize=12)
        ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 1.5)

    # Remove empty subplots
    for idx in range(n_alphas, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def create_summary_table(results: List[Dict], fname: Path) -> None:
    """Save a summary table of all computed quantities."""
    lines = [
        "# Theta Structure Function Analysis Summary",
        "# N=2048 Campaign",
        "#",
        f"# {'alpha':<8} {'kappa':<12} {'lambda_T_grad':<14} {'lambda_T_S2':<14} {'L_int':<10} " +
        " ".join([f"zeta_{int(p):<2}" for p in ORDERS]),
        "#" + "-" * 120,
    ]

    for res in sorted(results, key=lambda x: (x["alpha"], x["kappa"])):
        scales = res["scales"]
        exps = res["exponents"]

        zeta_str = " ".join([f"{z:6.3f}" if np.isfinite(z) else "   nan" for z in exps.zeta_p])

        line = (
            f"  {res['alpha']:<8.4f} {res['kappa']:<12.2e} "
            f"{scales.lambda_T_gradient:<14.2f} {scales.lambda_T_S2:<14.2f} "
            f"{scales.L_int:<10.2f} {zeta_str}"
        )
        lines.append(line)

    fname.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comprehensive theta SF analysis for 2048 campaign")
    parser.add_argument("--dry-run", action="store_true", help="List runs without processing")
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all 2048 runs
    runs = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir() or "N2048" not in run_dir.name:
            continue
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        theta = load_final_theta(run_dir)
        if theta is None:
            print(f"[skip] No theta data for {run_dir.name}")
            continue
        runs.append({
            "alpha": float(config["alpha"]),
            "kappa": float(config["kappa"]),
            "theta": theta,
            "run_dir": run_dir,
            "name": run_dir.name,
            "L": float(config.get("domain_size", 1.0)),
            "N": int(config.get("grid", 2048)),
        })

    runs.sort(key=lambda r: (r["alpha"], r["kappa"]))
    print(f"Found {len(runs)} N=2048 runs with theta data")

    if args.dry_run:
        for r in runs:
            print(f"  alpha={r['alpha']:.4f}, kappa={r['kappa']:.2e}")
        return 0

    # Process each run
    all_results = []
    orders_arr = np.array(ORDERS, dtype=float)

    for i, run in enumerate(runs):
        alpha = run["alpha"]
        kappa = run["kappa"]
        theta = run["theta"]
        L = run["L"]
        N = run["N"]

        print(f"\n[{i+1}/{len(runs)}] Processing alpha={alpha:.4f}, kappa={kappa:.2e}...")

        # Create run-specific output directory
        alpha_str = f"{alpha:.3f}".replace(".", "p")
        kappa_exp = np.log10(kappa)
        kappa_str = f"1e{kappa_exp:.1f}".replace(".", "p").replace("-", "m")
        run_out_dir = OUTPUT_DIR / f"alpha_{alpha_str}_kappa_{kappa_str}"
        run_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compute structure functions
        print("  Computing structure functions...")
        sf = structure_functions(
            theta,
            orders=ORDERS,
            n_ell_bins=60,
            n_disp_total=8192,
            use_fft_for_p2=True,
            seed=42,
        )
        r = sf["r"]
        S = sf["S"]

        # 2. Compute scales
        print("  Computing length scales...")
        j2 = np.where(np.isclose(orders_arr, 2.0))[0][0]
        S2 = S[j2]
        scales = compute_scales(theta, r, S2, L=L, N=N)
        print(f"    lambda_T (gradient): {scales.lambda_T_gradient:.2f} px")
        print(f"    lambda_T (S2):       {scales.lambda_T_S2:.2f} px")
        print(f"    L_int:               {scales.L_int:.2f} px")
        print(f"    Fit range:           [{scales.fit_min:.0f}, {scales.fit_max:.0f}] px")

        # 3. Fit scaling exponents
        print("  Fitting scaling exponents...")
        exponents = fit_scaling_exponents(r, S, orders_arr, scales.fit_min, scales.fit_max)
        for p, z in zip(exponents.orders, exponents.zeta_p):
            if np.isfinite(z):
                print(f"    zeta_{int(p)} = {z:.4f}")

        # 4. Generate plots
        print("  Generating plots...")

        # 4a. Structure functions with scales
        plot_structure_functions_with_scales(
            r, S, orders_arr, scales,
            run_out_dir / "structure_functions.png",
            rf"Scalar SF: $\alpha={alpha:.3f}$, $\kappa={kappa:.1e}$",
        )

        # 4b. ESS plot
        plot_ess(
            r, S, orders_arr, scales,
            run_out_dir / "ess_plot.png",
            rf"ESS: $\alpha={alpha:.3f}$, $\kappa={kappa:.1e}$",
        )

        # 4c. Power spectrum
        plot_power_spectrum(
            theta, L,
            run_out_dir / "power_spectrum.png",
            rf"Scalar Spectrum: $\alpha={alpha:.3f}$, $\kappa={kappa:.1e}$",
        )

        # Save per-run data
        np.savez_compressed(
            run_out_dir / "sf_data.npz",
            r=r,
            S=S,
            orders=orders_arr,
            lambda_T_gradient=scales.lambda_T_gradient,
            lambda_T_S2=scales.lambda_T_S2,
            L_int=scales.L_int,
            fit_min=scales.fit_min,
            fit_max=scales.fit_max,
            zeta_p=exponents.zeta_p,
            alpha=alpha,
            kappa=kappa,
        )

        all_results.append({
            "alpha": alpha,
            "kappa": kappa,
            "r": r,
            "S": S,
            "orders": orders_arr,
            "scales": scales,
            "exponents": exponents,
        })

    # 5. Generate summary plots
    print("\n" + "=" * 60)
    print("Generating summary plots...")

    # 5a. zeta_p vs p multipanel
    plot_zeta_vs_p_multipanel(all_results, OUTPUT_DIR / "zeta_vs_p_multipanel.png")
    print(f"  Saved: {OUTPUT_DIR / 'zeta_vs_p_multipanel.png'}")

    # 5b. Summary table
    create_summary_table(all_results, OUTPUT_DIR / "summary_table.txt")
    print(f"  Saved: {OUTPUT_DIR / 'summary_table.txt'}")

    # 5c. Save aggregated data
    np.savez_compressed(
        OUTPUT_DIR / "all_theta_sf.npz",
        alphas=np.array([r["alpha"] for r in all_results]),
        kappas=np.array([r["kappa"] for r in all_results]),
        r=all_results[0]["r"],
        orders=all_results[0]["orders"],
        S_all=np.array([r["S"] for r in all_results]),
        zeta_all=np.array([r["exponents"].zeta_p for r in all_results]),
        lambda_T_gradient=np.array([r["scales"].lambda_T_gradient for r in all_results]),
        lambda_T_S2=np.array([r["scales"].lambda_T_S2 for r in all_results]),
        L_int=np.array([r["scales"].L_int for r in all_results]),
    )
    print(f"  Saved: {OUTPUT_DIR / 'all_theta_sf.npz'}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
