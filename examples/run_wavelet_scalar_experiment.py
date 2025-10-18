#!/usr/bin/env python3
"""
Run wavelet-driven velocity and mean-gradient scalar advection experiments.

This script is inspired by ``examples/04_full_test.ipynb`` but adds the ability
to run fully automated experiments (test + production) from the command line.
It generates a synthetic divergence-free velocity field using wavelets, saves
velocity diagnostics, then evolves a passive scalar for multiple Peclet numbers.

For each scalar run we record:
    * Final scalar field snapshot.
    * Scalar structure functions.
    * Yaglom-like mixed increment statistics <|delta u| |delta theta|^2>.
    * Dissipation surrogate E_dot(nu) = nu * int_t <|grad theta|^2> dt (log-log scaling over nu).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

# Ensure headless plotting works even if a display is unavailable.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import (
    ScalarAdvectionAPI,
    ScalarConfig,
    generate_divfree_field,
    plot_structure_functions,
    structure_functions,
)
from scalar_advection.binning import find_ell_bin_edges
from scalar_advection.fitting import PowerLawFit, best_powerlaw_fit
from scalar_advection.structure import generate_displacements
from matplotlib.ticker import FixedLocator, FuncFormatter


DEFAULT_ORDERS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10)
DEFAULT_N_ELL_BINS = 40
DEFAULT_N_DISP = 4096


def best_fixed_slope_segment(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    *,
    min_points: int = 6,
    min_decades: float = 0.6,
    x_range: Tuple[float, float] | None = None,
) -> PowerLawFit | None:
    """
    Find the contiguous log-log segment that best matches y ~ A * x^s with fixed slope.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x_range is not None:
        lo, hi = x_range
        sel = (x >= lo) & (x <= hi)
        x = x[sel]
        y = y[sel]
    if x.size < min_points:
        return None

    lx = np.log10(x)
    ly = np.log10(y)

    best: PowerLawFit | None = None
    best_r2 = -np.inf
    n = lx.size
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            span = lx[j - 1] - lx[i]
            if span < min_decades:
                continue
            xi = lx[i:j]
            yi = ly[i:j]
            c = np.mean(yi - slope * xi)
            yi_fit = slope * xi + c
            ss_res = np.sum((yi - yi_fit) ** 2)
            ss_tot = np.sum((yi - yi.mean()) ** 2)
            if ss_tot <= 0:
                continue
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)
            if r2 > best_r2:
                best_r2 = r2
                xseg = x[i:j]
                yfit = 10 ** (slope * np.log10(xseg) + c)
                best = PowerLawFit(
                    i=i,
                    j=j,
                    m=slope,
                    A=10**c,
                    xseg=xseg,
                    yfit=yfit,
                )
    return best


@dataclass
class RunConfig:
    mode: str
    grid_size: int
    peclet_values: Sequence[float]
    t_end: float
    cfl: float
    integrator: str
    mean_grad: Tuple[float, float]
    fft_threads: int
    dtype: np.dtype
    velocity_seed: int
    disp_seed: int
    warm_cache: bool
    n_disp_total: int
    n_ell_bins: int
    yaglom_samples: int
    output_dir: Path
    verbose: bool


@dataclass
class ScalarRunResult:
    peclet: float
    kappa: float
    edot: float
    grad_sq_integral: float
    n_steps: int
    dt: float
    final_snapshot_path: Path
    structure_fn_path: Path
    yaglom_path: Path
    yaglom_unit_slope_range: Tuple[float, float] | None = None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wavelet velocity + scalar advection experiment driver."
    )
    parser.add_argument(
        "--mode",
        choices=("test", "full"),
        default="test",
        help="Select test (small grid) or full-resolution run.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        help="Override grid resolution (square). Defaults depend on mode.",
    )
    parser.add_argument(
        "--peclet",
        type=float,
        nargs="+",
        help="Explicit Peclet numbers to simulate (replace mode defaults).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=0.25,
        help="Simulation end time for scalar evolution (advective units).",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.75,
        help="CFL number used when auto-selecting dt.",
    )
    parser.add_argument(
        "--integrator",
        choices=("rk4", "etdrk4", "heun"),
        default="rk4",
        help="Time integrator for scalar solver.",
    )
    parser.add_argument(
        "--mean-grad",
        type=float,
        nargs=2,
        default=(1.0, 0.0),
        metavar=("Gx", "Gy"),
        help="Constant mean gradient forcing components.",
    )
    parser.add_argument(
        "--fft-threads",
        type=int,
        default=8,
        help="Threads for FFT backend.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point precision for grids.",
    )
    parser.add_argument(
        "--velocity-seed",
        type=int,
        default=1,
        help="Seed for wavelet velocity generation.",
    )
    parser.add_argument(
        "--disp-seed",
        type=int,
        default=0,
        help="Seed used for displacement sampling in diagnostics.",
    )
    parser.add_argument(
        "--n-disp-total",
        type=int,
        default=DEFAULT_N_DISP,
        help="Total displacement samples for structure functions.",
    )
    parser.add_argument(
        "--n-ell-bins",
        type=int,
        default=DEFAULT_N_ELL_BINS,
        help="Number of separation bins for structure/Yaglom statistics.",
    )
    parser.add_argument(
        "--yaglom-samples",
        type=int,
        default=8192,
        help="Number of random point pairs per displacement for Yaglom averages.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("examples") / "wavelet_scalar_outputs",
        help="Directory where experiment folders will be created.",
    )
    parser.add_argument(
        "--no-warm-cache",
        action="store_true",
        help="Skip FFT cache warming to shorten start-up.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress prints from the scalar solver.",
    )
    return parser.parse_args(argv)


def default_mode_settings(mode: str) -> Tuple[int, List[float]]:
    full_pe = [10 ** 4.5, 10 ** 4.0, 10 ** 3.5]
    if mode == "full":
        return 2048, full_pe
    # For test mode scale Peclet with grid size ratio (256/2048 = 1/8).
    scaled = [pe * (256.0 / 2048.0) for pe in full_pe]
    return 256, scaled


def build_run_config(args: argparse.Namespace) -> RunConfig:
    default_grid, default_pe = default_mode_settings(args.mode)
    grid = args.grid or default_grid
    peclet_values = args.peclet or default_pe
    dtype = np.float32 if args.dtype == "float32" else np.float64

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"{args.mode}_N{grid}_{timestamp}"

    return RunConfig(
        mode=args.mode,
        grid_size=grid,
        peclet_values=tuple(float(pe) for pe in peclet_values),
        t_end=args.t_end,
        cfl=args.cfl,
        integrator=args.integrator,
        mean_grad=(float(args.mean_grad[0]), float(args.mean_grad[1])),
        fft_threads=args.fft_threads,
        dtype=dtype,
        velocity_seed=args.velocity_seed,
        disp_seed=args.disp_seed,
        warm_cache=not args.no_warm_cache,
        n_disp_total=args.n_disp_total,
        n_ell_bins=args.n_ell_bins,
        yaglom_samples=args.yaglom_samples,
        output_dir=run_dir,
        verbose=not args.quiet,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_peclet(pe: float) -> str:
    exponent = math.log10(pe)
    return f"1e{exponent:.1f}" if abs(exponent - round(exponent)) > 1e-6 else f"1e{exponent:.0f}"


def save_array(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def save_velocity_diagnostics(
    velocity_dir: Path,
    ux: np.ndarray,
    uy: np.ndarray,
    orders: Sequence[int],
    n_ell_bins: int,
    n_disp_total: int,
    seed: int,
) -> dict:
    ensure_dir(velocity_dir)
    speed = np.hypot(ux, uy)
    np.savez_compressed(velocity_dir / "velocity_fields.npz", ux=ux, uy=uy, speed=speed)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200, constrained_layout=True)
    im = ax.imshow(speed, cmap=cmr.rainforest, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")
    fig.savefig(velocity_dir / "velocity_magnitude.png", bbox_inches="tight")
    plt.close(fig)

    sf_velocity = structure_functions(
        (ux, uy),
        orders=orders,
        n_ell_bins=n_ell_bins,
        n_disp_total=n_disp_total,
        seed=seed,
        use_fft_for_p2=True,
        signed_longitudinal=False,
    )
    np.savez_compressed(velocity_dir / "velocity_structure_functions.npz", **sf_velocity)

    plot_structure_functions(
        sf_velocity,
        title=f"Velocity structure functions (N={ux.shape[0]})",
        plot_long_and_tran=False,
        fname=velocity_dir / "velocity_structure_functions.png",
        fit_min_r=None,
        fit_max_r=None,
    )

    urms = float(np.sqrt(np.mean(ux**2 + uy**2)))
    summary = {
        "urms": urms,
        "orders": list(orders),
        "n_ell_bins": n_ell_bins,
        "n_displacements": n_disp_total,
    }
    save_json(velocity_dir / "velocity_summary.json", summary)
    return summary


def scalar_structure_function_plot(
    api: ScalarAdvectionAPI,
    theta: np.ndarray,
    orders: Sequence[int],
    n_ell_bins: int,
    n_disp_total: int,
    seed: int,
    fname: Path,
) -> dict:
    sf_scalar = api.scalar_structure_functions(
        theta,
        orders=orders,
        n_ell_bins=n_ell_bins,
        n_disp_total=n_disp_total,
        seed=seed,
        use_fft_for_p2=True,
    )
    np.savez_compressed(fname.with_suffix(".npz"), **sf_scalar)

    N = theta.shape[0]
    fit_min_r = max(16.0, N / 128.0)
    fit_max_r = N / 4.0

    r = sf_scalar["r"]
    orders_arr = sf_scalar["orders"]
    curves = sf_scalar["S"]
    colors = plt.cm.tab10.colors

    root_curves = np.array(
        [np.power(np.maximum(curves[j], 1e-30), 1.0 / orders_arr[j]) for j in range(len(orders_arr))]
    )

    fig, (ax_sf, ax_slope) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.05},
    )

    for j, p in enumerate(orders_arr):
        y = root_curves[j]
        color = colors[j % len(colors)]
        ax_sf.loglog(r, y, "-", lw=1.8, color=color, label=fr"p={p:g}")
        fit = best_powerlaw_fit(
            r,
            y,
            min_points=6,
            min_decades=0.5,
            x_range=(fit_min_r, fit_max_r),
        )
        if fit is not None:
            ax_sf.loglog(
                fit.xseg,
                fit.yfit,
                "--",
                lw=2.4,
                color=color,
                alpha=0.75,
            )

    ax_sf.set_ylabel(r"$(S_p)^{1/p}$")
    ax_sf.set_title(f"Scalar structure functions (final, N={N})")
    ax_sf.grid(True, which="both", ls=":", lw=0.6)
    ax_sf.legend(frameon=False, ncol=1, loc="lower right")

    def sliding_log_slope(rvals: np.ndarray, yvals: np.ndarray, window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        if rvals.size < window:
            return np.array([]), np.array([])
        lr = np.log(rvals)
        ly = np.log(np.maximum(yvals, 1e-30))
        slopes = []
        centers = []
        for i in range(rvals.size - window + 1):
            idx = slice(i, i + window)
            coeffs = np.polyfit(lr[idx], ly[idx], 1)
            slopes.append(coeffs[0])
            centers.append(np.exp(np.mean(lr[idx])))
        return np.array(centers), np.array(slopes)

    for j, p in enumerate(orders_arr):
        centers, slopes = sliding_log_slope(r, root_curves[j], window=4)
        if centers.size == 0:
            continue
        color = colors[j % len(colors)]
        ax_slope.semilogx(centers, slopes, "-", lw=1.6, color=color, label=fr"p={p:g}")

    ax_slope.set_xlabel(r"separation $\ell / \Delta x$")
    ax_slope.set_ylabel(r"$d\log (S_p^{1/p}) / d\log \ell$")
    ax_slope.grid(True, which="both", ls=":", lw=0.6)

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return sf_scalar


def yaglom_statistics(
    ux: np.ndarray,
    uy: np.ndarray,
    theta: np.ndarray,
    *,
    n_ell_bins: int,
    n_disp_total: int,
    samples_per_disp: int,
    seed: int,
    dx: float,
    L: float,
) -> dict:
    ny, nx = theta.shape
    r_max = min(nx, ny) // 2
    ell_edges = find_ell_bin_edges(1.0, r_max, n_ell_bins)
    n_per_bin = max(1, n_disp_total // n_ell_bins)
    disps = generate_displacements(ell_edges, n_per_bin, seed=seed)

    rng = np.random.default_rng(seed)
    accum = np.zeros(n_ell_bins, dtype=np.float64)
    counts = np.zeros(n_ell_bins, dtype=np.int64)

    for dx, dy in disps:
        r = math.hypot(dx, dy)
        if r == 0.0:
            continue
        b = np.searchsorted(ell_edges, r, side="right") - 1
        if b < 0 or b >= n_ell_bins:
            continue

        n_samples = min(samples_per_disp, nx * ny)
        iy = rng.integers(0, ny, size=n_samples, endpoint=False)
        ix = rng.integers(0, nx, size=n_samples, endpoint=False)
        iyp = (iy + dy) % ny
        ixp = (ix + dx) % nx

        dux = ux[iyp, ixp] - ux[iy, ix]
        duy = uy[iyp, ixp] - uy[iy, ix]
        dtheta = theta[iyp, ixp] - theta[iy, ix]

        delta_u_mag = np.sqrt(dux * dux + duy * duy)
        delta_theta_sq = dtheta * dtheta
        term = delta_u_mag * delta_theta_sq
        accum[b] += float(np.mean(term))
        counts[b] += 1

    r_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    with np.errstate(invalid="ignore"):
        yaglom = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)

    result = {
        "r": r_centers,
        "counts": counts,
        "yaglom": yaglom,
        "ell_edges": ell_edges,
        "samples_per_disp": samples_per_disp,
        "dx": dx,
        "L": L,
    }
    return result


def plot_yaglom(yaglom_data: dict, fname: Path) -> Tuple[float | None, PowerLawFit | None]:
    r = yaglom_data["r"]
    y = yaglom_data["yaglom"]
    mask = (r > 0) & (y > 0)
    r = r[mask]
    y = y[mask]

    ell_edges = yaglom_data["ell_edges"]
    dx = float(yaglom_data.get("dx", 1.0))
    L = float(yaglom_data.get("L", 1.0))
    N = max(2, int(round(L / dx)))
    fit_hi = min(N / 4.0, r.max())
    fit_lo = max(6.0, N / 256.0)
    if fit_hi <= fit_lo:
        fit_hi = r.max()
    fit_range = (fit_lo, fit_hi)
    fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.6, x_range=fit_range)
    slope_one_fit = best_fixed_slope_segment(
        r,
        y,
        1.0,
        min_points=6,
        min_decades=0.6,
        x_range=fit_range,
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=160)
    ax.loglog(r, y, "o-", lw=1.6, label=r"$\langle | \delta u | | \delta \theta |^2 \rangle$")
    slope = None
    if fit is not None:
        ax.loglog(
            fit.xseg,
            fit.yfit,
            "k--",
            lw=2.3,
            label=fr"best fit: $\ell/\Delta x \in [{fit.xseg[0]:.1f}, {fit.xseg[-1]:.1f}],\ m={fit.m:.3f}$",
        )
        slope = fit.m
    if slope_one_fit is not None:
        ax.loglog(
            slope_one_fit.xseg,
            slope_one_fit.yfit,
            "k-.",
            lw=2.0,
            label=fr"slope $1$: $\ell/\Delta x \in [{slope_one_fit.xseg[0]:.1f}, {slope_one_fit.xseg[-1]:.1f}]$",
        )
    ax.set_xlabel(r"separation $\ell / \Delta x$")
    ax.set_ylabel(r"$\langle | \delta u | | \delta \theta |^2 \rangle$")
    ax.grid(True, which="both", ls=":", lw=0.6)

    def to_fraction(x):
        return (x * dx) / L

    def from_fraction(x):
        return (x * L) / dx

    ax_top = ax.secondary_xaxis("top", functions=(to_fraction, from_fraction))
    ax_top.set_xlabel(r"$\ell / L$")
    ax_top.set_xscale("log")

    frac_limits = to_fraction(np.array(ax.get_xlim()))
    ax_top.set_xlim(frac_limits[0], frac_limits[1])

    n_levels = int(np.ceil(np.log2(N)))
    candidate = 1.0 / (2.0 ** np.arange(1, n_levels + 1, dtype=float))
    valid = candidate[(candidate >= min(frac_limits)) & (candidate <= max(frac_limits))]
    if valid.size:
        ax_top.xaxis.set_major_locator(FixedLocator(valid))

        def _fmt_fraction(val: float, _pos: int) -> str:
            if val <= 0:
                return ""
            denom = int(round(1.0 / val))
            if denom > 0 and np.isclose(val, 1.0 / denom, rtol=1e-6, atol=1e-9):
                return rf"$1/{denom}$"
            return f"{val:.3g}"

        ax_top.xaxis.set_major_formatter(FuncFormatter(_fmt_fraction))

    ax.legend(frameon=False)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return slope, slope_one_fit


def run_scalar_case(
    api: ScalarAdvectionAPI,
    ux: np.ndarray,
    uy: np.ndarray,
    peclet: float,
    run_dir: Path,
    run_cfg: RunConfig,
    orders: Sequence[int],
) -> ScalarRunResult:
    pe_dir = run_dir / f"scalar_pe_{format_peclet(peclet)}"
    ensure_dir(pe_dir)

    summary_path = pe_dir / "scalar_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        theta_final = np.load(pe_dir / "theta_final.npy")
        sf_path = pe_dir / "scalar_structure_functions.png"
        scalar_structure_function_plot(
            api,
            theta_final,
            orders=orders,
            n_ell_bins=run_cfg.n_ell_bins,
            n_disp_total=run_cfg.n_disp_total,
            seed=run_cfg.disp_seed,
            fname=sf_path,
        )
        yaglom_data = yaglom_statistics(
            ux,
            uy,
            theta_final,
            n_ell_bins=run_cfg.n_ell_bins,
            n_disp_total=run_cfg.n_disp_total,
            samples_per_disp=run_cfg.yaglom_samples,
            seed=run_cfg.disp_seed,
            dx=api.grid.dx,
            L=api.grid.L,
        )
        np.savez_compressed(pe_dir / "yaglom_stats.npz", **yaglom_data)
        slope, slope_one_fit = plot_yaglom(yaglom_data, pe_dir / "yaglom_scaling.png")
        data["yaglom_slope"] = float(slope) if slope is not None else None
        data["yaglom_counts"] = yaglom_data["counts"].tolist()
        data["samples_per_disp"] = run_cfg.yaglom_samples
        if slope_one_fit is not None:
            data["yaglom_unit_slope_range"] = [
                float(slope_one_fit.xseg[0]),
                float(slope_one_fit.xseg[-1]),
            ]
        else:
            data["yaglom_unit_slope_range"] = None
        save_json(summary_path, data)
        return ScalarRunResult(
            peclet=float(data["peclet"]),
            kappa=float(data["kappa"]),
            edot=float(data["edot"]),
            grad_sq_integral=float(data["grad_sq_integral"]),
            n_steps=int(data["n_steps"]),
            dt=float(data["dt"]),
            final_snapshot_path=pe_dir / "theta_final.npy",
            structure_fn_path=pe_dir / "scalar_structure_functions.npz",
            yaglom_path=pe_dir / "yaglom_stats.npz",
            yaglom_unit_slope_range=(
                tuple(data["yaglom_unit_slope_range"])
                if data.get("yaglom_unit_slope_range") is not None
                else None
            ),
        )

    theta0 = np.zeros((api.grid.N, api.grid.N), dtype=api.grid.dtype)
    scalar_cfg = ScalarConfig(
        peclet=peclet,
        t_end=run_cfg.t_end,
        mean_grad=run_cfg.mean_grad,
        save_every=None,
        output_frames=False,
        frame_interval=None,
        integrator=run_cfg.integrator,
        cfl=run_cfg.cfl,
    )

    theta_final, diagnostics = api.evolve_scalar(
        theta0, ux, uy, scalar_cfg, verbose=run_cfg.verbose
    )

    np.save(pe_dir / "theta_final.npy", theta_final)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=220, constrained_layout=True)
    vmax = np.percentile(np.abs(theta_final), 99.0)
    im = ax.imshow(
        theta_final,
        cmap=cmr.iceburn,
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\theta$")
    fig.savefig(pe_dir / "theta_final.png", bbox_inches="tight")
    plt.close(fig)

    sf_path = pe_dir / "scalar_structure_functions.png"
    scalar_structure_function_plot(
        api,
        theta_final,
        orders=orders,
        n_ell_bins=run_cfg.n_ell_bins,
        n_disp_total=run_cfg.n_disp_total,
        seed=run_cfg.disp_seed,
        fname=sf_path,
    )

    yaglom_data = yaglom_statistics(
        ux,
        uy,
        theta_final,
        n_ell_bins=run_cfg.n_ell_bins,
        n_disp_total=run_cfg.n_disp_total,
        samples_per_disp=run_cfg.yaglom_samples,
        seed=run_cfg.disp_seed,
        dx=api.grid.dx,
        L=api.grid.L,
    )
    np.savez_compressed(pe_dir / "yaglom_stats.npz", **yaglom_data)
    slope, slope_one_fit = plot_yaglom(yaglom_data, pe_dir / "yaglom_scaling.png")

    edot = diagnostics.kappa * diagnostics.grad_sq_integral

    summary = {
        "peclet": peclet,
        "kappa": diagnostics.kappa,
        "n_steps": diagnostics.n_steps,
        "dt": diagnostics.dt,
        "grad_sq_integral": diagnostics.grad_sq_integral,
        "edot": edot,
        "yaglom_slope": float(slope) if slope is not None else None,
        "yaglom_counts": yaglom_data["counts"].tolist(),
        "samples_per_disp": run_cfg.yaglom_samples,
    }
    if slope_one_fit is not None:
        summary["yaglom_unit_slope_range"] = [
            float(slope_one_fit.xseg[0]),
            float(slope_one_fit.xseg[-1]),
        ]
    else:
        summary["yaglom_unit_slope_range"] = None
    save_json(pe_dir / "scalar_summary.json", summary)

    return ScalarRunResult(
        peclet=peclet,
        kappa=diagnostics.kappa,
        edot=edot,
        grad_sq_integral=diagnostics.grad_sq_integral,
        n_steps=diagnostics.n_steps,
        dt=diagnostics.dt,
        final_snapshot_path=pe_dir / "theta_final.npy",
        structure_fn_path=sf_path.with_suffix(".npz"),
        yaglom_path=pe_dir / "yaglom_stats.npz",
        yaglom_unit_slope_range=(
            (float(slope_one_fit.xseg[0]), float(slope_one_fit.xseg[-1]))
            if slope_one_fit is not None
            else None
        ),
    )


def plot_edot_scaling(results: Sequence[ScalarRunResult], fname: Path) -> None:
    kappas = np.array([r.kappa for r in results], dtype=np.float64)
    edots = np.array([r.edot for r in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    ax.loglog(kappas, edots, "o-", lw=2)
    ax.set_xlabel(r"viscosity $\nu$")
    ax.set_ylabel(r"$\nu \int_0^T \int_\Omega |\nabla \theta|^2 \, d\mathbf{x}\, dt$")
    ax.grid(True, which="both", ls=":", lw=0.6)

    fit = best_powerlaw_fit(kappas, edots, min_points=3, min_decades=0.3)
    if fit is not None:
        ax.loglog(fit.xseg, fit.yfit, "k--", lw=2.0, label=fr"fit: $\propto \nu^{{{fit.m:.3f}}}$")
        ax.legend(frameon=False)

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_cfg = build_run_config(args)

    ensure_dir(run_cfg.output_dir)
    meta_path = run_cfg.output_dir / "run_config.json"
    save_json(
        meta_path,
        {
            "mode": run_cfg.mode,
            "grid_size": run_cfg.grid_size,
            "peclet_values": list(run_cfg.peclet_values),
            "t_end": run_cfg.t_end,
            "cfl": run_cfg.cfl,
            "integrator": run_cfg.integrator,
            "mean_grad": run_cfg.mean_grad,
            "fft_threads": run_cfg.fft_threads,
            "dtype": str(run_cfg.dtype),
            "velocity_seed": run_cfg.velocity_seed,
            "disp_seed": run_cfg.disp_seed,
            "warm_cache": run_cfg.warm_cache,
            "n_disp_total": run_cfg.n_disp_total,
            "n_ell_bins": run_cfg.n_ell_bins,
            "yaglom_samples": run_cfg.yaglom_samples,
        },
    )

    api = ScalarAdvectionAPI(
        N=run_cfg.grid_size,
        L=1.0,
        dtype=run_cfg.dtype,
        warm_cache=run_cfg.warm_cache,
    )
    api.set_fft_threads(run_cfg.fft_threads)

    ux, uy, _ = generate_divfree_field(
        N=run_cfg.grid_size,
        lam_min=3,
        lam_max=run_cfg.grid_size,
        slope=-5.0 / 3.0,
        wavelet="mexh",
        sparsity=0.0,
        seed=run_cfg.velocity_seed,
    )
    ux = ux.astype(api.grid.dtype, copy=False)
    uy = uy.astype(api.grid.dtype, copy=False)

    velocity_dir = run_cfg.output_dir / "velocity"
    velocity_summary = save_velocity_diagnostics(
        velocity_dir,
        ux,
        uy,
        DEFAULT_ORDERS,
        run_cfg.n_ell_bins,
        run_cfg.n_disp_total,
        run_cfg.disp_seed,
    )

    scalar_results: List[ScalarRunResult] = []
    for pe in run_cfg.peclet_values:
        if run_cfg.verbose:
            print(f"\n--- Running scalar simulation for Peclet={pe:.4g} ---")
        result = run_scalar_case(api, ux, uy, pe, run_cfg.output_dir, run_cfg, DEFAULT_ORDERS)
        scalar_results.append(result)

    plot_edot_scaling(scalar_results, run_cfg.output_dir / "edot_vs_viscosity.png")

    aggregate = {
        "velocity_summary": velocity_summary,
        "scalar_runs": [
            {
                "peclet": r.peclet,
                "kappa": r.kappa,
                "edot": r.edot,
                "grad_sq_integral": r.grad_sq_integral,
                "n_steps": r.n_steps,
                "dt": r.dt,
                "theta_final_path": str(r.final_snapshot_path),
                "structure_functions_path": str(r.structure_fn_path),
                "yaglom_path": str(r.yaglom_path),
                "yaglom_unit_slope_range": (
                    list(r.yaglom_unit_slope_range) if r.yaglom_unit_slope_range else None
                ),
            }
            for r in scalar_results
        ],
    }
    save_json(run_cfg.output_dir / "summary.json", aggregate)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
