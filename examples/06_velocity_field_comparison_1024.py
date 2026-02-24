#!/usr/bin/env python3
"""Compare 1024×1024 velocity field generators and plot maps + structure functions.

Generates three field types:
  - Fourier spectral (VelocityFieldGenerator via generate_velocity_field)
  - Wavelet-based mexican-hat
  - Wavelet-based Haar (three orientations combined per scale)

The script writes per-field map and structure-function plots plus a cross-field
comparison figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scalar_advection import VelocityConfig
from scalar_advection.grid import SpectralGrid
from scalar_advection.fitting import best_powerlaw_fit
from scalar_advection.structure import structure_functions
from scalar_advection.velocity import generate_divfree_field, generate_velocity_field


def _parse_orders(raw: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("--orders requires a comma-separated list")
    return tuple(vals)


def _scale_to_rms(ux: np.ndarray, uy: np.ndarray, target_rms: float) -> Tuple[np.ndarray, np.ndarray]:
    cur = np.sqrt(np.mean(ux * ux + uy * uy))
    if not np.isfinite(cur) or cur <= 0:
        return ux, uy
    s = float(target_rms) / float(cur)
    return ux * s, uy * s


def _divergence_rms(ux: np.ndarray, uy: np.ndarray, L: float) -> Dict[str, float]:
    N = int(ux.shape[0])
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky = np.meshgrid(k, k, indexing="xy")

    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    div_hat = 1j * (kx * ux_hat + ky * uy_hat)
    div = np.fft.ifft2(div_hat).real

    return {
        "l2": float(np.sqrt(np.mean(div**2))),
        "max_abs": float(np.max(np.abs(div))),
        "rms_field": float(np.sqrt(np.mean(ux * ux + uy * uy))),
    }


def _generate_fourier_field(
    grid: SpectralGrid,
    alpha: float,
    lam_min: float,
    lam_max: float,
    seed: int,
    urms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Use k units consistent with spectral grid normalization (k_norm in units of domain modes).
    # With L=N, lam = N/k.
    cfg = VelocityConfig(
        alpha=float(alpha),
        urms=float(urms),
        seed=int(seed),
        f_sol=1.0,
        kmin=float(grid.N / lam_max),
        kmax=float(grid.N / lam_min),
        taper_width=0.0,
    )
    return generate_velocity_field(grid, cfg)


def _generate_wavelet_field(
    name: str,
    N: int,
    alpha: float,
    lam_min: float,
    lam_max: float,
    seed: int,
    urms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    wavelet = "mexh" if name == "wavelet_mexh" else "haar"
    ux, uy, _ = generate_divfree_field(
        N=N,
        lam_min=float(lam_min),
        lam_max=float(lam_max),
        alpha=float(alpha),
        wavelet=wavelet,
        sparsity=0.0,
        seed=int(seed),
        taper_frac=0.05,
        amp=1.0,
    )
    return _scale_to_rms(ux, uy, urms)


def _structure_summary(
    ux: np.ndarray,
    uy: np.ndarray,
    orders: Sequence[int],
    n_ell_bins: int,
    n_disp_total: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sf = structure_functions(
        (ux, uy),
        orders=tuple(float(p) for p in orders),
        n_ell_bins=int(n_ell_bins),
        n_disp_total=int(n_disp_total),
        use_fft_for_p2=True,
        signed_longitudinal=False,
        seed=int(seed),
    )
    r = sf["r"]
    S = sf["mag"]
    orders_arr = sf["orders"]
    counts = sf["counts"]
    return r, S, orders_arr, counts


def _plot_velocity_map(
    ux: np.ndarray,
    uy: np.ndarray,
    label: str,
    out_dir: Path,
    quiver_stride: int = 32,
) -> Path:
    speed = np.hypot(ux, uy)

    finite = np.isfinite(speed)
    pmax = float(np.percentile(speed[finite], 99.0)) if np.any(finite) else 1.0
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = float(np.max(speed)) if np.size(speed) else 1.0
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0

    q = max(1, int(quiver_stride))
    ux_sub = ux[::q, ::q]
    uy_sub = uy[::q, ::q]
    sp_sub = np.hypot(ux_sub, uy_sub)
    finite_sub = np.isfinite(sp_sub)
    vmax = float(np.percentile(sp_sub[finite_sub], 90.0)) if np.any(finite_sub) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    target_len = 0.7 * float(q)
    scale_fac = target_len / (vmax + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=180)
    xs, ys = np.meshgrid(np.arange(0, ux.shape[1], q), np.arange(0, ux.shape[0], q))

    im0 = axes[0].imshow(ux, origin="lower", cmap="coolwarm")
    axes[0].set_title("u_x")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.047, pad=0.03)

    im1 = axes[1].imshow(uy, origin="lower", cmap="coolwarm")
    axes[1].set_title("u_y")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.047, pad=0.03)

    im2 = axes[2].imshow(speed, origin="lower", cmap="viridis", vmin=0.0, vmax=pmax)
    axes[2].set_title(r"$|\mathbf{u}|$ with downsampled vectors")
    axes[2].axis("off")
    axes[2].quiver(
        xs,
        ys,
        ux_sub * scale_fac,
        uy_sub * scale_fac,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="k",
        alpha=0.65,
        width=0.0022,
        headwidth=3.4,
        headlength=4.0,
        headaxislength=3.5,
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.047, pad=0.03, label=r"$|\mathbf{u}|$")

    fig.suptitle(label)
    fig.tight_layout()

    out = out_dir / f"{label}_velocity_map.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return out


def _plot_field_sf(
    label: str,
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    alpha: float,
    out_path: Path,
    fit_lo: float,
    fit_hi: float,
) -> None:
    fig, (ax_main, ax_zeta) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(orders)))

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / float(p))
        m = colors[j]
        ax_main.loglog(r, y, "o-", ms=3.5, lw=1.6, color=m, label=rf"$p={int(p)}")

        fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.5, x_range=(fit_lo, fit_hi))
        if fit is not None:
            ax_main.loglog(fit.xseg, fit.yfit, "--", lw=1.6, color=m, alpha=0.75)

        centers, slopes = _local_log_slope(r, y, window=8)
        if len(centers):
            ax_zeta.semilogx(centers, slopes, "-", lw=1.25, color=m, alpha=0.85)

    ax_main.set_ylabel(r"$(S_p)^{1/p}$", fontsize=12)
    ax_main.set_title(f"Structure functions: {label} (target α={alpha:g})", fontsize=13)
    ax_main.grid(True, which="both", ls=":", lw=0.5)
    ax_main.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.8)
    ax_main.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.8)
    ax_main.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")

    ax_zeta.set_xlabel(r"separation $r$ [pixels]", fontsize=12)
    ax_zeta.set_ylabel(r"local log slope")
    ax_zeta.grid(True, which="both", ls=":", lw=0.5)
    ax_zeta.axhline(alpha, color="k", ls="--", lw=1.3, alpha=0.75)
    ax_zeta.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.8)
    ax_zeta.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_compare_sfs(
    cases: List[Dict[str, object]],
    selected_orders: Sequence[int],
    out_path: Path,
    fit_lo: float,
    fit_hi: float,
    alpha: float,
) -> None:
    if not selected_orders:
        return

    ncols = len(selected_orders)
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 4.4), dpi=180, sharey=False)
    if ncols == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    colors = cmap(np.linspace(0.1, 0.9, len(cases)))

    for ax, p in zip(axes, selected_orders):
        for j, case in enumerate(cases):
            orders = np.asarray(case["orders"], dtype=float)
            idx = int(np.where(np.isclose(orders, float(p)))[0][0]) if np.any(np.isclose(orders, float(p))) else None
            if idx is None:
                continue

            r = np.asarray(case["r"], dtype=float)
            S = np.asarray(case["S"], dtype=float)
            y = np.power(np.maximum(S[idx], 1e-30), 1.0 / float(p))
            label = str(case["label"])
            c = colors[j]

            ax.loglog(r, y, "o-", ms=3.0, lw=1.6, color=c, label=label)
            fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.5, x_range=(fit_lo, fit_hi))
            if fit is not None:
                ax.loglog(fit.xseg, fit.yfit, "--", lw=1.5, color=c, alpha=0.75)

        ax.set_title(rf"$p={p:g}$")
        ax.set_xlabel(r"separation $r$")
        ax.grid(True, which="both", ls=":", lw=0.5)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$(S_p)^{1/p}$")
    fig.suptitle(f"Velocity SF comparison (target α={alpha:g})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _local_log_slope(r: np.ndarray, y: np.ndarray, window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    mask = (r > 0) & np.isfinite(r) & np.isfinite(y) & (y > 0)
    r2 = r[mask]
    y2 = y[mask]
    if len(r2) < window:
        return np.array([]), np.array([])

    lr = np.log(r2)
    ly = np.log(y2)
    slopes = []
    centers = []

    for i in range(len(r2) - window + 1):
        idx = slice(i, i + window)
        coeff = np.polyfit(lr[idx], ly[idx], 1)
        slopes.append(float(coeff[0]))
        centers.append(float(np.exp(np.mean(lr[idx]))))

    return np.asarray(centers), np.asarray(slopes)


def _ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1024×1024 velocity field visual comparison")
    parser.add_argument("--alpha", type=float, default=1.0 / 3.0, help="Target structure-function slope")
    parser.add_argument("--lam-min", type=float, default=2.0, help="Smallest wavelength scale (px) for wavelet cutoff")
    parser.add_argument("--lam-max", type=float, default=1024.0, help="Largest wavelength scale (px) for wavelet cutoff")
    parser.add_argument("--seed", type=int, default=1001, help="RNG seed")
    parser.add_argument("--urms", type=float, default=1.0, help="Target RMS velocity magnitude")
    parser.add_argument("--n-ell-bins", type=int, default=40, help="Structure-function radial bins")
    parser.add_argument("--n-disp-total", type=int, default=8192, help="Displacement samples for structure functions")
    parser.add_argument("--orders", type=_parse_orders, default="1,2,3,4,6,8", help="Comma-separated orders")
    parser.add_argument("--output-dir", type=Path, default=Path("examples") / "velocity_field_comparison_1024", help="Output directory")
    parser.add_argument("--compare-orders", type=_parse_orders, default="1,2,3", help="Orders used for cross-case comparison plots")
    parser.add_argument("--fit-lo", type=float, default=8.0, help="Fit/slope lower cutoff for r")
    parser.add_argument("--fit-hi", type=float, default=256.0, help="Fit/slope upper cutoff for r")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    N = 1024
    L = float(N)
    orders = args.orders
    compare_orders = args.compare_orders

    if args.lam_min >= args.lam_max:
        raise ValueError("--lam-min must be smaller than --lam-max")
    if N <= 0:
        raise ValueError("N must be positive")

    out_dir = _ensure_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparisons on {N}x{N} grid")
    print(f"alpha={args.alpha:.6g}, lam_min={args.lam_min:g}, lam_max={args.lam_max:g}, seed={args.seed}")
    print(f"orders={orders}")

    grid = SpectralGrid(N=N, L=L, dtype=np.float64)
    cases: List[Dict[str, object]] = []

    generators = {
        "fourier": (
            "Fourier spectral",
            lambda: _generate_fourier_field(grid, args.alpha, args.lam_min, args.lam_max, args.seed, args.urms),
        ),
        "wavelet_mexh": (
            "Wavelet mexh",
            lambda: _generate_wavelet_field("wavelet_mexh", N, args.alpha, args.lam_min, args.lam_max, args.seed + 1, args.urms),
        ),
        "wavelet_haar": (
            "Wavelet haar",
            lambda: _generate_wavelet_field("wavelet_haar", N, args.alpha, args.lam_min, args.lam_max, args.seed + 2, args.urms),
        ),
    }

    for idx, (key, (label, builder)) in enumerate(generators.items()):
        print(f"Generating {label} field")
        ux, uy = builder()
        ux = np.asarray(ux)
        uy = np.asarray(uy)

        metrics = _divergence_rms(ux, uy, L)
        metrics["label"] = label
        metrics["seed"] = args.seed + idx

        field_dir = out_dir / key
        _ensure_path(field_dir)

        map_path = _plot_velocity_map(ux, uy, label=label, out_dir=field_dir, quiver_stride=32)
        print(f"  wrote {map_path}")

        r, S, orders_arr, counts = _structure_summary(
            ux,
            uy,
            orders=orders,
            n_ell_bins=args.n_ell_bins,
            n_disp_total=args.n_disp_total,
            seed=args.seed + idx,
        )

        sf_path = field_dir / "velocity_structure_functions.png"
        _plot_field_sf(label, r, S, orders_arr, args.alpha, sf_path, fit_lo=args.fit_lo, fit_hi=args.fit_hi)

        cases.append(
            {
                "label": label,
                "r": r,
                "S": S,
                "orders": orders_arr,
                "counts": counts,
                "l2_div": metrics["l2"],
                "max_div": metrics["max_abs"],
                "rms": metrics["rms_field"],
                "map_path": map_path,
                "sf_path": sf_path,
            }
        )

        print(f"  RMS={metrics['rms_field']:.6g}, div-l2={metrics['l2']:.2e}, div-max={metrics['max_abs']:.2e}")

    compare_path = out_dir / "compare_velocity_structure_functions.png"
    _plot_compare_sfs(cases, compare_orders, compare_path, fit_lo=args.fit_lo, fit_hi=args.fit_hi, alpha=args.alpha)
    print(f"  wrote {compare_path}")

    with (out_dir / "velocity_field_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Grid: {N}x{N}\n")
        fh.write(f"alpha={args.alpha}\n")
        fh.write(f"lam_min={args.lam_min}, lam_max={args.lam_max}\n")
        fh.write(f"seed={args.seed}\n\n")

        for case in cases:
            fh.write(f"{case['label']}\n")
            fh.write(f"  map: {Path(case['map_path']).as_posix()}\n")
            fh.write(f"  sf: {Path(case['sf_path']).as_posix()}\n")
            fh.write(f"  rms: {case['rms']:.6g}\n")
            fh.write(f"  div_l2: {case['l2_div']:.3e}\n")
            fh.write(f"  div_max: {case['max_div']:.3e}\n")
            fh.write("\n")

    print(f"All outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
