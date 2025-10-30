#!/usr/bin/env python3
"""
Single-Peclet variant of the wavelet-driven scalar experiment.

This script reuses the diagnostics pipeline from
``examples/run_wavelet_scalar_experiment.py`` but specialises it for a single
simulation (default: N=4096, Pe=1e5).  It supports reusing previously generated
velocity fields, stores both the velocity and final scalar in `.npz` format so
they can be recycled, and exposes knobs that help make very large runs more
tractable (precision, FFT threads, integrator choice, etc.).

Example quick test (smaller grid):
    python examples/run_single_peclet_wavelet_scalar.py --grid 512 --peclet 1e4 --quiet

Example production run:
    python examples/run_single_peclet_wavelet_scalar.py --grid 4096 --peclet 1e5 --fft-threads 16
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import matplotlib

matplotlib.use("Agg")

from scalar_advection import ScalarAdvectionAPI, ScalarConfig, generate_divfree_field  # noqa: E402

from run_wavelet_scalar_experiment import (  # noqa: E402
    DEFAULT_ORDERS,
    DEFAULT_N_DISP,
    RunConfig,
    ScalarRunResult,
    ensure_dir,
    format_peclet,
    plot_yaglom,
    save_velocity_diagnostics,
    scalar_structure_function_plot,
    yaglom_statistics,
)


@dataclass
class SingleRunConfig:
    grid: int
    peclet: float
    dtype: np.dtype
    integrator: str
    cfl: float
    t_end: float
    fft_threads: int
    mean_grad: Tuple[float, float]
    n_disp_total: int
    n_ell_bins: int
    yaglom_samples: int
    velocity_seed: int
    disp_seed: int
    warm_cache: bool
    verbose: bool


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single Peclet scalar advection run with reusable velocity field."
    )
    parser.add_argument("--grid", type=int, default=4096, help="Grid resolution N (default: 4096).")
    parser.add_argument("--peclet", type=float, default=1e5, help="Target Peclet number.")
    parser.add_argument("--integrator", choices=("etdrk4", "rk4", "heun"), default="etdrk4")
    parser.add_argument("--t-end", type=float, default=0.25, help="Simulation end time.")
    parser.add_argument("--cfl", type=float, default=0.6, help="CFL number for adaptive dt.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Real-space dtype for the solver (float32 recommended for speed).",
    )
    parser.add_argument("--fft-threads", type=int, default=8, help="Threads for FFT backend.")
    parser.add_argument("--velocity-seed", type=int, default=1, help="RNG seed for velocity field.")
    parser.add_argument("--disp-seed", type=int, default=0, help="Seed for displacement sampling.")
    parser.add_argument(
        "--mean-grad",
        type=float,
        nargs=2,
        default=(1.0, 0.0),
        metavar=("Gx", "Gy"),
        help="Mean gradient forcing vector.",
    )
    parser.add_argument(
        "--velocity-file",
        type=Path,
        help="Optional .npz containing fields 'ux' and 'uy' to reuse instead of regenerating.",
    )
    parser.add_argument(
        "--reuse-velocity",
        action="store_true",
        help="If set, reuse velocity stored in the output directory when available.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("examples") / "single_wavelet_runs",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory (otherwise a timestamped folder is created).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag appended to the output directory name.",
    )
    parser.add_argument(
        "--n-ell-bins",
        type=int,
        default=40,
        help="Number of separation bins for diagnostics.",
    )
    parser.add_argument(
        "--n-disp-total",
        type=int,
        default=DEFAULT_N_DISP,
        help="Total displacement samples for structure/Yaglom stats.",
    )
    parser.add_argument(
        "--yaglom-samples",
        type=int,
        default=8192,
        help="Random pairs per displacement bin for Yaglom estimation.",
    )
    parser.add_argument(
        "--no-warm-cache",
        action="store_true",
        help="Skip FFT warm-up (speeds start-up, increases first-step cost).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress solver progress output.",
    )
    return parser.parse_args(argv)


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    name = f"N{args.grid}_Pe{format_peclet(args.peclet)}_{timestamp}{tag}"
    return args.output_root / name


def load_velocity_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "ux" not in data or "uy" not in data:
        raise ValueError(f"Velocity npz at {path} must contain 'ux' and 'uy'.")
    return data["ux"], data["uy"]


def maybe_reuse_velocity(args: argparse.Namespace, out_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if args.velocity_file:
        if not args.velocity_file.exists():
            raise FileNotFoundError(f"Provided velocity file does not exist: {args.velocity_file}")
        return load_velocity_from_npz(args.velocity_file)
    if args.reuse_velocity:
        candidate = out_dir / "velocity" / "velocity_fields.npz"
        if candidate.exists():
            return load_velocity_from_npz(candidate)
    return None


def run_single_simulation(args: argparse.Namespace) -> Tuple[ScalarRunResult, Path]:
    out_dir = build_output_dir(args)
    ensure_dir(out_dir)

    config = SingleRunConfig(
        grid=args.grid,
        peclet=args.peclet,
        dtype=np.float32 if args.dtype == "float32" else np.float64,
        integrator=args.integrator,
        cfl=args.cfl,
        t_end=args.t_end,
        fft_threads=args.fft_threads,
        mean_grad=(float(args.mean_grad[0]), float(args.mean_grad[1])),
        n_disp_total=args.n_disp_total,
        n_ell_bins=args.n_ell_bins,
        yaglom_samples=args.yaglom_samples,
        velocity_seed=args.velocity_seed,
        disp_seed=args.disp_seed,
        warm_cache=not args.no_warm_cache,
        verbose=not args.quiet,
    )

    meta = {
        "grid": config.grid,
        "peclet": config.peclet,
        "dtype": str(config.dtype),
        "integrator": config.integrator,
        "cfl": config.cfl,
        "t_end": config.t_end,
        "fft_threads": config.fft_threads,
        "mean_grad": config.mean_grad,
        "n_disp_total": config.n_disp_total,
        "n_ell_bins": config.n_ell_bins,
        "yaglom_samples": config.yaglom_samples,
        "velocity_seed": config.velocity_seed,
        "disp_seed": config.disp_seed,
        "warm_cache": config.warm_cache,
    }
    ensure_dir(out_dir)
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    api = ScalarAdvectionAPI(N=config.grid, L=1.0, dtype=config.dtype, warm_cache=config.warm_cache)
    api.set_fft_threads(config.fft_threads)

    velocity = maybe_reuse_velocity(args, out_dir)
    if velocity is None:
        ux, uy, _ = generate_divfree_field(
            N=config.grid,
            lam_min=3,
            lam_max=config.grid,
            slope=-5.0 / 3.0,
            wavelet="mexh",
            sparsity=0.0,
            seed=config.velocity_seed,
        )
    else:
        ux, uy = velocity

    ux = ux.astype(api.grid.dtype, copy=False)
    uy = uy.astype(api.grid.dtype, copy=False)

    velocity_dir = out_dir / "velocity"
    velocity_summary = save_velocity_diagnostics(
        velocity_dir,
        ux,
        uy,
        DEFAULT_ORDERS,
        config.n_ell_bins,
        config.n_disp_total,
        config.disp_seed,
    )

    run_cfg = RunConfig(
        mode="single",
        grid_size=config.grid,
        peclet_values=(config.peclet,),
        t_end=config.t_end,
        cfl=config.cfl,
        integrator=config.integrator,
        mean_grad=config.mean_grad,
        fft_threads=config.fft_threads,
        dtype=config.dtype,
        velocity_seed=config.velocity_seed,
        disp_seed=config.disp_seed,
        warm_cache=config.warm_cache,
        n_disp_total=config.n_disp_total,
        n_ell_bins=config.n_ell_bins,
        yaglom_samples=config.yaglom_samples,
        output_dir=out_dir,
        verbose=config.verbose,
    )

    result = run_scalar_case_single(
        api,
        ux,
        uy,
        config.peclet,
        out_dir,
        run_cfg,
    )

    npz_path = result.final_snapshot_path.with_suffix(".npz")
    theta_final = np.load(result.final_snapshot_path)
    np.savez_compressed(npz_path, theta=theta_final)

    summary = {
        "velocity_summary": velocity_summary,
        "scalar_run": {
            "peclet": result.peclet,
            "kappa": result.kappa,
            "edot": result.edot,
            "grad_sq_integral": result.grad_sq_integral,
            "n_steps": result.n_steps,
            "dt": result.dt,
            "theta_final_npy": str(result.final_snapshot_path),
            "theta_final_npz": str(npz_path),
            "structure_functions_path": str(result.structure_fn_path),
            "yaglom_path": str(result.yaglom_path),
            "yaglom_unit_slope_range": (
                list(result.yaglom_unit_slope_range) if result.yaglom_unit_slope_range else None
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    return result, out_dir


def run_scalar_case_single(
    api: ScalarAdvectionAPI,
    ux: np.ndarray,
    uy: np.ndarray,
    peclet: float,
    run_dir: Path,
    run_cfg: RunConfig,
) -> ScalarRunResult:
    pe_dir = run_dir / f"scalar_pe_{format_peclet(peclet)}"
    ensure_dir(pe_dir)

    summary_path = pe_dir / "scalar_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        theta_final = np.load(pe_dir / "theta_final.npy")
        scalar_structure_function_plot(
            api,
            theta_final,
            orders=DEFAULT_ORDERS,
            n_ell_bins=run_cfg.n_ell_bins,
            n_disp_total=run_cfg.n_disp_total,
            seed=run_cfg.disp_seed,
            fname=pe_dir / "scalar_structure_functions.png",
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
        if slope_one_fit is not None:
            data["yaglom_unit_slope_range"] = [
                float(slope_one_fit.xseg[0]),
                float(slope_one_fit.xseg[-1]),
            ]
        else:
            data["yaglom_unit_slope_range"] = None
        summary_path.write_text(json.dumps(data, indent=2, sort_keys=True))
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
                tuple(data.get("yaglom_unit_slope_range", []))
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

    theta_final, diagnostics = api.evolve_scalar(theta0, ux, uy, scalar_cfg, verbose=run_cfg.verbose)

    np.save(pe_dir / "theta_final.npy", theta_final)

    scalar_structure_function_plot(
        api,
        theta_final,
        orders=DEFAULT_ORDERS,
        n_ell_bins=run_cfg.n_ell_bins,
        n_disp_total=run_cfg.n_disp_total,
        seed=run_cfg.disp_seed,
        fname=pe_dir / "scalar_structure_functions.png",
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

    summary = {
        "peclet": peclet,
        "kappa": diagnostics.kappa,
        "n_steps": diagnostics.n_steps,
        "dt": diagnostics.dt,
        "grad_sq_integral": diagnostics.grad_sq_integral,
        "edot": diagnostics.kappa * diagnostics.grad_sq_integral,
        "yaglom_slope": float(slope) if slope is not None else None,
        "yaglom_unit_slope_range": (
            [
                float(slope_one_fit.xseg[0]),
                float(slope_one_fit.xseg[-1]),
            ]
            if slope_one_fit is not None
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return ScalarRunResult(
        peclet=peclet,
        kappa=diagnostics.kappa,
        edot=summary["edot"],
        grad_sq_integral=diagnostics.grad_sq_integral,
        n_steps=diagnostics.n_steps,
        dt=diagnostics.dt,
        final_snapshot_path=pe_dir / "theta_final.npy",
        structure_fn_path=pe_dir / "scalar_structure_functions.npz",
        yaglom_path=pe_dir / "yaglom_stats.npz",
        yaglom_unit_slope_range=(
            (float(slope_one_fit.xseg[0]), float(slope_one_fit.xseg[-1]))
            if slope_one_fit is not None
            else None
        ),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result, out_dir = run_single_simulation(args)

    if args.quiet:
        return 0

    print(f"\nRun complete in: {out_dir}")
    print(f"  Peclet: {result.peclet:.6g}")
    print(f"  kappa:  {result.kappa:.6g}")
    print(f"  steps:  {result.n_steps} (dt ≈ {result.dt:.3e})")
    if result.yaglom_unit_slope_range:
        lo, hi = result.yaglom_unit_slope_range
        print(f"  Yaglom slope=1 segment: ℓ/Δx in [{lo:.2f}, {hi:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
