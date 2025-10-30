#!/usr/bin/env python3
"""
Minimal example: scalar advection with a time-evolving OU wavelet velocity.

This demonstrates how to use WaveletOUTemporalVelocity with the solver.
Optionally outputs series of velocity and scalar maps at interval tau/10.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import matplotlib

# Ensure headless plotting works
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scalar_advection.api import (
    ScalarAdvectionAPI,
    ScalarConfig,
    WaveletOUConfig,
    WaveletOUTemporalVelocity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OU wavelet velocity + scalar demo")
    p.add_argument("--N", type=int, default=256, help="Grid size (even)")
    p.add_argument("--t-end", type=float, default=0.25, help="Final time (advective units)")
    p.add_argument("--peclet", type=float, default=1e4, help="Scalar Peclet number")
    p.add_argument("--tau", type=float, default=0.1, help="OU correlation timescale")
    p.add_argument("--seed", type=int, default=1, help="Random seed for velocity")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--out", type=Path, default=Path("examples") / "ou_demo_output.npy")
    p.add_argument(
        "--series",
        action="store_true",
        help="Output velocity and scalar map series at interval tau/10",
    )
    p.add_argument(
        "--series-dir",
        type=Path,
        help="Directory to save series frames (defaults to OUT stem + '_series')",
    )
    p.add_argument(
        "--mean-grad",
        type=float,
        nargs=2,
        metavar=("Gx", "Gy"),
        default=(0.0, 0.0),
        help="Mean gradient forcing components. If nonzero and --mean-grad-ic is set, IC is zero.",
    )
    p.add_argument(
        "--mean-grad-ic",
        action="store_true",
        help="Use zero initial condition when mean gradient is nonzero (recommended).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    api = ScalarAdvectionAPI(N=args.N, L=1.0, dtype=dtype)

    vel_cfg = WaveletOUConfig(
        N=args.N,
        lam_min=4.0,
        lam_max=float(args.N),
        slope=-5.0 / 3.0,
        wavelet="mexh",
        sparsity=0.0,
        seed=args.seed,
        tau=float(args.tau),
        amp=1.0,
        dtype=dtype,
    )
    velocity = WaveletOUTemporalVelocity(vel_cfg)

    # Optional: produce velocity series at dt = tau/10, then reset before solve
    if args.series:
        dt_frame = float(args.tau) / 10.0
        nframes = int(math.floor(args.t_end / dt_frame)) + 1
        series_dir = args.series_dir or args.out.parent / f"{args.out.stem}_series"
        vdir = series_dir / "velocity"
        sdir = series_dir / "scalar"
        vdir.mkdir(parents=True, exist_ok=True)
        sdir.mkdir(parents=True, exist_ok=True)

        # Estimate a consistent color scale for |u| using amp
        vmax_u = 3.0 * float(vel_cfg.amp)
        vmin_u = 0.0

        def save_field_image(arr: np.ndarray, path: Path, *, cmap: str = "viridis", vmin=None, vmax=None, title: str | None = None):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=160, constrained_layout=True)
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        # Generate velocity snapshots
        print(f"Saving velocity frames to {vdir} (dt={dt_frame})")
        # Start from t=0 state
        vel_for_series = velocity
        # Ensure clean start
        vel_for_series.reset(seed=args.seed)
        tnow = 0.0
        for i in range(nframes):
            ux, uy = vel_for_series.get_velocity()
            speed = np.hypot(ux, uy)
            save_field_image(speed, vdir / f"speed_t{i:04d}.png", cmap="plasma", vmin=vmin_u, vmax=vmax_u, title=f"|u|, t={tnow:.3f}")
            if i < nframes - 1:
                vel_for_series.step(dt_frame)
                tnow += dt_frame

        # Reset to initial state for the actual scalar evolution
        velocity.reset(seed=args.seed)

    # Initial condition
    Gx, Gy = args.mean_grad
    if args.mean_grad_ic and (Gx != 0.0 or Gy != 0.0):
        theta0 = np.zeros((args.N, args.N), dtype=dtype)
    else:
        theta0 = api.circle_initial_condition(radius=0.25)

    # Scalar parameters
    s_cfg = ScalarConfig(
        peclet=float(args.peclet),
        t_end=float(args.t_end),
        integrator="etdrk4",
        cfl=0.5,
        mean_grad=(float(Gx), float(Gy)),
        output_frames=bool(args.series),
        frame_interval=(float(args.tau) / 10.0) if args.series else None,
    )

    theta_final, diag = api.evolve_scalar_time_varying(theta0, velocity, s_cfg, verbose=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, theta_final)
    print(f"Saved final scalar field to: {args.out}")

    # Save scalar frames if requested
    if args.series and diag.frames is not None and len(diag.frames) > 0:
        series_dir = args.series_dir or args.out.parent / f"{args.out.stem}_series"
        sdir = series_dir / "scalar"
        sdir.mkdir(parents=True, exist_ok=True)
        dt_frame = float(args.tau) / 10.0
        print(f"Saving scalar frames to {sdir} (dt={dt_frame})")
        vmin_s = float(np.min(diag.frames[0]))
        vmax_s = float(np.max(diag.frames[0]))
        # Keep dynamic range consistent across frames by expanding by 10%
        for frame in diag.frames:
            vmin_s = min(vmin_s, float(np.min(frame)))
            vmax_s = max(vmax_s, float(np.max(frame)))
        span = vmax_s - vmin_s
        vmin_s -= 0.05 * span
        vmax_s += 0.05 * span

        for i, frame in enumerate(diag.frames):
            tnow = i * dt_frame
            fig, ax = plt.subplots(figsize=(6, 6), dpi=160, constrained_layout=True)
            im = ax.imshow(frame, origin="lower", cmap="viridis", vmin=vmin_s, vmax=vmax_s)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"theta, t={tnow:.3f}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(sdir / f"theta_t{i:04d}.png", bbox_inches="tight")
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
