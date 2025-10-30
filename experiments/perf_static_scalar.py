#!/usr/bin/env python3
"""
Performance sanity run (static velocity): N=1024, Pe=8192, t_end=1.0.

Generates a static divergence-free velocity, runs scalar evolution, records
dissipation over time, and produces a plot with 20 equally spaced samples in
time between t=0 and t=1. Also enables lightweight profiling.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import ScalarAdvectionAPI, ScalarConfig, VelocityConfig  # noqa: E402


@dataclass
class PerfConfig:
    N: int = 1024
    L: float = 1.0
    dtype: np.dtype = np.float32
    alpha: float = 1.0 / 3.0
    peclet: float = 8192.0
    t_end: float = 1.0
    cfl: float = 0.6
    integrator: str = "rk4"
    fft_threads: int = 8
    velocity_seed: int = 42
    output_root: Path = Path("experimental_results") / "perf_runs"
    outputs: int = 20
    mean_grad: tuple[float, float] = (1.0, 0.0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resample_even_times(t: np.ndarray, y: np.ndarray, n: int, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    tg = np.linspace(t0, t1, n)
    yg = np.interp(tg, t, y)
    return tg, yg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static velocity perf run with theta and dissipation outputs")
    p.add_argument("--N", type=int, default=1024, help="Grid size (default: 1024)")
    p.add_argument("--peclet", type=float, default=8192.0, help="Peclet number (default: 8192)")
    p.add_argument("--alpha", type=float, default=1.0/3.0, help="Spatial SF exponent alpha (default: 1/3)")
    p.add_argument("--integrator", choices=("rk4","etdrk4","heun"), default="rk4", help="Time integrator")
    p.add_argument("--t-end", type=float, default=1.0, help="End time (default: 1.0)")
    p.add_argument("--cfl", type=float, default=0.6, help="CFL for adaptive dt (default: 0.6)")
    p.add_argument("--outputs", type=int, default=20, help="# evenly spaced outputs between 0 and t_end")
    p.add_argument("--mean-grad", type=float, nargs=2, default=(1.0,0.0), metavar=("Gx","Gy"), help="Mean gradient forcing")
    p.add_argument("--dtype", choices=("float32","float64"), default="float32", help="Real dtype (default: float32)")
    p.add_argument("--fft-threads", type=int, default=8, help="FFTW threads (default: 8)")
    p.add_argument("--velocity-seed", type=int, default=42, help="Velocity RNG seed")
    p.add_argument("--output-root", type=Path, default=Path("experimental_results")/"perf_runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PerfConfig(
        N=args.N,
        L=1.0,
        dtype=(np.float32 if args.dtype=="float32" else np.float64),
        alpha=float(args.alpha),
        peclet=float(args.peclet),
        t_end=float(args.t_end),
        cfl=float(args.cfl),
        integrator=args.integrator,
        fft_threads=int(args.fft_threads),
        velocity_seed=int(args.velocity_seed),
        output_root=args.output_root,
        outputs=int(args.outputs),
        mean_grad=(float(args.mean_grad[0]), float(args.mean_grad[1])),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.output_root / f"static_N{cfg.N}_Pe{cfg.peclet:g}_{timestamp}"
    ensure_dir(out_dir)

    # Initialize API and FFT threads
    api = ScalarAdvectionAPI(N=cfg.N, L=cfg.L, dtype=cfg.dtype, warm_cache=True)
    api.set_fft_threads(cfg.fft_threads)

    # Generate static velocity field (alpha-based)
    vel_cfg = VelocityConfig(alpha=cfg.alpha, urms=1.0, seed=cfg.velocity_seed)
    ux, uy = api.generate_velocity(vel_cfg)

    # Scalar evolution from mean-gradient forcing only (zero initial perturbation)
    theta0 = np.zeros((cfg.N, cfg.N), dtype=api.grid.dtype)
    sim_cfg = ScalarConfig(
        peclet=cfg.peclet,
        t_end=cfg.t_end,
        cfl=cfg.cfl,
        integrator=cfg.integrator,
        mean_grad=cfg.mean_grad,
        save_every=None,
        output_frames=False,
        frame_interval=None,
        profile=True,
        profile_fft=True,
    )

    # Decide snapshot cadence to get ~outputs images including t=0, evenly spaced.
    # Replicate dt and nsteps calculation to choose save_every.
    urms = float(np.sqrt(np.mean(ux**2 + uy**2)))
    kappa = urms * api.grid.L / cfg.peclet
    denom = float(np.max(np.abs(ux)) + np.max(np.abs(uy)) + 1e-14)
    dt_adv = cfg.cfl * api.grid.dx / denom
    dt_diff = cfg.cfl * api.grid.dx**2 / (4 * kappa + 1e-14)
    dt_est = float(min(dt_adv, dt_diff))
    nsteps_est = int(np.ceil(cfg.t_end / dt_est))
    dt_adj = cfg.t_end / nsteps_est
    save_every = max(1, int(round(nsteps_est / max(1, (cfg.outputs - 1)))))

    sim_cfg.save_every = save_every
    theta_final, diag = api.solver.evolve(theta0, ux, uy, sim_cfg, verbose=True)

    # Dissipation over time (time series from diagnostics)
    t_ts = diag.times_ts
    eps_ts = diag.dissipation_ts
    if t_ts.size == 0 or eps_ts.size == 0:
        # Fallback: compute at end only
        t_ts = np.array([0.0, sim_cfg.t_end], dtype=float)
        e0 = api.scalar_dissipation(theta0, diag.kappa)
        e1 = api.scalar_dissipation(theta_final, diag.kappa)
        eps_ts = np.array([e0, e1], dtype=float)

    t_even, eps_even = resample_even_times(t_ts, eps_ts, cfg.outputs, 0.0, sim_cfg.t_end)

    # Plot dissipation over time (even samples)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(t_even, eps_even, "o-", lw=1.8)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\epsilon_\theta(t) = 2\,\kappa\,\langle |\nabla\theta|^2 \rangle$")
    ax.grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "dissipation_over_time.png", bbox_inches="tight")
    plt.close(fig)

    # Produce theta images (mean gradient added back) at each saved snapshot
    frames_dir = out_dir / "theta_frames"
    ensure_dir(frames_dir)
    # Build background gradient field
    x = np.linspace(-cfg.L/2, cfg.L/2, cfg.N, endpoint=False)
    y = np.linspace(-cfg.L/2, cfg.L/2, cfg.N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Gx, Gy = cfg.mean_grad
    bg = Gx*X + Gy*Y
    # Determine consistent color scale across frames
    if diag.snapshots:
        vals = []
        for th in diag.snapshots:
            vals.append(np.abs(th + bg))
        vmax = float(np.percentile(np.stack(vals), 99.0))
    else:
        vmax = float(np.percentile(np.abs(theta_final + bg), 99.0))
    # Save images
    for idx, tnow in enumerate(diag.times):
        if idx >= len(diag.snapshots):
            break
        th = diag.snapshots[idx]
        arr = th + bg
        fig, ax = plt.subplots(figsize=(6.0,6.0), dpi=160, constrained_layout=True)
        im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\theta$ (with mean gradient)")
        fig.suptitle(f"t = {tnow:.3f}")
        fig.savefig(frames_dir / f"theta_t{tnow:.4f}.png", bbox_inches="tight")
        plt.close(fig)

    # Persist data and summary
    np.savez_compressed(out_dir / "dissipation_series.npz", t=t_ts, epsilon=eps_ts, t_even=t_even, epsilon_even=eps_even)
    summary = {
        "N": cfg.N,
        "L": cfg.L,
        "dtype": str(cfg.dtype),
        "alpha": cfg.alpha,
        "peclet": cfg.peclet,
        "integrator": cfg.integrator,
        "cfl": cfg.cfl,
        "fft_threads": cfg.fft_threads,
        "velocity_seed": cfg.velocity_seed,
        "kappa": diag.kappa,
        "dt": diag.dt,
        "n_steps": diag.n_steps,
        "profile": getattr(diag, "profile", {}),
        "outputs": cfg.outputs,
        "mean_grad": cfg.mean_grad,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
