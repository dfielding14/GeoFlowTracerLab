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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resample_even_times(t: np.ndarray, y: np.ndarray, n: int, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    tg = np.linspace(t0, t1, n)
    yg = np.interp(tg, t, y)
    return tg, yg


def main() -> int:
    cfg = PerfConfig()

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
        mean_grad=(1.0, 0.0),
        save_every=None,
        output_frames=False,
        frame_interval=None,
        profile=True,
        profile_fft=True,
    )

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
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

