#!/usr/bin/env python3
"""
Fast test version of Edot_theta vs mu scaling.

Reduces resolution to 128^2 and scales Peclet accordingly:
  Pe ∈ {256, 512, 1024, 2048} for alphas {1/6, 1/3, 1/2, 2/3}.

Velocity: wavelet (mexh) with lam_min=8, lam_max=64, velocity_seed=1.
Defaults are set for speed (t_end=1.0). Aggregation logic matches the full
script; if no t>4 samples exist, median is taken over all times.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import ScalarAdvectionAPI  # noqa: E402
from scalar_advection.solver import ScalarConfig  # noqa: E402
from scalar_advection.velocity import generate_divfree_field  # noqa: E402
from scalar_advection.structure import structure_functions  # noqa: E402


@dataclass
class RunParams:
    N: int = 128
    L: float = 1.0
    dtype: np.dtype = np.float32
    t_end: float = 1.0
    cfl: float = 0.9
    integrator: str = "heun"
    outputs: int = 60
    lam_min: float = 8.0
    lam_max: float = 64.0
    wavelet: str = "mexh"
    velocity_seed: int = 1
    mean_grad: Tuple[float, float] = (1.0, 0.0)
    n_ell_bins: int = 24
    n_disp_total: int = 1024
    fft_threads: int = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAST Edot_theta vs mu scaling (N=128)")
    p.add_argument("--output-root", type=Path, default=Path("experimental_results") / "edot_mu_scaling_fast")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--t-end", type=float, default=1.0)
    p.add_argument("--integrator", choices=("rk4","etdrk4","heun"), default="heun")
    p.add_argument("--cfl", type=float, default=0.9)
    return p.parse_args(argv)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def choose_save_every(api: ScalarAdvectionAPI, ux: np.ndarray, uy: np.ndarray, peclet: float, t_end: float, cfl: float, target_outputs: int) -> int:
    urms = float(np.sqrt(np.mean(ux**2 + uy**2)))
    mu = urms * api.grid.L / peclet
    denom = float(np.max(np.abs(ux)) + np.max(np.abs(uy)) + 1e-14)
    dt_adv = cfl * api.grid.dx / denom
    dt_diff = cfl * api.grid.dx**2 / (4 * mu + 1e-14)
    dt_est = float(min(dt_adv, dt_diff))
    nsteps_est = int(np.ceil(t_end / dt_est))
    return max(1, int(round(nsteps_est / max(1, (target_outputs - 1)))))


def instantaneous_edot(api: ScalarAdvectionAPI, theta: np.ndarray, mu: float) -> float:
    val = api.scalar_dissipation(theta, mu)  # 2 mu <|grad theta|^2>
    return 0.5 * float(val)


def build_velocity(N: int, alpha: float, lam_min: float, lam_max: float, wavelet: str, seed: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    ux, uy, _ = generate_divfree_field(
        N=N,
        lam_min=lam_min,
        lam_max=lam_max,
        alpha=alpha,
        wavelet=wavelet,
        sparsity=0.0,
        seed=seed,
    )
    return ux.astype(dtype, copy=False), uy.astype(dtype, copy=False)


def save_theta_frames_and_movie(frames_dir: Path, times: np.ndarray, snapshots: List[np.ndarray], bg: np.ndarray, peclet: float) -> str | None:
    ensure_dir(frames_dir)
    levels = [-0.4, -0.2, 0.0, 0.2, 0.4]
    for idx, tnow in enumerate(times):
        if idx >= len(snapshots):
            break
        arr = np.nan_to_num(snapshots[idx] + bg, copy=False)
        fig, ax = plt.subplots(figsize=(5.0,5.0), dpi=140, constrained_layout=True)
        cs = ax.contour(arr, levels=levels, cmap="RdBu_r")
        ax.clabel(cs, fmt="%0.1f", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {tnow:.3f}")
        fig.savefig(frames_dir / f"theta_t{tnow:.4f}.png", bbox_inches="tight")
        plt.close(fig)
    try:
        movie = frames_dir.parent / f"evolution_Pe{int(peclet)}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "20", "-pattern_type", "glob", "-i", "*.png",
            "-s", "512x576", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(movie),
        ]
        subprocess.run(cmd, cwd=str(frames_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(movie)
    except Exception:
        return None


def sliding_log_slope_series(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size < 2:
        return np.array([]), np.array([])
    lx = np.log(x)
    ly = np.log(np.maximum(y, 1e-30))
    centers = np.sqrt(x[:-1] * x[1:])
    slopes = np.diff(ly) / np.diff(lx)
    return centers, slopes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = RunParams(dtype=(np.float32 if args.dtype == "float32" else np.float64), t_end=float(args.t_end), cfl=float(args.cfl), integrator=args.integrator, fft_threads=int(args.threads))

    out_root = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_root)

    api = ScalarAdvectionAPI(N=cfg.N, L=cfg.L, dtype=cfg.dtype, warm_cache=True)
    api.set_fft_threads(cfg.fft_threads)

    alphas = [1.0/6.0, 1.0/3.0, 1.0/2.0, 2.0/3.0]
    peclets = [256.0, 512.0, 1024.0, 2048.0]

    results: Dict[float, Dict[str, np.ndarray]] = {}

    for a in alphas:
        ux, uy = build_velocity(cfg.N, a, cfg.lam_min, cfg.lam_max, cfg.wavelet, cfg.velocity_seed, api.grid.dtype)
        alpha_dir = out_root / f"alpha_{a:.3f}"
        ensure_dir(alpha_dir)

        mus = []
        medians = []
        sigmas = []

        for pe in peclets:
            pe_dir = alpha_dir / f"pe_{int(pe)}"
            ensure_dir(pe_dir)

            theta0 = np.zeros((cfg.N, cfg.N), dtype=api.grid.dtype)
            save_every = choose_save_every(api, ux, uy, pe, cfg.t_end, cfg.cfl, cfg.outputs)
            sconf = ScalarConfig(
                peclet=pe,
                t_end=cfg.t_end,
                cfl=cfg.cfl,
                integrator=cfg.integrator,
                mean_grad=cfg.mean_grad,
                save_every=save_every,
                output_frames=False,
                frame_interval=None,
            )
            theta_final, diag = api.solver.evolve(theta0, ux, uy, sconf, verbose=True)

            mu = float(diag.kappa)
            mus.append(mu)
            np.savez_compressed(pe_dir / "theta_final.npz", theta=theta_final)

            sf = structure_functions(theta_final, orders=(1,2), n_ell_bins=cfg.n_ell_bins, n_disp_total=cfg.n_disp_total, seed=0, use_fft_for_p2=True)
            np.savez_compressed(pe_dir / "theta_structure_functions.npz", **sf)

            x = np.linspace(-cfg.L/2, cfg.L/2, cfg.N, endpoint=False)
            y = np.linspace(-cfg.L/2, cfg.L/2, cfg.N, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing="xy")
            bg = cfg.mean_grad[0]*X + cfg.mean_grad[1]*Y
            _ = save_theta_frames_and_movie(pe_dir / "theta_frames", np.array(diag.times), diag.snapshots, bg, pe)

            times = np.array(diag.times, dtype=float)
            edots = []
            etas = []
            for th in diag.snapshots:
                edots.append(instantaneous_edot(api, th, mu))
                etas.append(float(np.mean(th*th)))
            edots = np.array(edots)
            etas = np.array(etas)
            ratio = np.divide(edots, etas, out=np.zeros_like(edots), where=etas>0)
            np.savez_compressed(pe_dir / "edot_timeseries.npz", t=times, edot=edots, Etheta=etas, ratio=ratio)

            late = times > 4.0
            if np.any(late):
                med = float(np.median(edots[late]))
                sig = float(np.std(edots[late]))
            else:
                med = float(np.median(edots)) if edots.size else float("nan")
                sig = float(np.std(edots)) if edots.size else float("nan")
            medians.append(med)
            sigmas.append(sig)

            with open(pe_dir / "summary.json", "w") as f:
                json.dump({
                    "alpha": a,
                    "peclet": pe,
                    "mu": mu,
                    "median_edot": med,
                    "sigma_edot": sig,
                    "n_steps": diag.n_steps,
                    "dt": diag.dt,
                }, f, indent=2, sort_keys=True)

        mus = np.array(mus)
        medians = np.array(medians)
        sigmas = np.array(sigmas)
        o = np.argsort(mus)
        results[a] = {"mu": mus[o], "median": medians[o], "sigma": sigmas[o]}

    colors = plt.cm.tab10.colors
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.2, 5.4), dpi=150, sharex=True, gridspec_kw={"height_ratios":[2.2, 1.0], "hspace":0.05})
    for j, a in enumerate(sorted(results.keys())):
        mu = results[a]["mu"]; med = results[a]["median"]; sig = results[a]["sigma"]
        ax_top.errorbar(mu, med, yerr=sig, fmt="o-", lw=1.4, color=colors[j%len(colors)], label=f"alpha={a:g}")
        centers, slopes = sliding_log_slope_series(mu, med)
        if centers.size:
            ax_bot.semilogx(centers, slopes, "-", lw=1.4, color=colors[j%len(colors)])
    ax_top.set_xscale("log"); ax_top.set_yscale("log")
    ax_top.set_ylabel(r"median $\dot{E}_\theta$")
    ax_top.grid(True, which="both", ls=":", lw=0.6)
    ax_top.legend(frameon=False)
    ax_bot.set_xlabel(r"$\mu$")
    ax_bot.set_ylabel(r"$d\log \dot{E}_\theta / d\log \mu$")
    ax_bot.grid(True, which="both", ls=":", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_root / "edot_vs_mu_fast.png", bbox_inches="tight")
    plt.close(fig)

    with open(out_root / "aggregate.json", "w") as f:
        json.dump({f"alpha_{a:g}": {k: v.tolist() for k,v in results[a].items()} for a in results}, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
