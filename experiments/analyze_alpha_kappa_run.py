#!/usr/bin/env python3
"""Analyze a completed alpha/kappa simulation directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection.grid import SpectralGrid  # noqa: E402
from scalar_advection.structure import structure_functions  # noqa: E402
from scalar_advection.analysis_utils import (  # noqa: E402
    ensure_dir,
    plot_scalar_sf_with_slopes,
    plot_yaglom_with_slopes,
    save_theta_velocity_frames,
    sliding_log_slope_series,
    yaglom_statistics,
)

MEAN_GRAD = (1.0, 0.0)
DEFAULT_N_DISP = 4096
DEFAULT_N_ELL_BINS = 40
DEFAULT_ORDERS = (1, 2, 3, 4, 6, 8, 10)

@dataclass
class SnapshotTask:
    time: float
    snapshot_path: Path
    sf_plot_path: Path
    yaglom_plot_path: Path
    seed: int


@dataclass
class SnapshotResult:
    time: float
    sf_r: np.ndarray
    sf_S: np.ndarray
    y_r: np.ndarray
    y_vals: np.ndarray
    y_counts: np.ndarray
    epsilon: float
    lambda_t: float


_WORKER_STATE: dict | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze alpha/kappa simulation outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the simulation directory.")
    parser.add_argument("--n-workers", type=int, default=8, help="Worker processes for diagnostics.")
    parser.add_argument("--n-ell-bins", type=int, default=DEFAULT_N_ELL_BINS)
    parser.add_argument("--n-disp-total", type=int, default=DEFAULT_N_DISP)
    parser.add_argument("--yaglom-samples", type=int, default=8192)
    parser.add_argument("--disp-seed", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, help="Optional cap on snapshots to process.")
    return parser.parse_args(argv)


def load_manifest(run_dir: Path) -> List[dict]:
    manifest_path = run_dir / "fields" / "snapshots" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    return data.get("snapshots", [])


def _init_worker(
    velocity_npz: str,
    grid_size: int,
    domain_size: float,
    dtype_str: str,
    orders: Tuple[int, ...],
    n_ell_bins: int,
    n_disp_total: int,
    yaglom_samples: int,
    dx: float,
    L: float,
    kappa: float,
    frames_dir_str: str,
    bg_flat: np.ndarray,
    need_frames: bool,
) -> None:
    global _WORKER_STATE
    vel = np.load(velocity_npz, mmap_mode="r")
    dtype = np.float32 if dtype_str == "float32" else np.float64
    grid = SpectralGrid(N=grid_size, L=domain_size, dtype=np.float64)
    _WORKER_STATE = {
        "ux": vel["ux"],
        "uy": vel["uy"],
        "orders": orders,
        "n_ell_bins": n_ell_bins,
        "n_disp_total": n_disp_total,
        "yaglom_samples": yaglom_samples,
        "dx": dx,
        "L": L,
        "grid": grid,
        "kappa": float(kappa),
        "frames_dir": Path(frames_dir_str),
        "bg": bg_flat.reshape((grid_size, grid_size)),
        "need_frames": need_frames,
    }


def _compute_scalar_diss(theta: np.ndarray) -> Tuple[float, float]:
    assert _WORKER_STATE is not None
    grid: SpectralGrid = _WORKER_STATE["grid"]
    theta_hat = np.fft.fft2(theta.astype(np.float64, copy=False))
    theta_x = np.fft.ifft2(1j * grid.kx * theta_hat).real
    theta_y = np.fft.ifft2(1j * grid.ky * theta_hat).real
    grad_sq = float(np.mean(theta_x * theta_x + theta_y * theta_y))
    epsilon = 2.0 * _WORKER_STATE["kappa"] * grad_sq
    return epsilon, grad_sq


def _process_snapshot(task: SnapshotTask) -> SnapshotResult:
    assert _WORKER_STATE is not None
    data = np.load(task.snapshot_path)
    theta = data["theta"]

    # Generate frame in worker
    if _WORKER_STATE["need_frames"]:
        save_theta_velocity_frames(
            _WORKER_STATE["frames_dir"],
            np.array([task.time], dtype=float),
            [theta],
            _WORKER_STATE["bg"],
            _WORKER_STATE["ux"],
            _WORKER_STATE["uy"],
        )

    orders = _WORKER_STATE["orders"]
    sf = structure_functions(
        theta,
        orders=orders,
        n_ell_bins=_WORKER_STATE["n_ell_bins"],
        n_disp_total=_WORKER_STATE["n_disp_total"],
        seed=task.seed,
        use_fft_for_p2=True,
    )
    eps_val, grad_sq = _compute_scalar_diss(theta)
    theta_var = float(np.mean(theta * theta))
    lambda_t = float(np.sqrt(theta_var / grad_sq)) if grad_sq > 0 else float("nan")
    lambda_t_px = lambda_t / _WORKER_STATE["dx"] if grad_sq > 0 else None
    plot_scalar_sf_with_slopes(
        sf["r"],
        sf["S"],
        np.asarray(orders, dtype=float),
        task.sf_plot_path,
        title=f"Scalar SF (t={task.time:.4f})",
        lambda_t=lambda_t_px,
    )

    ydata = yaglom_statistics(
        _WORKER_STATE["ux"],
        _WORKER_STATE["uy"],
        theta,
        n_ell_bins=_WORKER_STATE["n_ell_bins"],
        n_disp_total=_WORKER_STATE["n_disp_total"],
        samples_per_disp=_WORKER_STATE["yaglom_samples"],
        seed=task.seed,
        dx=_WORKER_STATE["dx"],
        L=_WORKER_STATE["L"],
    )
    r_y = ydata["r"]
    y_vals = ydata["yaglom"]
    n = min(r_y.size, y_vals.size)
    r_y = r_y[:n]
    y_vals = y_vals[:n]
    y_counts = ydata["counts"][:n]
    plot_yaglom_with_slopes(r_y, y_vals, task.yaglom_plot_path)

    return SnapshotResult(
        time=task.time,
        sf_r=sf["r"],
        sf_S=sf["S"],
        y_r=r_y,
        y_vals=y_vals,
        y_counts=y_counts,
        epsilon=eps_val,
        lambda_t=lambda_t,
    )


def render_movie(frames_dir: Path, output_path: Path, fps: int = 16) -> None:
    pngs = sorted(frames_dir.glob("theta_u_t*.png"))
    if not pngs:
        return
    width, height = Image.open(pngs[0]).size
    scale_filter = None
    if (width % 2) or (height % 2):
        scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        "theta_u_t*.png",
    ]
    if scale_filter:
        cmd += ["-vf", scale_filter]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(frames_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        msg = f"{exc}."
        if stderr:
            msg += f" ffmpeg output:\n{stderr.strip()}"
        print(f"[warn] ffmpeg movie generation failed: {msg}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] ffmpeg movie generation failed: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    grid = int(config["grid"])
    dtype_str = config.get("dtype", "float32")
    domain_size = float(config.get("domain_size", 1.0))
    kappa = float(config["kappa"])

    manifest = load_manifest(run_dir)
    if args.max_snapshots:
        manifest = manifest[: args.max_snapshots]
    if not manifest:
        print("No snapshots found to analyze.")
        return 0

    analysis_dir = run_dir / "analysis"
    fields_dir = run_dir / "fields"
    frames_dir = analysis_dir / "theta_velocity_frames"
    sf_plot_dir = analysis_dir / "scalar_structure_functions"
    yaglom_plot_dir = analysis_dir / "yaglom"
    ensure_dir(frames_dir)
    ensure_dir(sf_plot_dir)
    ensure_dir(yaglom_plot_dir)

    velocity_npz = fields_dir / "velocity_fields.npz"
    vel = np.load(velocity_npz)
    ux = vel["ux"]
    uy = vel["uy"]
    spd = np.hypot(ux, uy)
    vmax = float(np.percentile(spd, 99.0))

    vel_sf = structure_functions(
        (ux, uy),
        orders=tuple(DEFAULT_ORDERS),
        n_ell_bins=args.n_ell_bins,
        n_disp_total=args.n_disp_total,
        use_fft_for_p2=True,
        seed=0,
        signed_longitudinal=False,
    )
    np.savez_compressed(analysis_dir / "velocity_structure_functions.npz", **vel_sf)
    plot_scalar_sf_with_slopes(
        vel_sf["r"],
        vel_sf["mag"],
        np.asarray(DEFAULT_ORDERS, dtype=float),
        analysis_dir / "velocity_structure_functions.png",
        title="Velocity structure functions",
    )

    x = np.linspace(-0.5 * domain_size, 0.5 * domain_size, grid, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    bg = MEAN_GRAD[0] * X + MEAN_GRAD[1] * Y

    generate_frames = not any(frames_dir.glob("theta_u_t*.png"))
    snapshot_tasks: List[SnapshotTask] = []
    for idx, entry in enumerate(manifest):
        snap_path = run_dir / entry["relative_path"]
        t = float(entry.get("time", idx))
        sf_plot = sf_plot_dir / f"scalar_sf_t{t:.4f}.png"
        yag_plot = yaglom_plot_dir / f"yaglom_t{t:.4f}.png"
        seed = args.disp_seed + idx
        snapshot_tasks.append(
            SnapshotTask(
                time=t,
                snapshot_path=snap_path,
                sf_plot_path=sf_plot,
                yaglom_plot_path=yag_plot,
                seed=seed,
            )
        )

    worker_args = (
        str(velocity_npz),
        grid,
        domain_size,
        dtype_str,
        tuple(DEFAULT_ORDERS),
        args.n_ell_bins,
        args.n_disp_total,
        args.yaglom_samples,
        domain_size / grid,
        domain_size,
        kappa,
        str(frames_dir),
        bg.ravel(),
        generate_frames,
    )

    workers = max(1, min(args.n_workers, len(snapshot_tasks)))
    results: List[SnapshotResult] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=worker_args) as pool:
        futures = {pool.submit(_process_snapshot, task): task.time for task in snapshot_tasks}
        for fut in as_completed(futures):
            results.append(fut.result())

    render_movie(frames_dir, analysis_dir / "theta_velocity_movie.mp4")

    results.sort(key=lambda r: r.time)
    if not results:
        print("No snapshot diagnostics were generated.")
        return 0

    times = np.array([res.time for res in results], dtype=float)
    sf_r = results[0].sf_r
    sf_data = np.stack([res.sf_S for res in results], axis=1)  # (orders, times, bins)
    y_r = results[0].y_r
    yag_data = np.stack([res.y_vals for res in results], axis=0)  # (times, bins)
    counts_arr = np.stack([res.y_counts for res in results], axis=0)
    eps = np.array([res.epsilon for res in results], dtype=float)
    lambda_t_arr = np.array([res.lambda_t for res in results], dtype=float)
    orders = np.asarray(DEFAULT_ORDERS, dtype=float)

    reduced = {
        "alpha": config["alpha"],
        "kappa": config["kappa"],
        "orders": orders,
        "times": times,
        "sf_r": sf_r,
        "scalar_structure_functions": sf_data,
        "yaglom_r": y_r,
        "yaglom": yag_data,
        "yaglom_counts": counts_arr,
        "dissipation": eps,
        "lambda_t": lambda_t_arr,
        "grid": grid,
        "dtype": dtype_str,
        "dx": domain_size / grid,
        "L": domain_size,
        "mean_grad": MEAN_GRAD,
        "n_ell_bins": args.n_ell_bins,
        "n_disp_total": args.n_disp_total,
        "yaglom_samples": args.yaglom_samples,
        "velocity_sf_r": vel_sf["r"],
        "velocity_sf_orders": vel_sf["orders"],
        "velocity_sf_mag": vel_sf["mag"],
        "velocity_sf_long": vel_sf.get("long"),
        "velocity_sf_tran": vel_sf.get("tran"),
    }
    out_npz = analysis_dir / "reduced_diagnostics.npz"
    np.savez_compressed(out_npz, **reduced)
    print(f"Saved reduced diagnostics to {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
