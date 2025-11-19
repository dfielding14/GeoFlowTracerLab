#!/usr/bin/env python3
"""Run a single (alpha, kappa) scalar simulation and stream snapshots to disk."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import shlex
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import ScalarAdvectionAPI, ScalarConfig, generate_divfree_field  # noqa: E402

from scalar_advection.analysis_utils import ensure_dir, plot_and_save_dissipation, save_theta_velocity_frames  # noqa: E402

MEAN_GRAD = (1.0, 0.0)
DEFAULT_OUTPUT_ROOT = Path("experimental_results") / "alpha_kappa_runs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-node (alpha, kappa) scalar simulation with streamed outputs",
    )
    parser.add_argument("--alpha", type=float, required=True, help="Velocity structure-function exponent.")
    parser.add_argument("--kappa", type=float, required=True, help="Scalar diffusivity (mu).")
    parser.add_argument("--grid", type=int, default=1024, help="Grid resolution N (default: 1024).")
    parser.add_argument("--t-end", type=float, default=0.25, help="Final simulation time (default: 0.25).")
    parser.add_argument(
        "--n-save",
        type=int,
        default=12,
        help="Target number of snapshots between 0 and t_end (excluding t=0).",
    )
    parser.add_argument(
        "--dt-save",
        type=float,
        help="Optional fixed time spacing between saved snapshots; overrides --n-save when provided.",
    )
    parser.add_argument("--cfl", type=float, default=0.7, help="CFL number for adaptive dt (default: 0.7).")
    parser.add_argument(
        "--integrator",
        choices=("rk4", "etdrk4", "heun"),
        default="rk4",
        help="Scalar integrator (default: rk4).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Solver dtype for real-space arrays.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Worker/process count (also used for FFT threads, default: 8).",
    )
    parser.add_argument("--velocity-seed", type=int, default=1, help="Seed for wavelet velocity generation.")
    parser.add_argument(
        "--velocity-urms",
        type=float,
        default=1.0,
        help="Target RMS speed for the velocity field (default: 1.0). Set <=0 to disable rescaling.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for outputs.")
    parser.add_argument("--tag", type=str, help="Optional tag appended to the output directory name.")
    parser.add_argument("--lam-min", type=float, default=2.0, help="Smallest wavelet scale (default: 2).")
    parser.add_argument("--lam-max", type=float, help="Largest wavelet scale (default: equals grid size).")
    parser.add_argument("--no-warm-cache", action="store_true", help="Skip FFT plan warm-up.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout progress prints.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Number of integration steps between progress log entries (default: 100).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional explicit log file path (default: <run-dir>/run.log).",
    )
    parser.add_argument("--velocity-file", type=Path, help="Existing velocity_fields.npz to reuse.")
    parser.add_argument("--restart-run", type=Path, help="Existing run directory to restart from.")
    parser.add_argument(
        "--restart-snapshot",
        type=Path,
        help="Explicit theta snapshot (.npz) used as the initial condition when restarting.",
    )
    parser.add_argument(
        "--restart-time",
        type=float,
        help="Physical time corresponding to the restart snapshot (overrides filename/manifest).",
    )
    return parser.parse_args(argv)


def format_kappa(kappa: float) -> str:
    exponent = math.log10(kappa)
    if abs(exponent - round(exponent)) < 1e-6:
        return f"1e{exponent:.0f}"
    return f"1e{exponent:.2f}"


def estimate_base_timestep(
    ux: np.ndarray,
    uy: np.ndarray,
    grid_dx: float,
    cfl: float,
    kappa: float,
    t_end: float,
) -> Tuple[float, int]:
    speed = np.hypot(ux, uy)
    umax = float(np.max(speed))
    denom = umax if umax > 0 else 1.0
    dt_adv = cfl * grid_dx / denom
    dt_diff = cfl * grid_dx**2 / max(4.0 * kappa, 1e-12)
    dt_est = min(dt_adv, dt_diff)
    steps = max(1, int(math.ceil(t_end / max(dt_est, 1e-12))))
    return dt_est, steps


def determine_save_stride(
    ux: np.ndarray,
    uy: np.ndarray,
    grid_dx: float,
    cfl: float,
    kappa: float,
    t_end: float,
    n_save: int,
    dt_save: float | None,
) -> int:
    dt_est, steps = estimate_base_timestep(ux, uy, grid_dx, cfl, kappa, t_end)
    if dt_save is not None and dt_save > 0:
        stride = max(1, int(round(dt_save / max(dt_est, 1e-12))))
    else:
        if n_save <= 0:
            return steps
        stride = max(1, steps // n_save)
    return stride


def velocity_diagnostics(ux: np.ndarray, uy: np.ndarray, fields_dir: Path, analysis_dir: Path) -> dict:
    ensure_dir(fields_dir)
    ensure_dir(analysis_dir)
    speed = np.hypot(ux, uy)
    np.savez_compressed(fields_dir / "velocity_fields.npz", ux=ux, uy=uy, speed=speed)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=220, constrained_layout=True)
    im = ax.imshow(speed, origin="lower", cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")
    fig.savefig(analysis_dir / "velocity_magnitude.png", bbox_inches="tight")
    plt.close(fig)

    urms = float(np.sqrt(np.mean(ux**2 + uy**2)))
    summary = {"urms": urms}
    (analysis_dir / "velocity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_time_from_snapshot(name: str) -> float:
    try:
        suffix = name.split("_t")[-1]
        return float(suffix.replace(".npz", "").replace(".npy", ""))
    except Exception:
        return float("nan")


def load_manifest_entries(snapshot_dir: Path) -> List[dict]:
    manifest_file = snapshot_dir / "manifest.json"
    if manifest_file.exists():
        data = json.loads(manifest_file.read_text())
        return data.get("snapshots", [])
    return []


def latest_snapshot_in_run(run_dir: Path) -> Tuple[Path | None, float | None]:
    snapshot_dir = run_dir / "fields" / "snapshots"
    manifest = load_manifest_entries(snapshot_dir)
    if manifest:
        entry = manifest[-1]
        rel = entry.get("relative_path")
        candidate = run_dir / rel if rel else None
        time_val = entry.get("time")
        return candidate, float(time_val) if time_val is not None else None
    files = sorted((run_dir / "fields" / "snapshots").glob("theta_*_t*.npz"))
    if files:
        return files[-1], parse_time_from_snapshot(files[-1].name)
    files = sorted((run_dir / "fields" / "snapshots").glob("theta_*_t*.npy"))
    if files:
        return files[-1], parse_time_from_snapshot(files[-1].name)
    return None, None


def time_from_manifest(snapshot_path: Path, manifest: List[dict], run_dir: Path) -> float | None:
    try:
        rel = snapshot_path.relative_to(run_dir)
    except ValueError:
        rel = snapshot_path.name
    for entry in manifest:
        entry_path = entry.get("relative_path")
        if entry_path and Path(entry_path) == rel:
            time_val = entry.get("time")
            if time_val is not None:
                return float(time_val)
    return None


def load_velocity_from_file(path: Path, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["ux"].astype(dtype, copy=False), data["uy"].astype(dtype, copy=False)

def snapshot_monitor_worker(
    stop_event: threading.Event,
    snapshot_dir: Path,
    run_dir: Path,
    frames_dir: Path,
    bg: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    manifest: List[dict],
) -> None:
    processed: set[str] = set()
    while True:
        new_found = False
        files = sorted(snapshot_dir.glob("theta_*.npz")) + sorted(snapshot_dir.glob("theta_*.npy"))
        for fname in files:
            key = Path(fname.name).stem
            if key in processed:
                continue
            target_path = fname
            if fname.suffix == ".npy":
                try:
                    arr_raw = np.load(fname)
                except Exception:
                    continue
                target_path = fname.with_suffix(".npz")
                np.savez_compressed(target_path, theta=arr_raw)
                os.remove(fname)
                data = {"theta": arr_raw}
            else:
                try:
                    data = np.load(fname)
                except Exception:
                    continue
            arr = data["theta"]
            tval = parse_time_from_snapshot(name=fname.name)
            save_theta_velocity_frames(
                frames_dir,
                np.array([tval], dtype=float),
                [arr],
                bg,
                ux,
                uy,
            )
            manifest.append(
                {
                    "index": len(manifest) + 1,
                    "time": tval,
                    "relative_path": str(target_path.relative_to(run_dir)),
                }
            )
            processed.add(key)
            new_found = True
        if stop_event.is_set() and not new_found:
            break
        time.sleep(0.5)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    lam_max = float(args.lam_max) if args.lam_max is not None else float(args.grid)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    kappa_label = format_kappa(args.kappa)
    alpha_label = f"{args.alpha:.3f}".rstrip("0").rstrip(".")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = args.output_root / f"alpha_{alpha_label}_kappa_{kappa_label}_{timestamp}{tag}"
    fields_dir = out_dir / "fields"
    analysis_dir = out_dir / "analysis"
    velocity_analysis_dir = analysis_dir / "velocity"
    ensure_dir(out_dir)
    ensure_dir(fields_dir)
    ensure_dir(analysis_dir)
    ensure_dir(velocity_analysis_dir)

    log_path = args.log_file if args.log_file else out_dir / "run.log"
    log_path = log_path.expanduser()
    if not log_path.is_absolute():
        log_path = log_path.resolve()
    ensure_dir(log_path.parent)
    log_stream = log_path.open("a", buffering=1, encoding="utf-8")

    def log(msg: str, *, stdout: bool = False) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        log_stream.write(line + "\n")
        log_stream.flush()
        if stdout and not args.quiet:
            print(line, flush=True)

    api = ScalarAdvectionAPI(N=args.grid, L=1.0, dtype=dtype, warm_cache=not args.no_warm_cache)
    api.set_fft_threads(args.n_workers)
    log(
        "Run configuration: "
        f"alpha={args.alpha:.6g}, kappa={args.kappa:.6g}, grid={args.grid}, "
        f"t_end={args.t_end}, n_save={args.n_save}, dt_save={args.dt_save}, "
        f"workers={args.n_workers}, tag={args.tag or 'none'}",
    )
    log(f"FFT worker threads set to {args.n_workers}")

    restart_run_dir = args.restart_run.resolve() if args.restart_run else None

    velocity_path = args.velocity_file
    if velocity_path is None and restart_run_dir is not None:
        candidate = restart_run_dir / "velocity" / "velocity_fields.npz"
        if candidate.exists():
            velocity_path = candidate
    if velocity_path:
        ux, uy = load_velocity_from_file(velocity_path, dtype)
        log(f"Loaded velocity field from {velocity_path}")
    else:
        ux, uy, _ = generate_divfree_field(
            N=args.grid,
            lam_min=args.lam_min,
            lam_max=lam_max,
            alpha=args.alpha,
            wavelet="mexh",
            sparsity=0.0,
            seed=args.velocity_seed,
        )
        ux = ux.astype(dtype, copy=False)
        uy = uy.astype(dtype, copy=False)
        log(
            "Generated new velocity field via wavelets "
            f"(lam_min={args.lam_min}, lam_max={lam_max}, seed={args.velocity_seed})"
        )
    if args.velocity_urms > 0:
        current = float(np.sqrt(np.mean(ux**2 + uy**2)))
        if current > 0:
            scale = float(args.velocity_urms) / current
            ux *= scale
            uy *= scale
            log(f"Rescaled velocity field to target RMS={args.velocity_urms} (current={current:.6f})")

    velocity_summary = velocity_diagnostics(ux, uy, fields_dir, velocity_analysis_dir)
    log(f"Wrote velocity diagnostics to {velocity_analysis_dir}")

    manifest_entries: List[dict] = []
    if restart_run_dir:
        manifest_entries = load_manifest_entries(restart_run_dir / "fields" / "snapshots")
        log(f"Loaded {len(manifest_entries)} manifest entries from {restart_run_dir}")

    restart_snapshot_path = args.restart_snapshot
    if restart_snapshot_path and restart_run_dir and not restart_snapshot_path.is_absolute():
        restart_snapshot_path = (restart_run_dir / restart_snapshot_path).resolve()
    restart_time = args.restart_time
    if restart_snapshot_path is None and restart_run_dir:
        auto_snapshot, auto_time = latest_snapshot_in_run(restart_run_dir)
        restart_snapshot_path = auto_snapshot
        if restart_time is None:
            restart_time = auto_time

    snapshot_source: Path | None = None
    if restart_snapshot_path is not None:
        if not restart_snapshot_path.exists():
            raise FileNotFoundError(f"Restart snapshot not found: {restart_snapshot_path}")
        data = np.load(restart_snapshot_path)
        theta0 = data["theta"].astype(dtype, copy=False)
        snapshot_source = restart_snapshot_path
        if restart_time is None:
            restart_time = time_from_manifest(restart_snapshot_path, manifest_entries, restart_run_dir) if restart_run_dir else None
        if restart_time is None:
            restart_time = parse_time_from_snapshot(restart_snapshot_path.name)
        if restart_time is None:
            raise ValueError("Unable to infer restart time; provide --restart-time.")
        t_start = float(restart_time)
        log(f"Restarting from snapshot {restart_snapshot_path} at t={t_start:.6f}")
    else:
        theta0 = np.zeros((args.grid, args.grid), dtype=dtype)
        t_start = 0.0
        log("No restart provided; starting from zero-mean-gradient initial condition.")
    remaining_time = args.t_end - t_start
    if remaining_time <= 0:
        raise ValueError("Requested t_end must be greater than the restart time.")

    save_stride = determine_save_stride(
        ux,
        uy,
        api.grid.dx,
        args.cfl,
        args.kappa,
        remaining_time,
        args.n_save,
        args.dt_save,
    )

    snapshot_dir = fields_dir / "snapshots"
    ensure_dir(snapshot_dir)
    frames_dir = analysis_dir / "theta_velocity_frames"
    ensure_dir(frames_dir)
    base_scalar_cfg = ScalarConfig(
        kappa=float(args.kappa),
        mean_grad=MEAN_GRAD,
        t_end=remaining_time,
        cfl=args.cfl,
        integrator=args.integrator,
        output_frames=False,
        save_to_disk=True,
        save_dir=str(snapshot_dir),
        t_start=t_start,
    )

    dt_actual = None
    nsteps_actual = None
    try:
        dt_actual, nsteps_actual, _, _ = api.solver._resolve_time_controls(theta0, ux, uy, base_scalar_cfg)  # type: ignore[attr-defined]
    except Exception:
        pass

    if dt_actual is not None and nsteps_actual is not None and nsteps_actual > 0:
        log(f"Resolved solver controls: dt={dt_actual:.6e}, steps={nsteps_actual}")
        if args.dt_save and args.dt_save > 0:
            save_stride = max(1, int(round(args.dt_save / max(dt_actual, 1e-12))))
        elif args.n_save > 0:
            save_stride = max(1, nsteps_actual // args.n_save)
        else:
            save_stride = 1
    else:
        save_stride = determine_save_stride(
            ux,
            uy,
            api.grid.dx,
            args.cfl,
            args.kappa,
            args.t_end,
            args.n_save,
            args.dt_save,
        )
    log(
        f"Snapshot stride set to every {save_stride} steps "
        f"(target n_save={args.n_save}, dt_save={args.dt_save})"
    )

    scalar_cfg = replace(
        base_scalar_cfg,
        save_every=save_stride,
        log_stream=log_stream,
        log_to_stdout=not args.quiet,
        progress_interval=max(1, args.progress_interval),
    )

    x = np.linspace(-0.5, 0.5, args.grid, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    bg = MEAN_GRAD[0] * X + MEAN_GRAD[1] * Y
    stop_event = threading.Event()
    manifest: List[dict] = []
    converter = threading.Thread(
        target=snapshot_monitor_worker,
        args=(stop_event, snapshot_dir, out_dir, frames_dir, bg, ux, uy, manifest),
        daemon=True,
    )
    converter.start()

    log("Starting scalar evolution loop.")
    start_wall = time.time()
    theta_final, diagnostics = api.evolve_scalar(theta0, ux, uy, scalar_cfg, verbose=True)
    stop_event.set()
    converter.join()
    np.savez_compressed(fields_dir / "theta_final.npz", theta=theta_final)
    elapsed_wall = time.time() - start_wall
    log(f"Scalar evolution finished in {elapsed_wall/3600:.3f} h of wall time.")

    meta = {
        "alpha": args.alpha,
        "kappa": args.kappa,
        "grid": args.grid,
        "domain_size": 1.0,
        "lam_min": args.lam_min,
        "lam_max": lam_max,
        "dtype": args.dtype,
        "t_start": t_start,
        "t_end": args.t_end,
        "cfl": args.cfl,
        "integrator": args.integrator,
        "n_save_target": args.n_save,
        "dt_save": args.dt_save,
        "save_stride": save_stride,
        "n_workers": args.n_workers,
        "velocity_seed": args.velocity_seed,
        "velocity_file": str(velocity_path) if velocity_path else None,
        "restart_run": str(restart_run_dir) if restart_run_dir else None,
        "restart_snapshot": str(snapshot_source) if snapshot_source else None,
        "log_file": str(log_path),
    }
    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    log(f"Wrote run configuration to {config_path}")

    manifest_path = snapshot_dir / "manifest.json"
    if manifest:
        manifest_path.write_text(json.dumps({"snapshots": manifest}, indent=2))
        log(f"Wrote snapshot manifest with {len(manifest)} entries to {manifest_path}")
    else:
        manifest_path.write_text(json.dumps({"snapshots": []}, indent=2))
        log("[warn] No snapshots were saved; consider adjusting --n-save or --dt-save.", stdout=not args.quiet)

    plot_and_save_dissipation(
        analysis_dir,
        diagnostics.times_ts,
        diagnostics.grad_sq_ts,
        diagnostics.dissipation_ts,
        diagnostics.kappa,
    )
    log("Wrote dissipation diagnostics.")

    summary = {
        "alpha": args.alpha,
        "kappa": args.kappa,
        "grid": args.grid,
        "t_start": t_start,
        "t_end": args.t_end,
        "n_snapshots": len(manifest),
        "snapshot_manifest": "fields/snapshots/manifest.json",
        "velocity_summary": velocity_summary,
        "log_file": str(log_path),
    }
    summary_path = analysis_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log(f"Wrote simulation summary to {summary_path}")
    analyze_cmd = [
        "python",
        "experiments/analyze_alpha_kappa_run.py",
        "--run-dir",
        str(out_dir),
        "--n-workers",
        str(args.n_workers),
    ]
    analyze_cmd_str = " ".join(shlex.quote(part) for part in analyze_cmd)
    log(
        f"Simulation complete. Saved {len(manifest)} snapshots. "
        f"Analyze with: {analyze_cmd_str}",
    )
    if not args.quiet:
        print("Simulation complete.")
        print(f"  Run directory: {out_dir}")
        print(f"  Log file: {log_path}")
        print("  Analyze with:")
        print(f"    {analyze_cmd_str}")
    log_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
