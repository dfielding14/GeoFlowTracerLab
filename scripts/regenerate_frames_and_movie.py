#!/usr/bin/env python
"""Regenerate theta_velocity_frames with correct cmasher colormaps and create movie.

Usage:
    python scripts/regenerate_frames_and_movie.py --run-dir <path> --movie-output-dir <path>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cmasher as cmr
except ImportError:
    raise ImportError("cmasher is required. Install with: pip install cmasher")

MEAN_GRAD = (1.0, 0.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate frames with correct colormaps")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to run directory")
    parser.add_argument("--movie-output-dir", type=Path, required=True, help="Directory for movie output")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for movie")
    parser.add_argument("--delete-old", action="store_true", default=True, help="Delete old frames first")
    return parser.parse_args(argv)


def load_snapshots_and_times(snapshots_dir: Path) -> tuple[list[np.ndarray], list[float]]:
    """Load all theta snapshots and extract times from filenames."""
    manifest_path = snapshots_dir / "manifest.json"
    if manifest_path.exists():
        print(f"  Found manifest at: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Handle both formats: list or dict with "snapshots" key
        if isinstance(manifest, dict) and "snapshots" in manifest:
            entries = manifest["snapshots"]
            print(f"  Manifest format: dict with 'snapshots' key ({len(entries)} entries)")
        elif isinstance(manifest, list):
            entries = manifest
            print(f"  Manifest format: list ({len(entries)} entries)")
        else:
            print(f"  WARNING: Unknown manifest format: {type(manifest)}")
            entries = []

        snapshots = []
        times = []
        # Sort by time
        sorted_entries = sorted(entries, key=lambda x: x["time"])
        for idx, entry in enumerate(sorted_entries):
            # Handle relative_path that may include parent dirs
            rel_path = entry["relative_path"]
            # If path starts with fields/snapshots/, strip it since we're already in snapshots_dir
            if rel_path.startswith("fields/snapshots/"):
                rel_path = rel_path.replace("fields/snapshots/", "")
            snap_path = snapshots_dir / rel_path

            if snap_path.exists():
                data = np.load(snap_path)
                snapshots.append(data["theta"])
                times.append(entry["time"])
            else:
                print(f"  WARNING: Snapshot not found: {snap_path}")

            if (idx + 1) % 50 == 0:
                print(f"  Loaded {idx + 1}/{len(sorted_entries)} snapshots...")

        print(f"  Successfully loaded {len(snapshots)} snapshots")
        return snapshots, times

    # Fallback: parse from filenames
    print(f"  No manifest found, falling back to filename parsing...")
    snap_files = sorted(snapshots_dir.glob("theta_*.npz"))
    print(f"  Found {len(snap_files)} theta_*.npz files")
    snapshots = []
    times = []
    for sf in snap_files:
        # Extract time from filename like theta_00000_t0.0999.npz
        name = sf.stem
        if "_t" in name:
            t_str = name.split("_t")[-1]
            try:
                t = float(t_str)
                data = np.load(sf)
                snapshots.append(data["theta"])
                times.append(t)
            except ValueError:
                continue
    return snapshots, times


def save_theta_velocity_frames(
    frames_dir: Path,
    times: Sequence[float],
    snapshots: Sequence[np.ndarray],
    bg: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> None:
    """Generate frames with cmasher colormaps."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Contour levels for scalar fluctuations (no background gradient added)
    levels = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"  Contour levels (fixed): {levels}")

    spd = np.hypot(ux, uy)
    vmax = float(np.percentile(spd, 99.0)) if np.isfinite(spd).any() else float(np.max(spd))
    print(f"  Speed range: [0.0, {vmax:.4f}] (99th percentile)")

    # Use cmasher colormaps
    speed_cmap = cmr.neutral
    contour_cmap = cmr.neon

    for idx, tnow in enumerate(times):
        if idx >= len(snapshots):
            break
        # Use scalar fluctuations directly (no background gradient)
        theta = np.nan_to_num(snapshots[idx], copy=False)
        fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160, constrained_layout=True)
        im = ax.imshow(spd, origin="lower", cmap=speed_cmap, vmin=0.0, vmax=vmax)
        ax.contour(theta, levels=levels, cmap=contour_cmap, linewidths=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {tnow:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")
        # Use zero-padded frame index for correct sorting by ffmpeg
        fig.savefig(frames_dir / f"frame_{idx:05d}.png", bbox_inches="tight")
        plt.close(fig)

        if (idx + 1) % 20 == 0:
            print(f"  Generated {idx + 1}/{len(times)} frames")


def render_movie(frames_dir: Path, output_path: Path, fps: int = 16) -> bool:
    """Render movie from frames using ffmpeg."""
    from PIL import Image

    # Use new frame naming pattern with zero-padded indices
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        print(f"[warn] No frames found in {frames_dir}")
        return False

    print(f"  Found {len(pngs)} frames: {pngs[0].name} ... {pngs[-1].name}")

    width, height = Image.open(pngs[0]).size
    scale_filter = None
    if (width % 2) or (height % 2):
        scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", "frame_*.png",
    ]
    if scale_filter:
        cmd += ["-vf", scale_filter]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(frames_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        print(f"[error] ffmpeg failed: {exc}")
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    movie_output_dir = args.movie_output_dir.resolve()

    print("=" * 50)
    print("REGENERATE FRAMES AND MOVIE")
    print("=" * 50)
    print(f"Run directory:    {run_dir}")
    print(f"Run name:         {run_dir.name}")
    print(f"Movie output dir: {movie_output_dir}")
    print(f"FPS:              {args.fps}")
    print(f"Delete old:       {args.delete_old}")
    print()

    # Load config
    print("Step 1: Loading configuration...")
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        print(f"  ERROR: No run_config.json found in {run_dir}")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    grid = int(config["grid"])
    domain_size = float(config.get("domain_size", 1.0))
    alpha = config["alpha"]
    kappa = config["kappa"]
    print(f"  Grid:        {grid}")
    print(f"  Domain size: {domain_size}")
    print(f"  Alpha:       {alpha}")
    print(f"  Kappa:       {kappa}")
    print()

    # Load velocity fields
    print("Step 2: Loading velocity fields...")
    velocity_path = run_dir / "fields" / "velocity_fields.npz"
    if not velocity_path.exists():
        print(f"  ERROR: No velocity_fields.npz found at {velocity_path}")
        return 1

    vel_data = np.load(velocity_path)
    ux = vel_data["ux"]
    uy = vel_data["uy"]
    print(f"  ux shape: {ux.shape}, dtype: {ux.dtype}")
    print(f"  uy shape: {uy.shape}, dtype: {uy.dtype}")
    print(f"  ux range: [{ux.min():.4f}, {ux.max():.4f}]")
    print(f"  uy range: [{uy.min():.4f}, {uy.max():.4f}]")
    print()

    # Compute background gradient
    print("Step 3: Computing background gradient...")
    ny, nx = ux.shape
    x = np.linspace(0, domain_size, nx, endpoint=False)
    y = np.linspace(0, domain_size, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    bg = MEAN_GRAD[0] * X + MEAN_GRAD[1] * Y
    print(f"  Mean gradient: {MEAN_GRAD}")
    print(f"  Background range: [{bg.min():.4f}, {bg.max():.4f}]")
    print()

    # Load snapshots
    print("Step 4: Loading snapshots...")
    snapshots_dir = run_dir / "fields" / "snapshots"
    if not snapshots_dir.exists():
        print(f"  ERROR: Snapshots directory not found: {snapshots_dir}")
        return 1

    snapshots, times = load_snapshots_and_times(snapshots_dir)
    print(f"  Loaded {len(snapshots)} snapshots")
    if times:
        print(f"  Time range: [{min(times):.4f}, {max(times):.4f}]")
    print()

    if not snapshots:
        print("  ERROR: No snapshots found")
        return 1

    # Delete old frames if requested
    frames_dir = run_dir / "analysis" / "theta_velocity_frames"
    if args.delete_old and frames_dir.exists():
        print(f"Step 5: Deleting old frames in {frames_dir}...")
        old_count = len(list(frames_dir.glob("*.png")))
        shutil.rmtree(frames_dir)
        print(f"  Deleted {old_count} old frames")
    else:
        print("Step 5: No old frames to delete (or --delete-old not set)")
    print()

    # Generate new frames
    print("Step 6: Generating frames with cmasher colormaps...")
    print(f"  Output directory: {frames_dir}")
    save_theta_velocity_frames(frames_dir, times, snapshots, bg, ux, uy)
    new_count = len(list(frames_dir.glob("*.png")))
    print(f"  Generated {new_count} frames")
    print()

    # Create movie output directory
    print("Step 7: Preparing movie output...")
    movie_output_dir.mkdir(parents=True, exist_ok=True)

    # Create movie name from run parameters
    movie_name = f"alpha_{alpha:.3f}_kappa_{kappa:.0e}.mp4".replace("+", "")
    movie_path = movie_output_dir / movie_name

    # Also create local movie in analysis dir
    local_movie_path = run_dir / "analysis" / "theta_velocity_movie.mp4"
    print(f"  Movie name:      {movie_name}")
    print(f"  Movie path:      {movie_path}")
    print(f"  Local copy path: {local_movie_path}")
    print()

    print("Step 8: Rendering movie...")
    success = render_movie(frames_dir, movie_path, fps=args.fps)

    if success:
        # Copy to local analysis dir as well
        print("Step 9: Copying movie to local analysis directory...")
        shutil.copy(movie_path, local_movie_path)
        print(f"  Movie saved to: {movie_path}")
        print(f"  Local copy:     {local_movie_path}")
        # Verify file sizes
        movie_size = movie_path.stat().st_size / (1024 * 1024)
        local_size = local_movie_path.stat().st_size / (1024 * 1024)
        print(f"  Movie size:     {movie_size:.2f} MB")
        print(f"  Local size:     {local_size:.2f} MB")
    else:
        print("  ERROR: Movie rendering failed")
        return 1

    print()
    print("=" * 50)
    print("SUCCESS - All steps completed")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())