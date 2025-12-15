#!/usr/bin/env python3
"""
Stitch theta_velocity_frames PNGs into MP4 movies using ffmpeg.

Finds all run directories in experimental_results/alpha_kappa_1024,
creates movies named alpha_X_kappa_Y_N1024_theta_velocity.mp4,
and places them in campaigns/alpha_kappa_1024/movies/.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def extract_params_from_dirname(dirname: str) -> dict:
    """Extract alpha, kappa, and N from directory name like:
    alpha_0.5_kappa_1e-4_20251207_210051_N1024_T40_a3of6_k1em4_w64
    """
    match = re.match(r'alpha_([\d.]+)_kappa_([\de.\-+]+)_.*_N(\d+)_', dirname)
    if not match:
        return None
    return {
        'alpha': match.group(1),
        'kappa': match.group(2),
        'N': match.group(3),
    }


def make_movie(frames_dir: Path, output_path: Path, fps: int = 30, crf: int = 23):
    """Create MP4 from PNG frames using ffmpeg."""
    # Get sorted list of PNGs (sorted by time value in filename)
    pngs = sorted(frames_dir.glob('theta_u_t*.png'),
                  key=lambda p: float(re.search(r't([\d.]+)\.png', p.name).group(1)))

    if not pngs:
        print(f"  No PNGs found in {frames_dir}")
        return False

    # Create temporary file list for ffmpeg concat demuxer
    filelist_path = frames_dir / '_filelist.txt'
    with open(filelist_path, 'w') as f:
        for png in pngs:
            # Escape single quotes in path
            escaped = str(png).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write(f"duration {1.0/fps:.6f}\n")
        # Add last frame again (ffmpeg concat quirk)
        escaped = str(pngs[-1]).replace("'", "'\\''")
        f.write(f"file '{escaped}'\n")

    # Run ffmpeg
    cmd = [
        'ffmpeg', '-y',  # overwrite output
        '-f', 'concat',
        '-safe', '0',
        '-i', str(filelist_path),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # ensure even dimensions
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        filelist_path.unlink()  # cleanup
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed: {e.stderr[:500]}")
        filelist_path.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Create movies from theta_velocity_frames')
    parser.add_argument('--input-root', type=Path,
                        default=Path('experimental_results/alpha_kappa_1024'),
                        help='Root directory containing run directories')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/mnt/home/dfielding/ceph/GeoFlowTracerLab/campaigns/alpha_kappa_1024/movies'),
                        help='Output directory for movies')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--crf', type=int, default=23, help='CRF quality (lower=better, 18-28 typical)')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done')
    parser.add_argument('--alpha-filter', type=str, default=None, help='Only process runs with this alpha value (e.g., "0.833")')
    args = parser.parse_args()

    # Find all run directories with theta_velocity_frames
    input_root = args.input_root
    if not input_root.is_absolute():
        input_root = Path('/mnt/ceph/users/dfielding/GeoFlowTracerLab') / input_root

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(input_root.iterdir()) if input_root.exists() else []

    success_count = 0
    skip_count = 0
    fail_count = 0

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        frames_dir = run_dir / 'analysis' / 'theta_velocity_frames'
        if not frames_dir.exists():
            continue

        params = extract_params_from_dirname(run_dir.name)
        if not params:
            print(f"Skipping {run_dir.name}: couldn't parse parameters")
            skip_count += 1
            continue

        # Apply alpha filter if specified
        if args.alpha_filter and params['alpha'] != args.alpha_filter:
            continue

        output_name = f"alpha_{params['alpha']}_kappa_{params['kappa']}_N{params['N']}_theta_velocity.mp4"
        output_path = output_dir / output_name

        if args.dry_run:
            print(f"Would create: {output_name}")
            print(f"  From: {frames_dir}")
            continue

        print(f"Creating {output_name}...")
        if make_movie(frames_dir, output_path, fps=args.fps, crf=args.crf):
            print(f"  Done: {output_path}")
            success_count += 1
        else:
            fail_count += 1

    if not args.dry_run:
        print(f"\nSummary: {success_count} created, {skip_count} skipped, {fail_count} failed")
        print(f"Movies saved to: {output_dir}")


if __name__ == '__main__':
    main()
