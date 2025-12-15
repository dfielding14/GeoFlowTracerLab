#!/usr/bin/env python
"""Launch sbatch jobs to regenerate frames for all 2048 campaign runs.

Usage:
    python scripts/launch_regen_campaign.py [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_DIR = Path("/mnt/ceph/users/dfielding/GeoFlowTracerLab")
RUNS_DIR = PROJECT_DIR / "experimental_results" / "alpha_kappa_2048"
SBATCH_SCRIPT = PROJECT_DIR / "scripts" / "regenerate_frames.sbatch"
MOVIE_OUTPUT_DIR = PROJECT_DIR / "campaigns" / "alpha_kappa_2048" / "movies"


def find_2048_runs() -> list[Path]:
    """Find all N2048 runs in experimental_results/alpha_kappa_2048."""
    print(f"Searching for runs in: {RUNS_DIR}")
    if not RUNS_DIR.exists():
        print(f"  ERROR: Directory does not exist: {RUNS_DIR}")
        return []

    runs = []
    all_dirs = list(RUNS_DIR.iterdir())
    print(f"  Found {len(all_dirs)} entries in directory")

    for d in all_dirs:
        if not d.is_dir():
            continue
        # Verify it has the required files
        config = d / "run_config.json"
        velocity = d / "fields" / "velocity_fields.npz"
        snapshots = d / "fields" / "snapshots"

        missing = []
        if not config.exists():
            missing.append("run_config.json")
        if not velocity.exists():
            missing.append("fields/velocity_fields.npz")
        if not snapshots.exists():
            missing.append("fields/snapshots/")

        if missing:
            print(f"  SKIP {d.name}: missing {', '.join(missing)}")
        else:
            runs.append(d)
            print(f"  OK   {d.name}")

    return sorted(runs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch frame regeneration jobs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("REGENERATE FRAMES CAMPAIGN LAUNCHER")
    print("=" * 60)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Runs directory:    {RUNS_DIR}")
    print(f"Sbatch script:     {SBATCH_SCRIPT}")
    print(f"Movie output:      {MOVIE_OUTPUT_DIR}")
    print()

    # Verify sbatch script exists
    if not SBATCH_SCRIPT.exists():
        print(f"ERROR: Sbatch script not found: {SBATCH_SCRIPT}")
        return 1
    print(f"Sbatch script exists: OK")

    runs = find_2048_runs()
    print()
    print(f"Found {len(runs)} valid runs to process")
    print()

    if not runs:
        print("ERROR: No runs found!")
        return 1

    # Create logs directory
    logs_dir = PROJECT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    print(f"Logs directory: {logs_dir}")

    # Create movies output directory
    MOVIE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Movies directory: {MOVIE_OUTPUT_DIR}")
    print()

    print("=" * 60)
    print("SUBMITTING JOBS")
    print("=" * 60)

    submitted = []
    failed = []
    for idx, run_dir in enumerate(runs, 1):
        cmd = [
            "sbatch",
            f"--export=RUN_DIR={run_dir},MOVIE_OUTPUT_DIR={MOVIE_OUTPUT_DIR}",
            f"--job-name=regen_{run_dir.name[:30]}",
            str(SBATCH_SCRIPT),
        ]

        print(f"[{idx}/{len(runs)}] {run_dir.name}")
        print(f"  Command: {' '.join(cmd)}")

        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                submitted.append((run_dir.name, job_id))
                print(f"  -> Submitted job ID: {job_id}")
            else:
                failed.append((run_dir.name, result.stderr.strip()))
                print(f"  -> ERROR: {result.stderr.strip()}")
        else:
            print(f"  -> [DRY RUN] Would submit")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if args.dry_run:
        print(f"[DRY RUN] Would submit {len(runs)} jobs")
    else:
        print(f"Submitted: {len(submitted)} jobs")
        print(f"Failed:    {len(failed)} jobs")

        if submitted:
            print("\nSubmitted jobs:")
            for name, job_id in submitted:
                print(f"  {job_id}: {name}")

        if failed:
            print("\nFailed submissions:")
            for name, err in failed:
                print(f"  {name}: {err}")

        print(f"\nMovies will be saved to: {MOVIE_OUTPUT_DIR}")
        print("\nMonitor with: squeue -u $USER")
        print("Check logs in: logs/regen_frames_*.out")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
