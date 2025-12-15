#!/usr/bin/env python3
"""Generate sbatch commands for the alpha–kappa production campaign."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shlex
from pathlib import Path
from typing import List


def default_alpha_values() -> List[float]:
    return [k / 6.0 for k in (1, 2, 3, 4, 5)]


def default_kappa_values() -> List[float]:
    return [10.0 ** exp for exp in (-4.0, -3.5, -3.0, -2.5)]


def format_kappa_tag(kappa: float) -> str:
    exp = math.log10(kappa)
    exp_str = f"{exp:.2f}".rstrip("0").rstrip(".")
    exp_str = exp_str.replace("-", "m").replace(".", "p")
    return f"k1e{exp_str}"


def format_alpha_tag(alpha: float) -> str:
    frac = int(round(alpha * 6.0))
    return f"a{frac}of6"


def build_tag(alpha: float, kappa: float, grid: int, t_end: float, workers: int) -> str:
    alpha_tag = format_alpha_tag(alpha)
    kappa_tag = format_kappa_tag(kappa)
    return f"N{grid}_T{int(t_end)}_{alpha_tag}_{kappa_tag}_w{workers}"


def build_python_command(args, alpha: float, kappa: float, tag: str, seed: int) -> str:
    cmd = [
        "python",
        "experiments/run_single_alpha_kappa.py",
        "--alpha",
        f"{alpha:.10f}",
        "--kappa",
        f"{kappa:.10e}",
        "--grid",
        str(args.grid),
        "--t-end",
        f"{args.t_end}",
        "--n-save",
        str(args.n_save),
        "--n-workers",
        str(args.n_workers),
        "--cfl",
        f"{args.cfl}",
        "--integrator",
        args.integrator,
        "--dtype",
        args.dtype,
        "--velocity-seed",
        str(seed),
        "--velocity-urms",
        "1.0",
        "--lam-min",
        f"{args.lam_min}",
        "--lam-max",
        f"{args.lam_max if args.lam_max is not None else args.grid}",
        "--progress-interval",
        str(args.progress_interval),
        "--quiet",
        "--output-root",
        str(args.output_root),
        "--tag",
        tag,
    ]
    if args.dt_save is not None:
        cmd.extend(["--dt-save", f"{args.dt_save}"])
    return " ".join(shlex.quote(part) for part in cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate commands for the alpha–kappa campaign.")
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--t-end", type=float, default=40.0)
    parser.add_argument("--n-save", type=int, default=400)
    parser.add_argument("--n-workers", type=int, default=64)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--integrator", default="etdrk4")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-root", type=Path, default=Path("experimental_results") / "alpha_kappa_runs")
    parser.add_argument("--lam-min", type=float, default=2.0)
    parser.add_argument("--lam-max", type=float)
    parser.add_argument("--dt-save", type=float)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--velocity-seed-base", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("campaigns") / "alpha_kappa_1024",
        help="Directory where the command list and manifest will be written.",
    )
    args = parser.parse_args(argv)

    alpha_vals = default_alpha_values()
    kappa_vals = default_kappa_values()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands_path = args.output_dir / "commands.txt"
    manifest_path = args.output_dir / "manifest.json"

    entries = []
    command_lines = []
    lam_max_val = args.lam_max if args.lam_max is not None else args.grid

    for idx, (alpha, kappa) in enumerate(itertools.product(alpha_vals, kappa_vals), start=1):
        seed = args.velocity_seed_base  # Same seed for all runs
        tag = build_tag(alpha, kappa, args.grid, args.t_end, args.n_workers)
        env = {
            "ALPHA": f"{alpha:.10f}",
            "KAPPA": f"{kappa:.10e}",
            "GRID": args.grid,
            "T_END": args.t_end,
            "N_SAVE": args.n_save,
            "OUTPUT_ROOT": str(args.output_root),
            "CFL": args.cfl,
            "INTEGRATOR": args.integrator,
            "DTYPE": args.dtype,
            "N_WORKERS": args.n_workers,
            "VEL_SEED": seed,
            "LAM_MIN": args.lam_min,
            "LAM_MAX": lam_max_val,
            "TAG": tag,
            "PROGRESS_INTERVAL": args.progress_interval,
        }
        if args.dt_save is not None:
            env["DT_SAVE"] = args.dt_save
        env_str = ",".join(f"{key}={value}" for key, value in env.items())
        sbatch_cmd = f"sbatch --export={env_str} scripts/run_single_alpha_kappa.sbatch"
        python_cmd = build_python_command(args, alpha, kappa, tag, seed)
        entries.append(
            {
                "alpha": alpha,
                "kappa": kappa,
                "tag": tag,
                "velocity_seed": seed,
                "sbatch": sbatch_cmd,
                "python": python_cmd,
            }
        )
        command_lines.append(sbatch_cmd)

    commands_path.write_text("\n".join(command_lines) + "\n")
    manifest_path.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} sbatch commands to {commands_path}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
