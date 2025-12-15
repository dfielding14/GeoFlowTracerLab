#!/usr/bin/env python3
"""Compare final-time dissipation across the 2048 alpha-kappa campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cmasher as cmr
    ALPHA_CMAP = cmr.ember  # sequential colormap for alpha lines
    KAPPA_CMAP = cmr.ocean  # sequential colormap for kappa lines
except ImportError:
    from matplotlib import cm as mpl_cm
    ALPHA_CMAP = mpl_cm.viridis
    KAPPA_CMAP = mpl_cm.plasma

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "experimental_results" / "alpha_kappa_runs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "campaigns" / "alpha_kappa_2048"


def load_2048_runs() -> list[dict]:
    """Load dissipation data from all N2048 runs."""
    runs = []
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir() or "N2048" not in run_dir.name:
            continue
        config_path = run_dir / "run_config.json"
        diss_path = run_dir / "analysis" / "dissipation_timeseries.npz"
        if not config_path.exists() or not diss_path.exists():
            continue
        config = json.loads(config_path.read_text())
        diss = np.load(diss_path)
        # Get final-time dissipation (average over last 50% of time series)
        epsilon = diss["epsilon"]
        n_avg = max(1, len(epsilon) // 2)
        eps_window = epsilon[-n_avg:]
        edot_mean = float(np.mean(eps_window))
        edot_min = float(np.min(eps_window))
        edot_max = float(np.max(eps_window))
        runs.append({
            "alpha": float(config["alpha"]),
            "kappa": float(config["kappa"]),
            "edot": edot_mean,
            "edot_min": edot_min,
            "edot_max": edot_max,
            "run_dir": run_dir.name,
        })
    return runs


def main() -> int:
    runs = load_2048_runs()
    if not runs:
        print("No 2048 runs found with dissipation data.")
        return 1

    # Extract unique alpha and kappa values
    alphas = sorted(set(r["alpha"] for r in runs))
    kappas = sorted(set(r["kappa"] for r in runs))

    # Build lookup table with mean, min, max
    edot_table = {}
    for r in runs:
        edot_table[(r["alpha"], r["kappa"])] = {
            "mean": r["edot"],
            "min": r["edot_min"],
            "max": r["edot_max"],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Edot vs kappa for each alpha ---
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=160)
    alpha_colors = ALPHA_CMAP(np.linspace(0.15, 0.85, len(alphas)))
    for i, alpha in enumerate(alphas):
        kappa_vals = []
        edot_vals = []
        edot_lo = []
        edot_hi = []
        for kappa in kappas:
            if (alpha, kappa) in edot_table:
                kappa_vals.append(kappa)
                d = edot_table[(alpha, kappa)]
                edot_vals.append(d["mean"])
                edot_lo.append(d["mean"] - d["min"])
                edot_hi.append(d["max"] - d["mean"])
        if kappa_vals:
            kappa_vals = np.array(kappa_vals)
            edot_vals = np.array(edot_vals)
            ax1.errorbar(
                kappa_vals, edot_vals,
                yerr=[edot_lo, edot_hi],
                fmt="o-", lw=2, ms=8, capsize=4, capthick=1.5,
                color=alpha_colors[i],
                label=rf"$\alpha = {alpha:.3f}$"
            )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\kappa$", fontsize=14)
    ax1.set_ylabel(r"$\dot{\varepsilon}_\theta$ (last 50% avg)", fontsize=14)
    ax1.set_title(r"Dissipation vs $\kappa$ for each $\alpha$ (N=2048)", fontsize=14)
    ax1.legend(frameon=False, fontsize=11)
    ax1.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    fig1.tight_layout()
    out1 = OUTPUT_DIR / "edot_vs_kappa_by_alpha.png"
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # --- Plot 2: Edot vs alpha for each kappa ---
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=160)
    kappa_colors = KAPPA_CMAP(np.linspace(0.15, 0.85, len(kappas)))
    for j, kappa in enumerate(kappas):
        alpha_vals = []
        edot_vals = []
        edot_lo = []
        edot_hi = []
        for alpha in alphas:
            if (alpha, kappa) in edot_table:
                alpha_vals.append(alpha)
                d = edot_table[(alpha, kappa)]
                edot_vals.append(d["mean"])
                edot_lo.append(d["mean"] - d["min"])
                edot_hi.append(d["max"] - d["mean"])
        if alpha_vals:
            alpha_vals = np.array(alpha_vals)
            edot_vals = np.array(edot_vals)
            ax2.errorbar(
                alpha_vals, edot_vals,
                yerr=[edot_lo, edot_hi],
                fmt="s-", lw=2, ms=8, capsize=4, capthick=1.5,
                color=kappa_colors[j],
                label=rf"$\kappa = {kappa:.1e}$"
            )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\alpha$", fontsize=14)
    ax2.set_ylabel(r"$\dot{\varepsilon}_\theta$ (last 50% avg)", fontsize=14)
    ax2.set_title(r"Dissipation vs $\alpha$ for each $\kappa$ (N=2048)", fontsize=14)
    ax2.legend(frameon=False, fontsize=11)
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    fig2.tight_layout()
    out2 = OUTPUT_DIR / "edot_vs_alpha_by_kappa.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Print summary table
    print(f"\nLoaded {len(runs)} runs")
    print(f"Alphas: {alphas}")
    print(f"Kappas: {kappas}")
    print("\nFinal-time dissipation table (last 50% average):")
    print(f"{'alpha':<10} {'kappa':<12} {'edot_mean':<12} {'edot_min':<12} {'edot_max':<12}")
    print("-" * 58)
    for alpha in alphas:
        for kappa in kappas:
            if (alpha, kappa) in edot_table:
                d = edot_table[(alpha, kappa)]
                print(f"{alpha:<10.4f} {kappa:<12.2e} {d['mean']:<12.4e} {d['min']:<12.4e} {d['max']:<12.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
