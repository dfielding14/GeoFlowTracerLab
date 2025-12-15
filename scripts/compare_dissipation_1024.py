#!/usr/bin/env python3
"""Compare final-time dissipation across the 1024 alpha-kappa campaign."""

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
except ImportError:
    from matplotlib import cm as mpl_cm
    ALPHA_CMAP = mpl_cm.viridis

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "experimental_results" / "alpha_kappa_1024"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "campaigns" / "alpha_kappa_1024"


def load_1024_runs() -> list[dict]:
    """Load dissipation data from all N1024 runs."""
    runs = []
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir():
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
    runs = load_1024_runs()
    if not runs:
        print("No 1024 runs found with dissipation data.")
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

    # Map alpha values to fraction labels
    alpha_labels = {
        0.1666666667: r"1/6",
        0.3333333333: r"1/3",
        0.5: r"1/2",
        0.6666666667: r"2/3",
        0.8333333333: r"5/6",
    }

    # --- Plot: Edot vs kappa for each alpha ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
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
            alpha_lbl = alpha_labels.get(alpha, f"{alpha:.3f}")
            ax.errorbar(
                kappa_vals, edot_vals,
                yerr=[edot_lo, edot_hi],
                fmt="o-", lw=2, ms=8, capsize=4, capthick=1.5,
                color=alpha_colors[i],
                label=rf"$\alpha = {alpha_lbl}$"
            )
            # Fit power-law to leftmost 3 points of highest alpha
            if alpha == alphas[-1] and len(kappa_vals) >= 3:
                fit_kappa = kappa_vals[:3]
                fit_edot = edot_vals[:3]
                slope, intercept = np.polyfit(np.log10(fit_kappa), np.log10(fit_edot), 1)
                kappa_fit = np.logspace(np.log10(0.9e-4), np.log10(3.5e-3), 100)
                edot_fit = 10**intercept * kappa_fit**slope
                ax.plot(kappa_fit, edot_fit, lw=1, color='grey', ls=':',
                        label=rf"$\propto\kappa^{{{slope:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa$", fontsize=14)
    ax.set_ylabel(r"$\dot{\varepsilon}_\theta = 2\kappa \langle |\nabla\theta|^2 \rangle$", fontsize=14)
    ax.set_title(r"Dissipation vs $\kappa$ for each $\alpha$ (N=1024)", fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "edot_vs_kappa_by_alpha.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

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
