#!/usr/bin/env python3
"""
Compute and plot scalar (theta) structure functions from final snapshots for the 1024 alpha-kappa campaign,
with fits focused on large scales (default fit window: N/6 to N/2).

This scans run directories under:
  experimental_results/alpha_kappa_1024/
and loads:
  fields/theta_final.npz

Per run it produces:
- A 2-panel plot: (S_p)^{1/p} vs r and local slopes zeta_p(r)
- ESS plots (compensated + exponent ratios)
- A compressed .npz with r, S_p(r), and metadata
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cmasher as cmr

    ORDER_CMAP = cmr.ember
except ImportError:
    from matplotlib import cm as mpl_cm

    ORDER_CMAP = mpl_cm.viridis

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import structure_functions_fft
from scalar_advection.fitting import best_powerlaw_fit


# Defaults matching typical campaign analysis settings
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experimental_results" / "alpha_kappa_1024"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "campaigns" / "alpha_kappa_1024" / "theta_structure_functions_large_scales"
DEFAULT_ORDERS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
DEFAULT_N_ELL_BINS = 60
DEFAULT_N_DISP_TOTAL = 8192
DEFAULT_FIT_LO = None


RUN_NAME_RE = re.compile(r"alpha_(?P<alpha>[^_]+)_kappa_(?P<kappa>[^_]+)_.*_N(?P<N>\d+)_")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute theta structure functions (final snapshot) for 1024 campaign (large-scale fits)"
    )
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--orders", type=int, nargs="+", default=list(DEFAULT_ORDERS))
    p.add_argument("--n-ell-bins", type=int, default=DEFAULT_N_ELL_BINS)
    p.add_argument("--n-disp-total", type=int, default=DEFAULT_N_DISP_TOTAL, help="Used only for odd orders")
    p.add_argument("--seed", type=int, default=42, help="Seed for odd-order displacement sampling")
    p.add_argument("--fit-lo", type=float, default=DEFAULT_FIT_LO, help="Default: N/6 from each run's grid")
    p.add_argument("--fit-hi", type=float, default=None, help="Default: N/2 from each run's grid")
    p.add_argument("--pad", action="store_true", help="Use non-periodic estimator (overlap only)")
    p.add_argument("--alpha-filter", type=str, default=None, help="Process only runs with this alpha string (e.g. 0.167)")
    p.add_argument("--kappa-filter", type=str, default=None, help="Process only runs with this kappa string (e.g. 1e-3.50)")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    alpha_str: str
    kappa_str: str
    grid: int
    domain_size: float


def iter_runs(results_root: Path) -> Iterable[RunInfo]:
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue

        m = RUN_NAME_RE.match(run_dir.name)
        if not m:
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue

        grid = int(config.get("grid", m.group("N")))
        domain_size = float(config.get("domain_size", 1.0))
        yield RunInfo(
            run_dir=run_dir,
            alpha_str=m.group("alpha"),
            kappa_str=m.group("kappa"),
            grid=grid,
            domain_size=domain_size,
        )


def load_theta_final(run_dir: Path) -> np.ndarray | None:
    final_path = run_dir / "fields" / "theta_final.npz"
    if final_path.exists():
        data = np.load(final_path)
        return data["theta"]
    return None


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    mask = (r > 0) & (y > 0) & np.isfinite(r) & np.isfinite(y)
    r = r[mask]
    y = y[mask]

    if len(r) < window:
        return np.array([]), np.array([])

    log_r = np.log(r)
    log_y = np.log(np.maximum(y, 1e-30))

    slopes = []
    centers = []
    for i in range(len(r) - window + 1):
        idx = slice(i, i + window)
        m, _ = np.polyfit(log_r[idx], log_y[idx], 1)
        slopes.append(m)
        centers.append(np.exp(np.mean(log_r[idx])))

    return np.array(centers), np.array(slopes)


def _safe_tag(s: str) -> str:
    return s.replace(".", "p").replace("-", "m")


def plot_theta_sf(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    title: str,
    fname: Path,
    *,
    fit_lo: float,
    fit_hi: float,
) -> None:
    fig, (ax_main, ax_zeta) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
    )

    colors = ORDER_CMAP(np.linspace(0.15, 0.85, len(orders)))

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / p)
        color = colors[j]

        ax_main.loglog(r, y, "o-", ms=3.5, lw=1.8, color=color, alpha=0.55, label=rf"$p={int(p)}$")

        fit = best_powerlaw_fit(r, y, min_points=6, min_decades=0.3, x_range=(fit_lo, fit_hi))
        if fit is not None:
            ax_main.loglog(fit.xseg, fit.yfit, "--", lw=2.4, color=color, alpha=0.95)

        centers, slopes = sliding_log_slope(r, y, window=8)
        if centers.size:
            ax_zeta.semilogx(centers, slopes, "-", lw=1.4, color=color)

    ax_main.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.5)
    ax_main.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.5)
    ax_zeta.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.5)
    ax_zeta.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.5)

    ax_main.set_ylabel(r"$(S_p)^{1/p}$", fontsize=13)
    ax_main.set_title(title, fontsize=14)
    ax_main.legend(frameon=False, fontsize=10, ncol=2, loc="lower right")
    ax_main.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)

    ax_zeta.set_xlabel(r"separation $r$ [pixels]", fontsize=13)
    ax_zeta.set_ylabel(r"$\zeta_p(r)$", fontsize=13)
    ax_zeta.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax_zeta.set_ylim(-0.1, 1.6)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _choose_ess_ref_order(orders: np.ndarray) -> int | None:
    """Prefer p=3, fall back to p=2 if p=3 is unavailable."""
    vals = set(int(p) for p in np.asarray(orders).astype(int))
    if 3 in vals:
        return 3
    if 2 in vals:
        return 2
    return None


def fit_ess_slopes(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    *,
    p_ref: int,
    fit_lo: float,
    fit_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ESS slopes: log10 S_p vs log10 S_pref over a restricted r window.

    Returns:
        slopes: slope for each p (NaN if insufficient data)
        slope_err: standard error of slope (NaN if unavailable)
        r2: R^2 of the log-log fit (NaN if unavailable)
    """
    orders_int = np.asarray(orders).astype(int)
    jref = np.where(orders_int == int(p_ref))[0]
    if not jref.size:
        raise ValueError(f"Reference order p_ref={p_ref} not present in orders={orders_int.tolist()}")
    jref = int(jref[0])

    Sref = np.asarray(S[jref], dtype=float)
    base_mask = (r >= fit_lo) & (r <= fit_hi) & (Sref > 0) & np.isfinite(Sref)

    slopes = np.full(len(orders_int), np.nan, dtype=float)
    slope_err = np.full(len(orders_int), np.nan, dtype=float)
    r2 = np.full(len(orders_int), np.nan, dtype=float)

    for j, p in enumerate(orders_int):
        Sp = np.asarray(S[j], dtype=float)
        valid = base_mask & (Sp > 0) & np.isfinite(Sp)
        if np.sum(valid) < 4:
            continue

        x = np.log10(Sref[valid])
        y = np.log10(Sp[valid])
        m, c = np.polyfit(x, y, 1)
        slopes[j] = float(m)

        y_pred = m * x + c
        resid = y - y_pred
        dof = x.size - 2
        if dof > 0:
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            r2[j] = 1.0 - ss_res / (ss_tot + 1e-30)
            s2 = ss_res / dof
            sxx = float(np.sum((x - float(np.mean(x))) ** 2))
            if sxx > 0:
                slope_err[j] = float(np.sqrt(s2 / sxx))

    return slopes, slope_err, r2


def plot_ess_compensated(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    *,
    p_ref: int,
    slopes: np.ndarray,
    fit_lo: float,
    fit_hi: float,
    title: str,
    fname: Path,
) -> None:
    """
    Compensated ESS: plot S_p / S_pref^{zeta_p} vs S_pref, where zeta_p is the ESS slope.
    """
    orders_int = np.asarray(orders).astype(int)
    jref = int(np.where(orders_int == int(p_ref))[0][0])
    Sref = np.asarray(S[jref], dtype=float)
    base_mask = (r >= fit_lo) & (r <= fit_hi) & (Sref > 0) & np.isfinite(Sref)

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=160)
    colors = ORDER_CMAP(np.linspace(0.15, 0.85, len(orders_int)))
    comp_values: List[np.ndarray] = []

    for j, p in enumerate(orders_int):
        zeta = float(slopes[j])
        if not np.isfinite(zeta):
            continue
        Sp = np.asarray(S[j], dtype=float)
        valid = base_mask & (Sp > 0) & np.isfinite(Sp)
        if np.sum(valid) < 3:
            continue
        x = Sref[valid]
        comp = Sp[valid] / np.power(x, zeta)
        comp_values.append(comp)
        ax.loglog(x, comp, "D:", ms=4.0, lw=1.2, color=colors[j], alpha=0.85, label=rf"$p={p}$")

        med = float(np.median(comp))
        ax.hlines(med, xmin=float(x.min()), xmax=float(x.max()), colors=[colors[j]], linestyles=":", lw=1.0, alpha=0.5)

    if comp_values:
        all_comp = np.concatenate(comp_values)
        valid_comp = np.isfinite(all_comp) & (all_comp > 0)
        if np.any(valid_comp):
            logc = np.log10(all_comp[valid_comp])
            q_lo, q_hi = np.quantile(logc, [0.05, 0.95])
            span = float(q_hi - q_lo)
            pad = max(0.15, 0.05 * span)  # decades
            y_lo = float(10 ** (q_lo - pad))
            y_hi = float(10 ** (q_hi + pad))
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo > 0 and y_hi > y_lo:
                ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(rf"$S_{{{p_ref}}}$", fontsize=12)
    ax.set_ylabel(rf"$S_p / S_{{{p_ref}}}^{{\zeta_p}}$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_ess_exponent_ratios(
    orders: np.ndarray,
    *,
    p_ref: int,
    slopes: np.ndarray,
    slope_err: np.ndarray,
    title: str,
    fname: Path,
) -> None:
    """
    Plot normalized ESS exponent ratios:
      zeta_p / (p/p_ref)
    where zeta_p is the fitted ESS slope (log S_p vs log S_pref).
    """
    orders_int = np.asarray(orders).astype(int)
    p_over = orders_int.astype(float) / float(p_ref)
    ratio = slopes / p_over
    ratio_err = slope_err / p_over

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=160)

    valid = np.isfinite(ratio)
    if np.any(valid):
        ax.errorbar(
            orders_int[valid],
            ratio[valid],
            yerr=ratio_err[valid],
            fmt="o-",
            ms=5,
            lw=1.8,
            capsize=3,
            color="k",
        )

    ax.axhline(1.0, color="0.4", ls=":", lw=1.4, label=rf"$\zeta_p = p/{p_ref}$")
    ax.set_xlabel(r"$p$", fontsize=12)
    ax.set_ylabel(rf"$\zeta_p / (p/{p_ref})$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)
    ax.set_xlim(0.5, float(np.max(orders_int)) + 0.5)
    ax.legend(frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = args.results_root
    if not results_root.is_absolute():
        results_root = (REPO_ROOT / results_root).resolve()

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = tuple(int(p) for p in args.orders)
    runs = list(iter_runs(results_root))
    if args.alpha_filter is not None:
        runs = [r for r in runs if r.alpha_str == args.alpha_filter]
    if args.kappa_filter is not None:
        runs = [r for r in runs if r.kappa_str == args.kappa_filter]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print(f"Found {len(runs)} candidate runs under {results_root}")
    if args.dry_run:
        for r in runs:
            print(f"  {r.run_dir.name}")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for i, run in enumerate(runs):
        theta = load_theta_final(run.run_dir)
        if theta is None:
            print(f"[{i+1}/{len(runs)}] Skipping (no theta_final): {run.run_dir.name}")
            skipped += 1
            continue

        if theta.ndim != 2:
            print(f"[{i+1}/{len(runs)}] Skipping (theta not 2D): {run.run_dir.name}")
            skipped += 1
            continue

        if run.grid != theta.shape[0]:
            print(f"[{i+1}/{len(runs)}] WARNING: grid mismatch config={run.grid} theta={theta.shape}")

        fit_lo = float(args.fit_lo) if args.fit_lo is not None else float(run.grid) / 6.0
        fit_hi = float(args.fit_hi) if args.fit_hi is not None else float(run.grid) / 2.0

        out_tag = f"alpha_{_safe_tag(run.alpha_str)}_kappa_{_safe_tag(run.kappa_str)}"
        png_path = out_dir / f"theta_sf_{out_tag}.png"
        ess_comp_path = out_dir / f"theta_ess_comp_{out_tag}.png"
        ess_ratio_path = out_dir / f"theta_ess_ratio_{out_tag}.png"
        npz_path = out_dir / f"theta_sf_{out_tag}.npz"
        if not args.overwrite and png_path.exists() and npz_path.exists() and ess_comp_path.exists() and ess_ratio_path.exists():
            print(f"[{i+1}/{len(runs)}] Skipping (exists): {out_tag}")
            skipped += 1
            continue

        print(f"[{i+1}/{len(runs)}] Processing {out_tag} ...")
        try:
            sf = structure_functions_fft(
                theta,
                orders=orders,
                n_ell_bins=int(args.n_ell_bins),
                n_disp_total=int(args.n_disp_total),
                seed=int(args.seed),
                pad=bool(args.pad),
            )
            r = sf["r"]
            S = sf["S"]
            orders_arr = sf["orders"]

            title = rf"Scalar SF (final $\theta$): $\alpha={run.alpha_str}$, $\kappa={run.kappa_str}$, N={run.grid}"
            plot_theta_sf(r, S, orders_arr, title, png_path, fit_lo=fit_lo, fit_hi=fit_hi)

            # ESS plots
            p_ref = _choose_ess_ref_order(orders_arr)
            if p_ref is None:
                print("  [warn] ESS skipped: need p=3 or p=2 in --orders")
                ess_slopes = np.full_like(orders_arr, np.nan, dtype=float)
                ess_slope_err = np.full_like(orders_arr, np.nan, dtype=float)
                ess_r2 = np.full_like(orders_arr, np.nan, dtype=float)
            else:
                ess_slopes, ess_slope_err, ess_r2 = fit_ess_slopes(
                    r,
                    S,
                    orders_arr,
                    p_ref=p_ref,
                    fit_lo=fit_lo,
                    fit_hi=fit_hi,
                )
                ess_title = rf"Compensated ESS (final $\theta$): $\alpha={run.alpha_str}$, $\kappa={run.kappa_str}$, N={run.grid}"
                plot_ess_compensated(
                    r,
                    S,
                    orders_arr,
                    p_ref=p_ref,
                    slopes=ess_slopes,
                    fit_lo=fit_lo,
                    fit_hi=fit_hi,
                    title=ess_title,
                    fname=ess_comp_path,
                )
                ratio_title = rf"ESS exponent ratios (final $\theta$): $\alpha={run.alpha_str}$, $\kappa={run.kappa_str}$, N={run.grid}"
                plot_ess_exponent_ratios(
                    orders_arr,
                    p_ref=p_ref,
                    slopes=ess_slopes,
                    slope_err=ess_slope_err,
                    title=ratio_title,
                    fname=ess_ratio_path,
                )

            np.savez_compressed(
                npz_path,
                r=r,
                S=S,
                orders=orders_arr,
                ell_edges=sf["ell_edges"],
                alpha=float(json.loads((run.run_dir / "run_config.json").read_text())["alpha"]),
                kappa=float(json.loads((run.run_dir / "run_config.json").read_text())["kappa"]),
                run_dir=str(run.run_dir),
                grid=run.grid,
                domain_size=run.domain_size,
                fit_lo=fit_lo,
                fit_hi=fit_hi,
                pad=bool(args.pad),
                n_ell_bins=int(args.n_ell_bins),
                n_disp_total=int(args.n_disp_total),
                seed=int(args.seed),
                ess_ref_order=(int(p_ref) if p_ref is not None else -1),
                ess_slopes=ess_slopes,
                ess_slope_err=ess_slope_err,
                ess_r2=ess_r2,
            )

            processed += 1
        except Exception as exc:
            print(f"[{i+1}/{len(runs)}] ERROR: {run.run_dir.name}: {exc}")
            failed += 1

    print(f"\nSummary: {processed} processed, {skipped} skipped, {failed} failed")
    print(f"Outputs: {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
