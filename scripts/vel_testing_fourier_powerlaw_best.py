#!/usr/bin/env python3
"""vel_testing_fourier_powerlaw_best.py

Companion to ``vel_testing_wavelet_powerlaw_best.py``, but uses the
Fourier amplitude-envelope generator ``VelocityFieldGenerator``.

This method is usually the easiest way to get *extremely* clean power-law
structure functions, because the target scaling is imposed directly in
spectral space.

Conventions
-----------
- ``LAM_MIN`` / ``LAM_MAX`` are separations in *pixels*.
- We set the spectral grid domain length to ``L=N`` so that dx=1 pixel.
- The generator’s band is specified in the internal k_norm units
  (mode-number units):

      kmin_norm = N / LAM_MAX
      kmax_norm = N / LAM_MIN

The structure-function side is identical to the wavelet script:
- per-component scalar SFs via ``structure_functions_fft``
- then average components
- plot (S_p)^(1/p) and local slopes
- compute ESS diagnostics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import cmasher as cmr

ORDER_CMAP = cmr.ember

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalar_advection import structure_functions_fft
from scalar_advection.fitting import best_powerlaw_fit
from scalar_advection.grid import SpectralGrid
from scalar_advection.velocity import VelocityConfig, VelocityFieldGenerator


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

N = 4096
LAM_MIN = 16.0
LAM_MAX = 512.0

SEED0 = 1000
ALPHAS = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]

# For the cleanest curves, prefer even orders (exact via FFT). Add odd orders
# only if you really need them.
ORDERS = (2, 4, 6, 8, 10)

# Binning / sampling
N_ELL_BINS = 256
N_DISP_TOTAL = 65536  # only used for odd orders

# Fourier generator controls
URMS = 1.0
F_SOL = 1.0

# Smooth taper at band edges in k_norm units (mode-number units)
TAPER_FRAC = 0.08

OUTPUT_DIR = REPO_ROOT / "velocity_structure_functions_fourier_powerlaw_best"

ALPHA_LABELS = {
    1 / 6: r"1/6",
    1 / 3: r"1/3",
    1 / 2: r"1/2",
    2 / 3: r"2/3",
    5 / 6: r"5/6",
}


def _geom_centers(edges: np.ndarray) -> np.ndarray:
    lo = np.asarray(edges[:-1], dtype=float)
    hi = np.asarray(edges[1:], dtype=float)
    return np.sqrt(lo * hi)


def sliding_log_slope(r: np.ndarray, y: np.ndarray, window: int = 11) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_velocity_sf(r: np.ndarray, S: np.ndarray, orders: np.ndarray, alpha: float, fname: Path) -> None:
    fig, (ax_main, ax_zeta) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
    )

    colors = ORDER_CMAP(np.linspace(0.15, 0.85, len(orders)))
    fit_lo = float(LAM_MIN)
    fit_hi = float(LAM_MAX)
    alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")

    for j, p in enumerate(orders):
        y = np.power(np.maximum(S[j], 1e-30), 1.0 / float(p))
        color = colors[j]
        ax_main.loglog(r, y, "o-", ms=4, lw=1.8, color=color, label=rf"$p={int(p)}$")

        fit = best_powerlaw_fit(r, y, min_points=8, min_decades=0.7, x_range=(fit_lo, fit_hi))
        if fit is not None:
            ax_main.loglog(fit.xseg, fit.yfit, "--", lw=2.0, color=color, alpha=0.6)

        centers, slopes = sliding_log_slope(r, y, window=11)
        if len(centers) > 0:
            ax_zeta.semilogx(centers, slopes, "-", lw=1.4, color=color)

    ax_zeta.axhline(alpha, color="k", ls="--", lw=1.5, alpha=0.7, label=rf"$\alpha = {alpha_label}$")
    for ax in (ax_main, ax_zeta):
        ax.axvline(fit_lo, color="gray", ls=":", lw=1, alpha=0.6)
        ax.axvline(fit_hi, color="gray", ls=":", lw=1, alpha=0.6)

    ax_main.set_ylabel(r"$(S_p)^{1/p}$", fontsize=13)
    ax_main.legend(frameon=False, fontsize=10, ncol=2, loc="lower right")
    ax_main.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)

    ax_zeta.set_xlabel(r"separation $r$ [pixels]", fontsize=13)
    ax_zeta.set_ylabel(r"$\zeta_p(r)$", fontsize=13)
    ax_zeta.legend(frameon=False, fontsize=10, loc="upper right")
    ax_zeta.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax_zeta.set_ylim(0, 1.2)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_field_map(ux: np.ndarray, uy: np.ndarray, *, alpha: float, fname: Path) -> None:
    n = int(ux.shape[0])
    speed = np.hypot(ux, uy)
    finite = np.isfinite(speed)
    vmax = float(np.percentile(speed[finite], 99.0)) if np.any(finite) else float(np.max(speed))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(speed)) if np.size(speed) else 1.0

    alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=160, constrained_layout=True)
    im = ax.imshow(speed, origin="lower", cmap=cmr.neutral, vmin=0.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(rf"$|\mathbf{{u}}|$ map: $\alpha={alpha_label}$, $N={n}$", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _choose_ess_ref_order(orders: np.ndarray) -> int | None:
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

    for j, _p in enumerate(orders_int):
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
    orders_int = np.asarray(orders).astype(int)
    jref = int(np.where(orders_int == int(p_ref))[0][0])
    Sref = np.asarray(S[jref], dtype=float)
    base_mask = (r >= fit_lo) & (r <= fit_hi) & (Sref > 0) & np.isfinite(Sref)

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=160)
    colors = ORDER_CMAP(np.linspace(0.15, 0.85, len(orders_int)))

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
        ax.loglog(x, comp, "D:", ms=4.0, lw=1.2, color=colors[j], alpha=0.85, label=rf"$p={p}$")

        med = float(np.median(comp))
        ax.hlines(med, xmin=float(x.min()), xmax=float(x.max()), colors=[colors[j]], linestyles=":", lw=1.0, alpha=0.5)

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


def _make_grid(N: int, L: float, dtype: np.dtype) -> SpectralGrid:
    """Best-effort SpectralGrid constructor across possible signatures."""
    for args, kwargs in (
        ((), {"N": int(N), "L": float(L), "dtype": dtype}),
        ((), {"N": int(N), "L": float(L)}),
        ((int(N), float(L), dtype), {}),
        ((int(N), float(L)), {}),
    ):
        try:
            return SpectralGrid(*args, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            continue
    raise TypeError(
        "Could not construct SpectralGrid with any of the tried signatures. "
        "Update _make_grid() to match your SpectralGrid API."
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match pixel units: dx=1 => L=N.
    L = float(N)
    dtype = np.float32
    grid = _make_grid(N=N, L=L, dtype=dtype)
    gen = VelocityFieldGenerator(grid)

    # Convert lam bounds (pixels) to generator k_norm bounds.
    kmin = float(N / float(LAM_MAX))
    kmax = float(N / float(LAM_MIN))
    taper_width = float(max(2.0, TAPER_FRAC * (kmax - kmin)))

    print("Fourier power-law test (best settings)")
    print(f"  N={N}, L={L}")
    print(f"  LAM_MIN={LAM_MIN}, LAM_MAX={LAM_MAX}")
    print(f"  kmin_norm={kmin:.3f}, kmax_norm={kmax:.3f}, taper_width={taper_width:.3f}")
    print(f"  f_sol={F_SOL}, urms={URMS}")
    print(f"  orders={ORDERS}")
    print(f"  output={OUTPUT_DIR}")
    print()

    for alpha in ALPHAS:
        alpha_label = ALPHA_LABELS.get(alpha, f"{alpha:.3f}")
        alpha_str = f"{alpha:.4f}".replace(".", "p")
        print(f"Processing alpha={alpha_label}...")

        seed = SEED0 + 10_000 * int(round(alpha * 1000))
        cfg = VelocityConfig(
            alpha=float(alpha),
            urms=float(URMS),
            seed=int(seed),
            f_sol=float(F_SOL),
            kmin=float(kmin),
            kmax=float(kmax),
            taper_width=float(taper_width),
            precision="float32",
        )

        ux, uy = gen.generate(cfg)
        ux = ux.astype(np.float32, copy=False)
        uy = uy.astype(np.float32, copy=False)

        vel_map_path = OUTPUT_DIR / f"velocity_field_alpha_{alpha_str}.png"
        plot_velocity_field_map(ux, uy, alpha=float(alpha), fname=vel_map_path)

        sf_x = structure_functions_fft(
            ux,
            orders=ORDERS,
            n_ell_bins=N_ELL_BINS,
            n_disp_total=N_DISP_TOTAL,
            seed=seed,
            pad=False,
            r_min=1.0,
            r_max=min(N // 2, int(max(LAM_MAX * 2.0, LAM_MAX))),
        )
        sf_y = structure_functions_fft(
            uy,
            orders=ORDERS,
            n_ell_bins=N_ELL_BINS,
            n_disp_total=N_DISP_TOTAL,
            seed=seed,
            pad=False,
            r_min=1.0,
            r_max=min(N // 2, int(max(LAM_MAX * 2.0, LAM_MAX))),
        )

        if not np.all(sf_x["ell_edges"] == sf_y["ell_edges"]):
            raise RuntimeError("Component bin edges disagree; cannot combine results.")

        ell_edges = sf_x["ell_edges"]
        orders_arr = sf_x["orders"]
        r = _geom_centers(ell_edges)
        S = 0.5 * (sf_x["S"] + sf_y["S"]).astype(np.float64)

        sf_path = OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.png"
        plot_velocity_sf(r, S, orders_arr, float(alpha), sf_path)
        print(f"  Saved: {sf_path}")

        # ESS (same window)
        fit_lo = float(LAM_MIN)
        fit_hi = float(LAM_MAX)
        p_ref = _choose_ess_ref_order(orders_arr)
        if p_ref is None:
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
            ess_comp_path = OUTPUT_DIR / f"velocity_ess_comp_alpha_{alpha_str}.png"
            ess_ratio_path = OUTPUT_DIR / f"velocity_ess_ratio_alpha_{alpha_str}.png"
            ess_title = rf"Compensated ESS (Fourier): $\alpha={alpha_label}$, N={N}"
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
            ratio_title = rf"ESS exponent ratios (Fourier): $\alpha={alpha_label}$, N={N}"
            plot_ess_exponent_ratios(
                orders_arr,
                p_ref=p_ref,
                slopes=ess_slopes,
                slope_err=ess_slope_err,
                title=ratio_title,
                fname=ess_ratio_path,
            )

        np.savez_compressed(
            OUTPUT_DIR / f"velocity_sf_alpha_{alpha_str}.npz",
            r=r,
            S=S,
            orders=orders_arr,
            alpha=float(alpha),
            ell_edges=ell_edges,
            fit_lo=fit_lo,
            fit_hi=fit_hi,
            ess_ref_order=(int(p_ref) if p_ref is not None else -1),
            ess_slopes=ess_slopes,
            ess_slope_err=ess_slope_err,
            ess_r2=ess_r2,
            N=int(N),
            L=float(L),
            lam_min=float(LAM_MIN),
            lam_max=float(LAM_MAX),
            kmin_norm=float(kmin),
            kmax_norm=float(kmax),
            taper_width=float(taper_width),
            urms=float(URMS),
            f_sol=float(F_SOL),
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
