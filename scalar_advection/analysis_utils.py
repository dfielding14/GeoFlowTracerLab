from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # optional color maps
    import cmasher as cmr
except Exception:  # pragma: no cover - optional dependency
    cmr = None
from matplotlib import cm as mpl_cm

from .structure import structure_functions
from .binning import find_ell_bin_edges
from .structure import generate_displacements


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_theta_velocity_frames(
    frames_dir: Path,
    times: Sequence[float],
    snapshots: Sequence[np.ndarray],
    bg: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> None:
    ensure_dir(frames_dir)
    levels = [-0.3, -0.15, 0.0, 0.15, 0.3]
    spd = np.hypot(ux, uy)
    vmax = float(np.percentile(spd, 99.0)) if np.isfinite(spd).any() else float(np.max(spd))
    speed_cmap = cmr.neutral if cmr is not None else mpl_cm.viridis
    contour_cmap = cmr.neon if cmr is not None else mpl_cm.coolwarm
    for idx, tnow in enumerate(times):
        if idx >= len(snapshots):
            break
        theta = np.nan_to_num(snapshots[idx] + bg, copy=False)
        fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160, constrained_layout=True)
        im = ax.imshow(spd, origin="lower", cmap=speed_cmap, vmin=0.0, vmax=vmax)
        cs = ax.contour(theta, levels=levels, cmap=contour_cmap, linewidths=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {tnow:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\mathbf{u}|$")
        fig.savefig(frames_dir / f"theta_u_t{tnow:.4f}.png", bbox_inches="tight")
        plt.close(fig)


def plot_and_save_dissipation(
    out_dir: Path,
    times: np.ndarray,
    grad_sq: np.ndarray,
    eps: np.ndarray,
    kappa: float,
) -> None:
    np.savez_compressed(
        out_dir / "dissipation_timeseries.npz",
        t=times,
        grad_sq=grad_sq,
        epsilon=eps,
        kappa=kappa,
    )
    if times.size >= 2:
        dt = np.diff(times)
        mid = 0.5 * (grad_sq[:-1] + grad_sq[1:])
        cum = np.concatenate([[0.0], np.cumsum(kappa * mid * dt)])
    else:
        cum = np.zeros_like(times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.6), dpi=160, sharex=True)
    ax1.plot(times, eps, "-", lw=1.6)
    ax1.set_ylabel(r"$\epsilon_\theta(t) = 2\,\kappa\,\langle |\nabla\theta|^2 \rangle$")
    ax1.grid(True, ls=":", lw=0.6)
    ax2.plot(times, cum, "-", lw=1.8)
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\kappa \int_0^t \langle |\nabla\theta|^2 \rangle \, dt$")
    ax2.grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "dissipation_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def sliding_log_slope_series(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.array([]), np.array([])
    lx = np.log(x)
    ly = np.log(y)
    centers = np.sqrt(x[:-1] * x[1:])
    slopes = np.diff(ly) / np.diff(lx)
    return centers, slopes


def plot_scalar_sf_with_slopes(
    r: np.ndarray,
    S: np.ndarray,
    orders: np.ndarray,
    fname: Path,
    title: str,
    *,
    lambda_t: float | None = None,
) -> None:
    orders = np.asarray(orders, dtype=float)
    root_curves = np.power(np.maximum(S, 1e-30), (1.0 / orders[:, None]))
    fig, (ax_main, ax_slope) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
    )
    colors = plt.cm.tab10.colors
    for j, p in enumerate(orders):
        y = root_curves[j]
        color = colors[j % len(colors)]
        ax_main.loglog(r, y, "o-", lw=1.6, color=color, label=rf"p={p:g}")
        centers, slopes = sliding_log_slope_series(r, y)
        if centers.size:
            ax_slope.semilogx(centers, slopes, "-", lw=1.2, color=color)
    ax_main.set_ylabel(r"$(S_p)^{1/p}$")
    ax_main.set_title(title)
    ax_main.grid(True, which="both", ls=":", lw=0.6)
    ax_main.legend(frameon=False, ncol=1, loc="best")
    ax_slope.set_xlabel(r"separation $\ell / \Delta x$")
    ax_slope.set_ylabel(r"$d\log (S_p^{1/p}) / d\log \ell$")
    ax_slope.grid(True, which="both", ls=":", lw=0.6)
    if lambda_t is not None and np.isfinite(lambda_t) and lambda_t > 0:
        ax_main.axvline(
            lambda_t,
            color="0.3",
            ls=":",
            lw=1.0,
            alpha=0.9,
            label=r"$\lambda_T$",
        )
        ax_slope.axvline(lambda_t, color="0.3", ls=":", lw=1.0, alpha=0.9)
        ax_main.legend(frameon=False, ncol=1, loc="best")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_yaglom_with_slopes(r: np.ndarray, y: np.ndarray, fname: Path) -> None:
    n = min(len(r), len(y))
    r = np.asarray(r[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    fig, (ax_main, ax_slope) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.6),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
    )
    ax_main.loglog(r, y, "o-", lw=1.6, label=r"$\langle | \delta u | | \delta \theta |^2 \rangle$")
    ax_main.set_ylabel(r"$\langle | \delta u | | \delta \theta |^2 \rangle$")
    ax_main.grid(True, which="both", ls=":", lw=0.6)
    ax_main.legend(frameon=False)
    centers, slopes = sliding_log_slope_series(r, y)
    if centers.size:
        ax_slope.semilogx(centers, slopes, "-", lw=1.5)
    ax_slope.set_xlabel(r"separation $\ell / \Delta x$")
    ax_slope.set_ylabel(r"$d\log Y / d\log \ell$")
    ax_slope.grid(True, which="both", ls=":", lw=0.6)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def yaglom_statistics(
    ux: np.ndarray,
    uy: np.ndarray,
    theta: np.ndarray,
    *,
    n_ell_bins: int,
    n_disp_total: int,
    samples_per_disp: int,
    seed: int,
    dx: float,
    L: float,
) -> dict:
    ny, nx = theta.shape
    r_max = min(nx, ny) // 2
    ell_edges = find_ell_bin_edges(1.0, r_max, n_ell_bins)
    n_per_bin = max(1, n_disp_total // n_ell_bins)
    disps = generate_displacements(ell_edges, n_per_bin, seed=seed)

    rng = np.random.default_rng(seed)
    accum = np.zeros(n_ell_bins, dtype=np.float64)
    counts = np.zeros(n_ell_bins, dtype=np.int64)

    for dx_int, dy_int in disps:
        r = float(np.hypot(dx_int, dy_int))
        if r == 0.0:
            continue
        b = np.searchsorted(ell_edges, r, side="right") - 1
        if b < 0 or b >= n_ell_bins:
            continue

        n_samples = min(samples_per_disp, nx * ny)
        iy = rng.integers(0, ny, size=n_samples, endpoint=False)
        ix = rng.integers(0, nx, size=n_samples, endpoint=False)
        iyp = (iy + dy_int) % ny
        ixp = (ix + dx_int) % nx

        dux = ux[iyp, ixp] - ux[iy, ix]
        duy = uy[iyp, ixp] - uy[iy, ix]
        dtheta = theta[iyp, ixp] - theta[iy, ix]

        delta_u_mag = np.sqrt(dux * dux + duy * duy)
        delta_theta_sq = dtheta * dtheta
        term = delta_u_mag * delta_theta_sq
        accum[b] += float(np.mean(term))
        counts[b] += 1

    r_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    with np.errstate(invalid="ignore"):
        yaglom = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)

    return {
        "r": r_centers,
        "counts": counts,
        "yaglom": yaglom,
        "ell_edges": ell_edges,
        "samples_per_disp": samples_per_disp,
        "dx": dx,
        "L": L,
    }
