"""
Power spectrum utilities for velocity and scalar fields.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .binning import find_ell_bin_edges
from .fitting import best_powerlaw_fit
from .fft import fft2
from .grid import SpectralGrid


def _subtract_linear_trend(
    field: np.ndarray,
    grid: SpectralGrid,
    gradient: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Subtract a best-fit linear plane (mean gradient) from the field."""
    f = np.asarray(field, dtype=np.float64)
    N = grid.N
    x = np.linspace(-grid.L / 2, grid.L / 2, N, endpoint=False)
    y = np.linspace(-grid.L / 2, grid.L / 2, N, endpoint=False)

    if gradient is None:
        # Exploit symmetry: sums over x/y vanish, cross terms drop out.
        sum_x2 = np.sum(x**2) * N
        sum_y2 = np.sum(y**2) * N
        sum_xf = np.dot(f.sum(axis=0), x)
        sum_yf = np.dot(f.sum(axis=1), y)
        Gx = sum_xf / max(sum_x2, 1e-30)
        Gy = sum_yf / max(sum_y2, 1e-30)
    else:
        Gx, Gy = gradient

    plane = np.outer(y, np.ones_like(x)) * Gy + np.outer(np.ones_like(y), x) * Gx
    mean_offset = f.mean()
    return f - plane - mean_offset


def scalar_power_spectrum(
    field: np.ndarray,
    grid: SpectralGrid,
    *,
    subtract_mean: bool = True,
    subtract_mean_gradient: bool = False,
    mean_grad: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shell-averaged power spectrum of a scalar field.
    """
    data = np.asarray(field, dtype=np.float64)

    if subtract_mean_gradient:
        data = _subtract_linear_trend(data, grid, gradient=mean_grad)
    elif subtract_mean:
        data = data - data.mean()

    field_hat = fft2(data)
    power = np.abs(field_hat) ** 2

    k_shells = []
    E_shells = []
    for k_int in range(1, grid.N // 2 + 1):
        mask = np.floor(grid.k_norm + 1e-12) == k_int
        if np.any(mask):
            k_shells.append(grid.k_norm[mask].mean())
            E_shells.append(power[mask].sum())
    return np.array(k_shells), np.array(E_shells)

def plot_scalar_spectrum(
    k: np.ndarray,
    E: np.ndarray,
    *,
    fname: str | None = None,
    title: str | None = None,
    fit_min_points: int = 6,
    fit_min_decades: float = 0.5,
    annotate_fit: bool = False,
    label: str = r"$P_\theta(k)$",
    ax: plt.Axes | None = None,
    fit_min_k: float | None = None,
    fit_max_k: float | None = None,
) -> plt.Axes:
    """
    Plot scalar spectrum and overlay best-fit power law.

    Parameters
    ----------
    k, E : np.ndarray
        Wavenumber centers and corresponding spectral density.
    fname : str, optional
        Save path for figure.
    title : str, optional
        Plot title.
    fit_min_points : int
        Minimum number of consecutive points for fitting.
    fit_min_decades : float
        Minimum log-span for fit window.
    annotate_fit : bool
        Whether to annotate slope text at mid-segment.
    label : str
        Legend label for the spectrum.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    k = np.asarray(k)
    E = np.asarray(E)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    else:
        fig = ax.figure

    (line,) = ax.loglog(k, E, lw=1.8, alpha=0.9, label=label)
    color = line.get_color()

    x_range = None
    if fit_min_k is not None or fit_max_k is not None:
        lo = fit_min_k if fit_min_k is not None else k.min()
        hi = fit_max_k if fit_max_k is not None else k.max()
        x_range = (lo, hi)

    fit = best_powerlaw_fit(
        k,
        E,
        min_points=fit_min_points,
        min_decades=fit_min_decades,
        x_range=x_range,
    )
    if fit is not None:
        ax.loglog(
            fit.xseg,
            fit.yfit,
            color=color,
            lw=4,
            alpha=0.5,
            solid_capstyle="round",
            label=fr"$\propto k^{{{fit.m:.3f}}}$",
        )
        if annotate_fit:
            x_mid = np.sqrt(fit.xseg[0] * fit.xseg[-1])
            y_mid = fit.A * x_mid**fit.m
            ax.text(
                x_mid,
                y_mid,
                fr"$m={fit.m:.3f}$",
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel(r"$k$   (fundamental units; $k{=}1\equiv 2\pi/L$)")
    ax.set_ylabel(r"$P_\theta(k)$")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    if fname:
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


def kinetic_energy_spectrum(
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    n_bins: int = 48,
) -> Dict[str, np.ndarray]:
    """
    Isotropized kinetic energy spectrum E(k) from a 2-D velocity field.
    """
    ux = np.asarray(ux)
    uy = np.asarray(uy)
    ny, nx = ux.shape

    kx_i = (np.fft.fftfreq(nx) * nx).astype(np.float64)
    ky_i = (np.fft.fftfreq(ny) * ny).astype(np.float64)
    KX, KY = np.meshgrid(kx_i, ky_i, indexing="xy")
    Kidx = np.hypot(KX, KY)

    Ux = np.fft.fft2(ux)
    Uy = np.fft.fft2(uy)
    S = 0.5 * (np.abs(Ux) ** 2 + np.abs(Uy) ** 2) / (nx * ny) ** 2
    S = S.copy()
    S[Kidx == 0.0] = 0.0

    kmin_i = 1.0
    kmax_i = float(np.floor(np.max(Kidx)))
    edges_i = find_ell_bin_edges(kmin_i, kmax_i, n_bins)
    edges_i = np.unique(edges_i)
    if edges_i.size < 3:
        edges_i = np.array([1, 2, 3], dtype=int)
    dk = np.diff(edges_i).astype(float)
    centers = 0.5 * (edges_i[:-1] + edges_i[1:])

    shell = np.digitize(Kidx.ravel(), edges_i) - 1
    valid = (shell >= 0) & (shell < dk.size)
    shell_sum = np.bincount(shell[valid], weights=S.ravel()[valid], minlength=dk.size)

    E1d = shell_sum / np.maximum(dk, 1e-300)

    return {
        "k": centers,
        "E": E1d,
        "edges": edges_i.astype(float),
        "dk": dk,
        "E_total": float(E1d @ dk),
    }


def plot_energy_spectrum(
    spec: Dict[str, np.ndarray],
    fname: str | None = None,
    title: str | None = None,
    *,
    fit_min_points: int = 6,
    fit_min_decades: float = 0.5,
    annotate_fit: bool = False,
    label: str = r"$E(k)$",
) -> None:
    """
    Plot E(k) vs k and optionally overlay the best-fit power-law segment.
    """
    k = spec["k"]
    E = spec["E"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=140)
    (line,) = ax.loglog(k, E, lw=1.8, alpha=0.9, label=label)
    color = line.get_color()

    fit = best_powerlaw_fit(k, E, min_points=fit_min_points, min_decades=fit_min_decades)
    if fit is not None:
        ax.loglog(
            fit.xseg,
            fit.yfit,
            color=color,
            lw=4,
            alpha=0.5,
            solid_capstyle="round",
            label=fr"$\propto k^{{{fit.m:.3f}}}$",
        )
        if annotate_fit:
            x_mid = np.sqrt(fit.xseg[0] * fit.xseg[-1])
            y_mid = fit.A * x_mid**fit.m
            ax.text(x_mid, y_mid, fr"$m={fit.m:.3f}$", color=color, fontsize=9, ha="center", va="bottom")

    ax.set_xlabel(r"$k$   (fundamental units; $k{=}1\equiv 2\pi/L$)")
    ax.set_ylabel(r"$E(k)$")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


__all__ = ["scalar_power_spectrum", "plot_scalar_spectrum", "kinetic_energy_spectrum", "plot_energy_spectrum"]
