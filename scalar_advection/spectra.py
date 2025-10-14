"""
Power spectrum utilities for velocity and scalar fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .binning import find_ell_bin_edges
from .fitting import best_powerlaw_fit
from .fft import fft2
from .grid import SpectralGrid


@dataclass
class Spectrum:
    k: np.ndarray
    Pk: np.ndarray
    edges: np.ndarray
    dk: np.ndarray
    Etot: np.ndarray

def _binning_helper(
    ks: np.ndarray,
    Es: np.ndarray,
    kmin: float = 1,
    kmax: Optional[float] = None,
) -> Spectrum:
    """
    Take 2D arrays in k-space and bin according to the norm of k
    args:
        ks - wavenumbers to bin along
        Es - amplitudes at each wave-number
    returns:
        Spectrum - The spectrum dataclass describing the resulting binned spectrum
    """

    if ks.ndim != 2 or Es.ndim != 2 or ks.shape != Es.shape:
        raise ValueError("ks and Es must be 2D arrays")
    N = ks.shape[0]

    if kmax is None:
        kmax = np.floor(np.max(ks))

    kbins = find_ell_bin_edges(kmin, kmax, n_ell_bins=int(kmax - kmin))
    kcs = 0.5 * (kbins[:-1] + kbins[1:])
    dk = kbins[1:] - kbins[:-1]

    #kbins = np.geomspace(kmin, kmax, num=N//2 + 1)
    #kcs = 0.5 * (kbins[:-1] + kbins[1:])
    #dk = np.diff(kbins)

    (E1d,be) = np.histogram(ks.flatten(), bins=kbins, weights=Es.flatten())
    E1d /= dk

    return Spectrum(k=kcs, Pk=E1d, edges=be, dk=dk, Etot=E1d.sum())

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
) -> Spectrum:
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

    return _binning_helper(grid.k_norm, power, kmin=1.0, kmax=grid.N // 2 + 1)


def plot_scalar_spectrum(
    spec: Spectrum,
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
    k = spec.k
    E = spec.Pk
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
    grid: SpectralGrid,
    *,
    n_bins: int = 48,
) -> Spectrum:
    """
    Isotropized kinetic energy spectrum E(k) from a 2-D velocity field.
    """
    ux = np.asarray(ux)
    uy = np.asarray(uy)
    ny, nx = ux.shape

    Ux = np.fft.fft2(ux)
    Uy = np.fft.fft2(uy)
    S = 0.5 * (np.abs(Ux) ** 2 + np.abs(Uy) ** 2) / (nx * ny) ** 2
    S = S.copy()
    S[grid.k == 0.0] = 0.0

    return _binning_helper(grid.k_norm, S, kmin=1.0, kmax=grid.N // 2 + 1)


def plot_energy_spectrum(
    spec: Spectrum,
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
    k = spec.k
    E = spec.Pk

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
