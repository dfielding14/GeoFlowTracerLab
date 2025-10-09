"""
Spectral grid utilities shared across velocity generation and the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralGrid:
    """
    Pre-computed spectral quantities for a periodic square domain.

    Parameters
    ----------
    N : int
        Number of grid points per dimension (must be even to support dealiasing).
    L : float
        Domain size.
    dtype : np.dtype
        Floating-point dtype for physical-space arrays.
    """

    N: int
    L: float
    dtype: np.dtype = np.float64

    def __post_init__(self) -> None:
        if self.N % 2 != 0:
            raise ValueError("Grid resolution N must be even")

        object.__setattr__(self, "cdtype", np.complex64 if self.dtype == np.float32 else np.complex128)
        object.__setattr__(self, "dx", self.L / self.N)

        k1 = (2.0 * np.pi) * np.fft.fftfreq(self.N, d=self.dx)
        k1 = k1.astype(self.dtype)
        kx, ky = np.meshgrid(k1, k1, indexing="xy")
        object.__setattr__(self, "kx", kx)
        object.__setattr__(self, "ky", ky)
        k2 = kx**2 + ky**2
        object.__setattr__(self, "k2", k2)
        k = np.sqrt(k2)
        object.__setattr__(self, "k", k)
        k_norm = k / (2.0 * np.pi / self.L)
        object.__setattr__(self, "k_norm", k_norm)

        kcut = np.max(np.abs(k1)) * (2.0 / 3.0)
        dealias_mask = (np.abs(kx) <= kcut) & (np.abs(ky) <= kcut)
        object.__setattr__(self, "dealias_mask", dealias_mask)

    def zeros(self, *, complex_: bool = False):
        """Return a zero array shaped like the grid."""
        dtype = self.cdtype if complex_ else self.dtype
        return np.zeros((self.N, self.N), dtype=dtype)


__all__ = ["SpectralGrid"]
