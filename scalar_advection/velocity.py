"""
Velocity field generation utilities.

This module houses the configuration dataclass and generator that were
previously bundled inside ``turbulent_scalar_sim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from .fft import fft2, ifft2
from .grid import SpectralGrid


@dataclass
class VelocityConfig:
    """
    Configuration for synthetic turbulent velocity field generation.
    """

    beta: float = 5.0 / 3.0
    urms: float = 1.0
    seed: Optional[int] = None
    f_sol: float = 1.0
    kmin: Optional[float] = None
    kmax: Optional[float] = None
    taper_width: float = 0.0
    precision: str = "auto"


class VelocityFieldGenerator:
    """Generate solenoidal/potential velocity fields with prescribed spectra."""

    def __init__(self, grid: SpectralGrid):
        self.grid = grid

    def generate(self, config: VelocityConfig) -> Tuple[np.ndarray, np.ndarray]:
        if config.precision == "auto":
            base_dtype = self.grid.dtype
            dtype = np.float32 if base_dtype == np.float32 else np.float64
        elif config.precision == "float32":
            dtype = np.float32
        elif config.precision == "float64":
            dtype = np.float64
        else:
            raise ValueError("precision must be 'auto', 'float32', or 'float64'")

        cdtype = np.complex64 if dtype == np.float32 else np.complex128

        rng = np.random.default_rng(config.seed)
        N = self.grid.N

        xi1x = rng.normal(size=(N, N)).astype(dtype)
        xi1y = rng.normal(size=(N, N)).astype(dtype)
        xi2x = rng.normal(size=(N, N)).astype(dtype)
        xi2y = rng.normal(size=(N, N)).astype(dtype)

        u1x_hat = fft2(xi1x).astype(cdtype)
        u1y_hat = fft2(xi1y).astype(cdtype)
        u2x_hat = fft2(xi2x).astype(cdtype)
        u2y_hat = fft2(xi2y).astype(cdtype)

        denom = self.grid.k2.copy()
        denom[denom == 0.0] = 1.0

        kx, ky = self.grid.kx, self.grid.ky

        # Solenoidal component
        kdotu1 = kx * u1x_hat + ky * u1y_hat
        usx = u1x_hat - kx * kdotu1 / denom
        usy = u1y_hat - ky * kdotu1 / denom

        # Potential component
        kdotu2 = kx * u2x_hat + ky * u2y_hat
        upx = kx * kdotu2 / denom
        upy = ky * kdotu2 / denom

        amp = self._spectral_amplitude(config.beta, dtype)
        window = self._compute_band_window(config.kmin, config.kmax, config.taper_width, dtype)

        usx *= amp * window
        usy *= amp * window
        upx *= amp * window
        upy *= amp * window

        Es = 0.5 * np.sum(np.abs(usx) ** 2 + np.abs(usy) ** 2)
        Ep = 0.5 * np.sum(np.abs(upx) ** 2 + np.abs(upy) ** 2)

        a = np.sqrt(max(config.f_sol, 0.0) / max(Es, 1e-30))
        b = np.sqrt(max(1.0 - config.f_sol, 0.0) / max(Ep, 1e-30))

        ux_hat = (a * usx + b * upx).astype(cdtype)
        uy_hat = (a * usy + b * upy).astype(cdtype)

        ux_hat[0, 0] = 0.0
        uy_hat[0, 0] = 0.0

        ux = ifft2(ux_hat).real.astype(dtype)
        uy = ifft2(uy_hat).real.astype(dtype)

        ur = np.sqrt(np.mean(ux**2 + uy**2))
        if ur > 0:
            s = config.urms / ur
            ux *= s
            uy *= s

        return ux, uy

    def _spectral_amplitude(self, beta: float, dtype) -> np.ndarray:
        k0 = dtype(2.0 * np.pi / self.grid.L)
        amp = np.zeros_like(self.grid.k)
        mask = self.grid.k > 0.0
        amp[mask] = (self.grid.k[mask] / k0) ** (-(1.0 + beta) / 2.0)
        return amp

    def _compute_band_window(
        self,
        kmin: Optional[float],
        kmax: Optional[float],
        taper_width: float,
        dtype,
    ) -> np.ndarray:
        if kmin is not None:
            kmin = float(np.clip(kmin, 1.0, self.grid.N / 2))
        if kmax is not None:
            kmax = float(np.clip(kmax, 1.0, self.grid.N / 2))

        if (kmin is not None) and (kmax is not None) and (kmin >= kmax):
            kmax = kmin

        window = np.ones_like(self.grid.k_norm, dtype=dtype)

        if kmin is not None:
            window[self.grid.k_norm < kmin] = 0.0
            if taper_width > 0:
                lo = (self.grid.k_norm >= kmin) & (self.grid.k_norm < kmin + taper_width)
                phi = (self.grid.k_norm[lo] - kmin) / taper_width
                window[lo] = 0.5 * (1.0 - np.cos(np.pi * phi))

        if kmax is not None:
            window[self.grid.k_norm > kmax] = 0.0
            if taper_width > 0:
                hi = (self.grid.k_norm <= kmax) & (self.grid.k_norm > kmax - taper_width)
                phi = (kmax - self.grid.k_norm[hi]) / taper_width
                window[hi] *= 0.5 * (1.0 - np.cos(np.pi * phi))

        return window


def generate_velocity_field(grid: SpectralGrid, config: VelocityConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper around :class:`VelocityFieldGenerator`.
    """
    return VelocityFieldGenerator(grid).generate(config)


# ---------------------------------------------------------------------------
# Wavelet-based velocity generator
# ---------------------------------------------------------------------------


def _mexhat_Wk(k, lam, N):
    s = np.sqrt(2.0) * lam / (2.0 * np.pi)
    Wk = (k * s) ** 2 * np.exp(-0.5 * (k * s) ** 2)
    Wk[k == 0] = 0.0
    Wk *= np.sqrt((N * N) / (np.sum(np.abs(Wk) ** 2) + 1e-30))
    return Wk


def _haar_Wk(N, lam):
    ksz = max(2, int(round(lam)))
    if ksz % 2:
        ksz += 1
    m = ksz // 2
    patch = np.empty((ksz, ksz), float)
    patch[:m, :m] = 1.0
    patch[:m, m:] = -1.0
    patch[m:, :m] = -1.0
    patch[m:, m:] = 1.0
    patch /= np.linalg.norm(patch) + 1e-30

    wN = np.zeros((N, N), float)
    c0 = N // 2 - m
    wN[c0 : c0 + ksz, c0 : c0 + ksz] = patch
    Wk = np.fft.fft2(np.fft.ifftshift(wN))
    Wk /= np.sqrt(np.sum(np.abs(Wk) ** 2) / (N * N) + 1e-30)
    return Wk


def generate_divfree_field(
    N: int = 256,
    lam_min: float = 8,
    lam_max: float = 64,
    slope: float = -5.0 / 3.0,
    wavelet: str = "mexh",
    lam_ref: Optional[float] = None,
    amp: float = 1.0,
    sparsity: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wavelet-based incompressible velocity generator used for exploration.
    """
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * np.fft.fftfreq(N)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.hypot(KX, KY)
    kmin = 2 * np.pi / lam_max
    kmax = 2 * np.pi / lam_min
    if lam_ref is None:
        lam_ref = np.sqrt(lam_min * lam_max)
    k_ref = 2 * np.pi / lam_ref
    n_scales = max(1, int(np.ceil(np.log2(lam_max / lam_min))) + 1)
    lams = lam_min * 2.0 ** np.linspace(0, np.log2(lam_max / lam_min), n_scales)
    Psi_k = np.zeros((N, N), dtype=np.complex128)

    for lam in lams:
        if wavelet.lower().startswith("mex"):
            Wk = _mexhat_Wk(K, lam, N)
        elif wavelet.lower().startswith("haar"):
            Wk = _haar_Wk(N, lam)
        else:
            raise ValueError("wavelet must be 'mexh' or 'haar'.")
        kj = 2 * np.pi / lam
        Aj = (kj / k_ref) ** ((slope - 1.0) / 2.0)
        Zk = np.fft.fft2(rng.normal(size=(N, N)))
        if sparsity > 0:
            mask = rng.random((N, N)) > sparsity
            Zk *= mask
        Psi_k += Aj * Wk * Zk

    band = (K >= kmin) & (K <= kmax)
    Psi_k *= band

    Ux_k = 1j * KY * Psi_k
    Uy_k = -1j * KX * Psi_k
    ux = np.fft.ifft2(Ux_k).real
    uy = np.fft.ifft2(Uy_k).real

    spd = np.hypot(ux, uy)
    cur = spd.std()
    if cur > 0:
        scale = amp / cur
        ux *= scale
        uy *= scale
        spd = np.hypot(ux, uy)
    return ux, uy, spd


__all__ = [
    "VelocityConfig",
    "VelocityFieldGenerator",
    "generate_velocity_field",
    "generate_divfree_field",
]

# ---------------------------------------------------------------------------
# Time-evolving wavelet velocity via Ornstein–Uhlenbeck process
# ---------------------------------------------------------------------------


@dataclass
class WaveletOUConfig:
    """
    Configuration for a time-evolving, divergence-free wavelet velocity field
    where the underlying stochastic coefficients follow an
    Ornstein–Uhlenbeck (OU) process with correlation timescale ``tau``.

    Notes
    -----
    - The spatial field is built from a superposition of wavelet bands
      between ``lam_min`` and ``lam_max``. Within each time step, the
      streamfunction's per-band coefficient field is updated using the exact
      OU discretization z_{t+dt} = rho z_t + sqrt(1-rho^2) * eta, with
      rho = exp(-dt/tau) and eta ~ N(0,1) i.i.d. in physical space.
    - The resulting velocity is incompressible by construction and is scaled
      to have approximately ``amp`` speed standard deviation each time it is
      synthesized (matching the static generator convention).
    """

    N: int = 256
    lam_min: float = 8.0
    lam_max: float = 64.0
    slope: float = -5.0 / 3.0
    wavelet: str = "mexh"  # or 'haar'
    lam_ref: Optional[float] = None
    amp: float = 1.0
    sparsity: float = 0.0  # probability of zeroing Fourier coefficients per band
    seed: Optional[int] = None
    tau: float = 0.1  # OU correlation timescale (simulation time units)
    dtype: np.dtype = np.float64


class WaveletOUTemporalVelocity:
    """
    Time-evolving divergence-free velocity built from wavelets with OU dynamics.

    Usage
    -----
    >>> cfg = WaveletOUConfig(N=256, lam_min=4, lam_max=64, tau=0.2, amp=1.0)
    >>> vel = WaveletOUTemporalVelocity(cfg)
    >>> ux, uy = vel.get_velocity()  # t = 0
    >>> vel.step(0.01)
    >>> ux2, uy2 = vel.get_velocity()  # t = 0.01
    """

    def __init__(self, config: WaveletOUConfig):
        self.cfg = config
        self.N = int(config.N)
        self.dtype = np.float32 if config.dtype == np.float32 else np.float64
        self.cdtype = np.complex64 if self.dtype == np.float32 else np.complex128

        rng = np.random.default_rng(config.seed)
        self.rng = rng
        self.t = 0.0

        # Wavenumbers and helpers
        kx = 2 * np.pi * np.fft.fftfreq(self.N)
        ky = 2 * np.pi * np.fft.fftfreq(self.N)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")
        self.K = np.hypot(self.KX, self.KY)

        # Band limits and reference scale
        lam_min = float(config.lam_min)
        lam_max = float(config.lam_max)
        if lam_min <= 0 or lam_max <= 0 or lam_min > lam_max:
            raise ValueError("Require 0 < lam_min <= lam_max")
        self.kmin = 2 * np.pi / lam_max
        self.kmax = 2 * np.pi / lam_min
        lam_ref = np.sqrt(lam_min * lam_max) if config.lam_ref is None else float(config.lam_ref)
        self.k_ref = 2 * np.pi / lam_ref

        n_scales = max(1, int(np.ceil(np.log2(lam_max / lam_min))) + 1)
        self.lams = lam_min * 2.0 ** np.linspace(0, np.log2(lam_max / lam_min), n_scales)

        # Precompute wavelet frequency responses Wk and band amplitudes Aj
        self.Wk: List[np.ndarray] = []
        self.Aj: List[float] = []
        for lam in self.lams:
            if config.wavelet.lower().startswith("mex"):
                Wk = _mexhat_Wk(self.K, lam, self.N)
            elif config.wavelet.lower().startswith("haar"):
                Wk = _haar_Wk(self.N, lam)
            else:
                raise ValueError("wavelet must be 'mexh' or 'haar'.")
            kj = 2 * np.pi / lam
            Aj = (kj / self.k_ref) ** ((config.slope - 1.0) / 2.0)
            self.Wk.append(Wk.astype(self.cdtype, copy=False))
            self.Aj.append(float(Aj))

        # Band-pass mask in k-space
        self.band = (self.K >= self.kmin) & (self.K <= self.kmax)

        # Optional sparsity masks in Fourier space, fixed in time
        self.masks_k: List[Optional[np.ndarray]] = []
        if config.sparsity > 0:
            for _ in self.lams:
                m = (rng.random((self.N, self.N)) > config.sparsity)
                self.masks_k.append(m)
        else:
            self.masks_k = [None for _ in self.lams]

        # OU state per scale in physical space (real fields)
        self.z_fields: List[np.ndarray] = [rng.normal(size=(self.N, self.N)).astype(self.dtype) for _ in self.lams]

        # Cached velocity
        self._ux: Optional[np.ndarray] = None
        self._uy: Optional[np.ndarray] = None
        self._dirty = True

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset OU state and time (t=0)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.z_fields = [rng.normal(size=(self.N, self.N)).astype(self.dtype) for _ in self.lams]
        self.t = 0.0
        self._dirty = True

    def step(self, dt: float) -> None:
        """Advance the OU process by time step ``dt`` using exact update."""
        dt = float(dt)
        if dt == 0.0:
            return
        rho = np.exp(-dt / max(self.cfg.tau, 1e-30))
        sig = np.sqrt(max(0.0, 1.0 - rho * rho))
        for j in range(len(self.z_fields)):
            eta = self.rng.normal(size=self.z_fields[j].shape).astype(self.dtype)
            self.z_fields[j] = (rho * self.z_fields[j] + sig * eta).astype(self.dtype, copy=False)
        self.t += dt
        self._dirty = True

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current velocity field (ux, uy)."""
        if self._dirty or self._ux is None or self._uy is None:
            self._synthesize_velocity()
        return self._ux, self._uy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _synthesize_velocity(self) -> None:
        Psi_k = np.zeros((self.N, self.N), dtype=self.cdtype)
        for j, lam in enumerate(self.lams):
            Zk = np.fft.fft2(self.z_fields[j]).astype(self.cdtype, copy=False)
            if self.masks_k[j] is not None:
                Zk *= self.masks_k[j]
            Psi_k += self.Aj[j] * self.Wk[j] * Zk

        Psi_k *= self.band
        Ux_k = 1j * self.KY * Psi_k
        Uy_k = -1j * self.KX * Psi_k
        ux = np.fft.ifft2(Ux_k).real.astype(self.dtype, copy=False)
        uy = np.fft.ifft2(Uy_k).real.astype(self.dtype, copy=False)

        spd = np.hypot(ux, uy)
        cur = float(spd.std())
        if cur > 0:
            scale = float(self.cfg.amp) / cur
            ux *= scale
            uy *= scale

        self._ux = ux
        self._uy = uy
        self._dirty = False


__all__ += [
    "WaveletOUConfig",
    "WaveletOUTemporalVelocity",
]
