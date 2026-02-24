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

    # Spatial structure-function slope: E[|v(x) - v(x+ℓ)|] ∝ ℓ^alpha
    alpha: float = 1.0 / 3.0
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

        amp = self._spectral_amplitude_alpha(config.alpha, dtype)
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

    def _spectral_amplitude_alpha(self, alpha: float, dtype) -> np.ndarray:
        """Amplitude envelope to target spatial SF slope alpha.

        With beta = 2*alpha + 1, the amplitude scales as (k/k0)^{-(alpha+1)}.
        """
        k0 = dtype(2.0 * np.pi / self.grid.L)
        amp = np.zeros_like(self.grid.k)
        mask = self.grid.k > 0.0
        amp[mask] = (self.grid.k[mask] / k0) ** (-(alpha + 1.0))
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


def _haar_Wk(N, lam, mode: str = "diag"):
    """
    Discrete 2D Haar wavelet in Fourier space.

    Parameters
    ----------
    N : int
        Grid size (NxN).
    lam : float
        Spatial scale (in grid units / pixels).
    mode : {'diag','h','v'}
        Which 2D Haar wavelet to use:
          - 'h': horizontal (sign flip across x / left-right)
          - 'v': vertical   (sign flip across y / top-bottom)
          - 'diag': diagonal checkerboard (sign flip in both x and y)

    Notes
    -----
    For an orthogonal 2D Haar basis you need all three orientations at each scale.
    """
    ksz = max(2, int(round(lam)))
    if ksz % 2:
        ksz += 1
    m = ksz // 2
    patch = np.empty((ksz, ksz), float)

    mode_l = mode.lower()
    if mode_l in ("diag", "d", "dd"):
        patch[:m, :m] = 1.0
        patch[:m, m:] = -1.0
        patch[m:, :m] = -1.0
        patch[m:, m:] = 1.0
    elif mode_l in ("h", "x", "horizontal"):
        patch[:, :m] = 1.0
        patch[:, m:] = -1.0
    elif mode_l in ("v", "y", "vertical"):
        patch[:m, :] = 1.0
        patch[m:, :] = -1.0
    else:
        raise ValueError("mode must be 'diag', 'h', or 'v'")

    patch /= np.linalg.norm(patch) + 1e-30

    wN = np.zeros((N, N), float)
    c0 = N // 2 - m
    wN[c0 : c0 + ksz, c0 : c0 + ksz] = patch
    Wk = np.fft.fft2(np.fft.ifftshift(wN))
    # L2-normalize so different scales have comparable energy before Aj scaling
    Wk /= np.sqrt(np.sum(np.abs(Wk) ** 2) / (N * N) + 1e-30)
    return Wk


def _tapered_ring_window(K: np.ndarray, kmin: float, kmax: float, taper_frac: float = 0.0) -> np.ndarray:
    """Cosine-tapered ring window in |k| between [kmin, kmax].

    Parameters
    ----------
    K : array
        |k| magnitude grid (same shape as Fourier arrays).
    kmin, kmax : float
        Inner/outer cutoff in the same units as K.
    taper_frac : float
        Fractional width of the taper region near each cutoff. Use 0 for a hard cutoff.
        A good default is ~0.1–0.2.

    Returns
    -------
    window : array
        Values in [0,1].
    """
    kmin = float(kmin)
    kmax = float(kmax)
    window = np.ones_like(K, dtype=np.float64)
    window[K < kmin] = 0.0
    window[K > kmax] = 0.0

    tf = float(taper_frac)
    if tf > 0.0:
        # Low-k ramp
        if kmin > 0.0:
            lo = (K >= kmin) & (K < kmin * (1.0 + tf))
            if np.any(lo):
                phi = (K[lo] - kmin) / (kmin * tf)
                window[lo] = 0.5 * (1.0 - np.cos(np.pi * phi))

        # High-k ramp
        if kmax > 0.0:
            hi = (K <= kmax) & (K > kmax * (1.0 - tf))
            if np.any(hi):
                phi = (kmax - K[hi]) / (kmax * tf)
                window[hi] *= 0.5 * (1.0 - np.cos(np.pi * phi))

    return window

def generate_divfree_field(
    N: int = 256,
    lam_min: float = 8,
    lam_max: float = 64,
    slope: float = -5.0 / 3.0,
    *,
    alpha: Optional[float] = None,
    wavelet: str = "mexh",
    scales_per_octave: int = 1,
    taper_frac: float = 0.0,
    haar_modes: Tuple[str, ...] = ("h", "v", "diag"),
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

octaves = float(np.log2(lam_max / lam_min))
spo = max(1, int(scales_per_octave))
n_scales = max(1, int(np.ceil(octaves * spo)) + 1)
lams = lam_min * 2.0 ** np.linspace(0.0, octaves, n_scales)
    Psi_k = np.zeros((N, N), dtype=np.complex128)

    # If `alpha` is provided, map to an equivalent spectral slope used here
    # where negative `slope` corresponds to E(k) ∝ k^{slope}.
    if alpha is not None:
        slope = -(2.0 * float(alpha) + 1.0)


for lam in lams:
    kj = 2 * np.pi / lam
    Aj = (kj / k_ref) ** ((slope - 1.0) / 2.0)

    if wavelet.lower().startswith("mex"):
        wavelet_kernels = (_mexhat_Wk(K, lam, N),)
    elif wavelet.lower().startswith("haar"):
        # A proper 2D Haar basis has three orientations per scale.
        # Summing them improves isotropy and (crucially) avoids chopping
        # off the smallest-scale energy when using a radial k-band cutoff.
        wavelet_kernels = tuple(_haar_Wk(N, lam, mode=m) for m in haar_modes)
    else:
        raise ValueError("wavelet must be 'mexh' or 'haar'.")

    for Wk in wavelet_kernels:
        Zk = np.fft.fft2(rng.normal(size=(N, N)))
        if sparsity > 0:
            mask = rng.random((N, N)) > sparsity
            Zk *= mask
        Psi_k += Aj * Wk * Zk


# A hard spectral cut introduces ringing (and therefore curvature) in real-space
# structure functions. Use a cosine-tapered band to suppress that.
kmax_eff = float(kmax)
if wavelet.lower().startswith("haar"):
    # The diagonal Haar wavelet has its strongest energy near |k| ~ sqrt(2) * 2π/λ.
    # Widen the high-k cutoff so the smallest scales aren't artificially removed.
    kmax_eff = min(float(K.max()), float(kmax) * np.sqrt(2.0))

window = _tapered_ring_window(K, kmin, kmax_eff, taper_frac=taper_frac)
Psi_k *= window

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
# Time-evolving divergence-free velocity via Fourier coefficients with
# fractional-increment dynamics (temporal structure function ∝ Δt^beta)
# ---------------------------------------------------------------------------


@dataclass
class FourierTemporalConfig:
    N: int = 256
    L: float = 1.0
    alpha: float = 1.0 / 3.0  # spatial structure exponent
    beta: Optional[float] = None  # temporal SF exponent; defaults to alpha
    kmin: Optional[float] = None
    kmax: Optional[float] = None
    taper_width: float = 0.0
    tau: float = 0.2  # relaxation to avoid unbounded drift
    sigma: float = 1.0  # noise amplitude scale
    seed: Optional[int] = None
    dtype: np.dtype = np.float64


class FourierTemporalVelocityProcess:
    """
    Time-varying divergence-free velocity built from Fourier streamfunction
    coefficients whose increments scale like dt^beta. This approximates a
    fractional-increment process per mode with optional OU damping to keep the
    field bounded.

    The resulting temporal structure function of |u(t+Δt)-u(t)| scales roughly
    like Δt^beta (for small Δt), with beta defaulting to the spatial alpha.
    """

    def __init__(self, config: FourierTemporalConfig):
        self.cfg = config
        self.N = int(config.N)
        self.L = float(config.L)
        self.alpha = float(config.alpha)
        self.beta = float(config.alpha if config.beta is None else config.beta)
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        self.dtype = np.float32 if config.dtype == np.float32 else np.float64
        self.cdtype = np.complex64 if self.dtype == np.float32 else np.complex128
        self.rng = np.random.default_rng(config.seed)
        self.t = 0.0

        # Wavenumbers and masks
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")
        self.K = np.hypot(self.KX, self.KY)

        self.kmin = None if config.kmin is None else float(config.kmin)
        self.kmax = None if config.kmax is None else float(config.kmax)
        band = np.ones((self.N, self.N), dtype=bool)
        if self.kmin is not None:
            band &= self.K >= self.kmin
        if self.kmax is not None:
            band &= self.K <= self.kmax
        self.band = band

        # Amplitude envelope for spatial structure (alpha)
        self.Ak = np.zeros((self.N, self.N), dtype=self.dtype)
        mask = self.K > 0.0
        k0 = 2.0 * np.pi / self.L
        self.Ak[mask] = (self.K[mask] / k0) ** (-(self.alpha + 1.0))
        self.Ak[~self.band] = 0.0

        # Streamfunction coefficients
        self.Psi_k = np.zeros((self.N, self.N), dtype=self.cdtype)
        self._ux: Optional[np.ndarray] = None
        self._uy: Optional[np.ndarray] = None
        self._dirty = True

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.Psi_k[...] = 0.0
        self.t = 0.0
        self._dirty = True

    def step(self, dt: float) -> None:
        dt = float(dt)
        if dt == 0.0:
            return
        rho = np.exp(-dt / max(self.cfg.tau, 1e-30))
        # Complex Gaussian noise with unit variance per real/imag component
        eta = (self.rng.normal(size=self.Psi_k.shape) + 1j * self.rng.normal(size=self.Psi_k.shape)) / np.sqrt(2.0)
        incr = (self.cfg.sigma * (dt ** self.beta)) * self.Ak * eta
        self.Psi_k = (rho * self.Psi_k + incr).astype(self.cdtype, copy=False)
        self.t += dt
        self._dirty = True

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._dirty or self._ux is None or self._uy is None:
            Ux_k = 1j * self.KY * self.Psi_k
            Uy_k = -1j * self.KX * self.Psi_k
            ux = np.fft.ifft2(Ux_k).real.astype(self.dtype, copy=False)
            uy = np.fft.ifft2(Uy_k).real.astype(self.dtype, copy=False)
            self._ux, self._uy = ux, uy
            self._dirty = False
        return self._ux, self._uy


__all__ += [
    "FourierTemporalConfig",
    "FourierTemporalVelocityProcess",
]
