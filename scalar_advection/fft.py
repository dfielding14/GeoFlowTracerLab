"""
Shared FFT utilities with optional FFTW acceleration.

This module mirrors the lazy backend selection found in the original
``turbulent_scalar_sim`` script so that both the velocity generator and
scalar solver reuse the same functions without duplicating logic.
"""

from __future__ import annotations

import os
import numpy as np

FFTW_THREADS = int(os.environ.get("FFTW_THREADS", "4"))

try:  # pragma: no cover - relies on optional dependency
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft2 as _fft2
    from pyfftw.interfaces.numpy_fft import ifft2 as _ifft2

    pyfftw.interfaces.cache.enable()

    def fft2(a):
        """2D FFT using FFTW if available."""
        return _fft2(a, threads=FFTW_THREADS)

    def ifft2(a):
        """2D inverse FFT using FFTW if available."""
        return _ifft2(a, threads=FFTW_THREADS)

    FFT_BACKEND = "FFTW"
except ImportError:  # pragma: no cover - falls back automatically
    from numpy.fft import fft2, ifft2  # type: ignore  # noqa: F401

    FFT_BACKEND = "NumPy"

def set_fftw_threads(n: int) -> None:
    """
    Update the number of threads used by the FFTW backend.

    Parameters
    ----------
    n : int
        Desired number of threads (>=1). Ignored when FFTW is unavailable.
    """
    global FFTW_THREADS
    FFTW_THREADS = max(1, int(n))


def warm_fft_cache(shape, dtype=np.float64) -> None:
    """
    Perform dummy FFTs to warm plan caches for the given array shape.

    Parameters
    ----------
    shape : tuple[int, int]
        Array shape to warm.
    dtype : np.dtype
        Real-space dtype to emulate (np.float64 by default).
    """
    arr = np.zeros(shape, dtype=dtype)
    coeffs = fft2(arr)
    _ = ifft2(coeffs)


__all__ = ["fft2", "ifft2", "FFT_BACKEND", "FFTW_THREADS", "set_fftw_threads", "warm_fft_cache"]
