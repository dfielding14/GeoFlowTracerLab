"""
Shared FFT utilities with optional FFTW acceleration.

This module mirrors the lazy backend selection found in the original
``turbulent_scalar_sim`` script so that both the velocity generator and
scalar solver reuse the same functions without duplicating logic.
"""

from __future__ import annotations

import os
import time
import numpy as np

FFTW_THREADS = int(os.environ.get("FFTW_THREADS", "4"))

_FFT_PROFILE_ENABLED = False
_FFT_PROFILE = {"fft2_calls": 0, "ifft2_calls": 0, "fft2_time": 0.0, "ifft2_time": 0.0}


def enable_fft_profiling(enabled: bool = True) -> None:
    global _FFT_PROFILE_ENABLED
    _FFT_PROFILE_ENABLED = bool(enabled)


def get_fft_profile(reset: bool = False) -> dict:
    prof = dict(_FFT_PROFILE)
    if reset:
        _FFT_PROFILE.update({"fft2_calls": 0, "ifft2_calls": 0, "fft2_time": 0.0, "ifft2_time": 0.0})
    return prof


try:  # pragma: no cover - relies on optional dependency
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft2 as _fft2_impl
    from pyfftw.interfaces.numpy_fft import ifft2 as _ifft2_impl

    pyfftw.interfaces.cache.enable()

    def fft2(a):
        """2D FFT using FFTW if available, with optional timing."""
        if _FFT_PROFILE_ENABLED:
            t0 = time.perf_counter()
            out = _fft2_impl(a, threads=FFTW_THREADS)
            dt = time.perf_counter() - t0
            _FFT_PROFILE["fft2_calls"] += 1
            _FFT_PROFILE["fft2_time"] += dt
            return out
        return _fft2_impl(a, threads=FFTW_THREADS)

    def ifft2(a):
        """2D inverse FFT using FFTW if available, with optional timing."""
        if _FFT_PROFILE_ENABLED:
            t0 = time.perf_counter()
            out = _ifft2_impl(a, threads=FFTW_THREADS)
            dt = time.perf_counter() - t0
            _FFT_PROFILE["ifft2_calls"] += 1
            _FFT_PROFILE["ifft2_time"] += dt
            return out
        return _ifft2_impl(a, threads=FFTW_THREADS)

    FFT_BACKEND = "FFTW"
except ImportError:  # pragma: no cover - falls back automatically
    from numpy.fft import fft2 as _fft2_impl, ifft2 as _ifft2_impl  # type: ignore  # noqa: F401

    FFT_BACKEND = "NumPy"
    def fft2(a):
        if _FFT_PROFILE_ENABLED:
            t0 = time.perf_counter()
            out = _fft2_impl(a)
            dt = time.perf_counter() - t0
            _FFT_PROFILE["fft2_calls"] += 1
            _FFT_PROFILE["fft2_time"] += dt
            return out
        return _fft2_impl(a)

    def ifft2(a):
        if _FFT_PROFILE_ENABLED:
            t0 = time.perf_counter()
            out = _ifft2_impl(a)
            dt = time.perf_counter() - t0
            _FFT_PROFILE["ifft2_calls"] += 1
            _FFT_PROFILE["ifft2_time"] += dt
            return out
        return _ifft2_impl(a)

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


__all__ = [
    "fft2",
    "ifft2",
    "FFT_BACKEND",
    "FFTW_THREADS",
    "set_fftw_threads",
    "warm_fft_cache",
    "enable_fft_profiling",
    "get_fft_profile",
]
