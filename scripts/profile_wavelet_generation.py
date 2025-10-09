#!/usr/bin/env python3
"""
Standalone runtime probe for ``generate_divfree_field``.

This replicates the wavelet-based velocity setup used in
``examples/03_flow_statistics.ipynb`` so we can time and diagnose the cost
outside of the notebook environment.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scalar_advection import generate_divfree_field, set_fftw_threads, warm_fft_cache


def main() -> None:
    N = 512
    params = dict(
        N=N,
        lam_min=6,
        lam_max=N // 2,
        slope=-5.0 / 3.0,
        wavelet="mexh",
        sparsity=0.0,
        seed=42,
    )

    # Mirror the notebook setup: enable FFT threading and warm the cache.
    set_fftw_threads(8)
    warm_fft_cache((N, N))

    start = time.perf_counter()
    ux, uy, speed = generate_divfree_field(**params)
    elapsed = time.perf_counter() - start

    print(f"Wavelet field generated in {elapsed:.3f} s")
    print(f"ux stats: mean={ux.mean():+.3e}, std={ux.std():.3e}")
    print(f"uy stats: mean={uy.mean():+.3e}, std={uy.std():.3e}")
    print(f"|u| stats: mean={speed.mean():+.3e}, std={speed.std():.3e}")


if __name__ == "__main__":
    main()
