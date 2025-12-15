#!/usr/bin/env python3
"""Quick benchmark of FFT thread scaling for 2048^2 grid."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scalar_advection import ScalarAdvectionAPI, ScalarConfig, generate_divfree_field

GRID = 2048
T_END = 1e-3
ALPHA = 1/3
KAPPA = 1e-3
THREAD_COUNTS = [64, 124]

def run_benchmark(n_threads):
    dtype = np.float32
    api = ScalarAdvectionAPI(N=GRID, L=1.0, dtype=dtype, warm_cache=True)
    api.set_fft_threads(n_threads)
    
    ux, uy, _ = generate_divfree_field(
        N=GRID, lam_min=2.0, lam_max=GRID, alpha=ALPHA,
        wavelet="mexh", sparsity=0.0, seed=42
    )
    ux = ux.astype(dtype)
    uy = uy.astype(dtype)
    
    theta0 = np.zeros((GRID, GRID), dtype=dtype)
    cfg = ScalarConfig(
        kappa=KAPPA, mean_grad=(1.0, 0.0), t_end=T_END,
        cfl=0.6, integrator="heun", save_every=None
    )
    
    # Warm-up run
    _, _ = api.evolve_scalar(theta0.copy(), ux, uy, cfg, verbose=False)
    
    # Timed run
    start = time.perf_counter()
    _, diag = api.evolve_scalar(theta0.copy(), ux, uy, cfg, verbose=False)
    elapsed = time.perf_counter() - start
    
    return elapsed, diag.n_steps

if __name__ == "__main__":
    print(f"Benchmarking 2048^2 grid, t_end={T_END}, dtype=float32, integrator=heun")
    print(f"{'Threads':>8} {'Time (s)':>10} {'Steps':>8} {'ms/step':>10}")
    print("-" * 40)
    
    results = []
    for n in THREAD_COUNTS:
        elapsed, steps = run_benchmark(n)
        ms_per_step = 1000 * elapsed / steps
        results.append((n, elapsed, steps, ms_per_step))
        print(f"{n:>8} {elapsed:>10.3f} {steps:>8} {ms_per_step:>10.2f}")
    
    # Summary
    best = min(results, key=lambda x: x[1])
    print("-" * 40)
    print(f"Fastest: {best[0]} threads ({best[1]:.3f}s)")
