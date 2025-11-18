Profiling Helpers
=================

Overview
--------
The solver and FFT utilities include optional lightweight profiling to help identify hotspots.

FFT profiling
-------------
- The FFT layer can record call counts and walltime for `fft2` and `ifft2`.
- Enable via `ScalarConfig(profile_fft=True)` or programmatically in `scalar_advection.fft`:

```python
from scalar_advection.fft import enable_fft_profiling, get_fft_profile
enable_fft_profiling(True)
# ... run code that uses fft2/ifft2 ...
print(get_fft_profile(reset=True))
```

Solver timing
-------------
- Set `ScalarConfig(profile=True)` to accumulate approximate RHS/nonlinear evaluation walltime inside the solver loop (RK4/Heun paths).
- After evolution, `diagnostics.profile` (a dict) includes totals and, if FFT profiling was enabled, embedded FFT stats:

```python
cfg = ScalarConfig(peclet=1e4, t_end=0.25, profile=True, profile_fft=True)
theta_final, diag = api.solver.evolve(theta0, ux, uy, cfg)
print(diag.profile)
```

Notes
-----
- Timing is coarse-grained and wallclock-based; it’s intended for relative comparisons.
- FFTW thread count is controlled by `FFTW_THREADS` env var or `set_fftw_threads(n)`.

