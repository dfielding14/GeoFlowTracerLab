# GeoFlowTracerLab – Development TODO

This checklist organizes the upcoming work around performance, diagnostics, and the velocity modeling updates you requested. Items are grouped and ordered roughly by dependency and impact.

## Performance

- [ ] Profiling: add lightweight timers around FFTs and nonlinear term to identify hotspots (small helper in `solver.py`).
- [ ] FFT threading: expose `FFTW_THREADS` and ensure `set_fft_threads()` is called early in scripts/notebooks.
- [ ] Float32 path: end-to-end float32 simulation option (solver, velocity, stats) with clear validation vs float64.
- [ ] Allocate-once arrays in solver: reuse work arrays in `_nonlinear_term` to reduce allocations.
- [ ] Avoid repeated FFT plans: add optional warm-up at API init and reuse pyFFTW plans when available.
- [ ] Dealias mask precompute: ensure `grid.dealias_mask` is dtype- and backend-compatible without casts in loop.
- [ ] Vectorize structure/statistics: batch displacement evaluations to cut Python overhead; consider numba for hotspots.
- [ ] I/O efficiency: use `.npz` compressed snapshots optionally; avoid writing large intermediates by default.

## Diagnostics – Scalar Energy Dissipation

- [x] Add time-resolved scalar dissipation rate output: store `epsilon_theta(t) = 2 kappa <|∇θ|^2>` each step.
- [x] Return and persist dissipation time series in `SimulationDiagnostics` (array + cumulative integral).
- [x] Plot helpers: quick plot of `epsilon_theta(t)` and cumulative dissipation vs time; save alongside runs.
- [x] Back-fill into `run_wavelet_scalar_experiment.py` and single-run driver; include in summaries.

## Velocity Model – Spatial Structure Function Slope

- [x] Replace “spectrum slope beta” control with “structure-function slope alpha” for spatial increments.
- [x] Derive mapping alpha → spectral amplitude law A(k): A(k) ∝ k^{-(alpha+1)}.
- [x] Implement generator parameter `alpha` (removed beta altogether for spatial).
- [ ] Validate mapping numerically: generate velocity, measure structure functions, and verify slope across scales.
- [ ] Update `ScalarAdvectionAPI.generate_velocity()` to accept `alpha`-based config and docstring.

## Velocity Model – Temporal Structure Function

- [x] Implement time-varying Fourier-based velocity with random coefficients scaled to enforce temporal SF ∝ (Δt)^beta.
- [x] Default `beta = alpha`; allow user override; document valid ranges and physical interpretation.
- [x] Provide `TemporalVelocityProcess` interface (get_velocity/step) as `FourierTemporalVelocityProcess`.
- [x] Add diagnostics to measure temporal SF from point probes and confirm scaling.
- [ ] Optionally compare with OU and add scale-dependent τ_k.

## Experiments/CLI

- [x] Update experiment scripts to accept `alpha` (spatial) and plumb into generators.
- [ ] Add CLI path to run time-varying velocity (FourierTemporal) with `beta` (temporal) parameter.
- [ ] Switch defaults and CLI help text to new semantics; keep compatibility flags for beta-based runs.
- [ ] Add command to export diagnostics (spectra, structure, dissipation) for both velocity and scalar consistently.
- [ ] Ensure outputs write under `experimental_results/` and create per-run metadata with versions/config.

## Docs & Notebooks

- [ ] Update README and docs to describe `alpha`/`beta` parameters and dissipation diagnostics.
- [x] Add temporal diagnostics HOWTO (docs/TEMPORAL_DIAGNOSTICS.md).
- [x] Add profiling HOWTO (docs/PROFILING.md).
- [ ] Refresh notebooks to import scripts from `experiments/` and read from `experimental_results/`.
- [ ] Add short HOWTO for running time-varying velocity experiments and interpreting outputs.

## Testing & Validation

- [ ] Unit tests for mapping alpha ↔ observed structure slopes (tolerant checks across scales).
- [ ] Regression tests for solver correctness (energy/dissipation conservation limits, mean-gradient forcing sanity).
- [ ] Quick CI job to lint and run small N=64 sanity simulations (no heavy I/O).

## Housekeeping

- [ ] Add `.gitignore` entries for `experimental_results/` and large artifacts; keep examples/ for notebooks only.
- [ ] Pre-commit hook for stripping `.DS_Store`, `__pycache__`, and large files from commits.
- [ ] Optional: lightweight configuration registry for experiments to ensure reproducible parameter sets.

Please review and reprioritize. I’ll start with diagnostics (dissipation), then the alpha-based velocity mapping, followed by the time-varying temporal SF implementation, while applying performance wins opportunistically.
