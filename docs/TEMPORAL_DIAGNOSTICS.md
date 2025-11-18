Temporal Velocity Diagnostics
=============================

Overview
--------
This note describes how to measure temporal structure functions of a time-varying velocity process using fixed spatial probes. For a process with target temporal scaling parameter `beta`, we expect

    S_p(Δt) = E[ |u(t+Δt) - u(t)|^p ] ∝ (Δt)^{p·beta}

for small Δt (in practice, we often focus on p=1 or p=2 and fit slopes on a log–log plot).

API
---
Use `measure_temporal_structure_function` to sample a velocity process and compute S_p(Δt):

- The process must implement `get_velocity() -> (ux, uy)` and `step(dt)`.
- Sampling runs at a fixed `dt` for `n_steps`, and probes `n_points` random locations on the grid.

Example
-------
```python
from scalar_advection import (
    FourierTemporalConfig,
    FourierTemporalVelocityProcess,
    measure_temporal_structure_function,
)

cfg = FourierTemporalConfig(N=256, L=1.0, alpha=1/3, beta=None, tau=0.2, sigma=1.0, seed=1)
proc = FourierTemporalVelocityProcess(cfg)

res = measure_temporal_structure_function(proc, N=cfg.N, dt=1e-3, n_steps=500, n_points=128, seed=0, orders=(1,2), warmup_steps=100)

# res.lags (Δt), res.orders, res.S[0] (S1), res.S[1] (S2)
```

Interpreting results
--------------------
- Plot `S1` and/or `S2` vs `Δt` on log–log axes and fit a slope over a clean intermediate range.
- With `beta = alpha` (default), you should see slopes near `beta` (for S1) and `2*beta` (for S2) at sufficiently small Δt.

Notes
-----
- The FourierTemporalVelocityProcess is a simple fractional-increment model with optional OU damping; it aims for approximate temporal scaling while preserving incompressibility via a streamfunction.
- Increase `n_steps` and reduce `dt` to widen the scaling window; use `warmup_steps` to decorrelate initialization.

