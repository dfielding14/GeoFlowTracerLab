"""
Temporal diagnostics for velocity processes.

Utilities to measure temporal structure functions of a time-varying velocity
field by probing a set of fixed spatial locations over time.

Definitions
-----------
Given a velocity field u(x, t) sampled at fixed points x_i and discrete times
t_n = n Δt, we estimate temporal structure functions

    S_p(Δt_m) = E_i,n [ |u(x_i, t_{n+m}) - u(x_i, t_n)|^p ]

for chosen orders p (e.g., 1 and 2) and time lags Δt_m = m Δt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TemporalSFResult:
    lags: np.ndarray            # Δt array (shape [M])
    orders: np.ndarray          # p array
    S: np.ndarray               # shape [len(orders), M]
    dt: float                   # base timestep used in sampling
    points: np.ndarray          # sampled probe points (y, x) integer indices (shape [P, 2])


def _choose_random_points(N: int, n_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_points, 2), endpoint=False)
    return idx.astype(np.int64, copy=False)


def _compute_temporal_structure_from_series(
    ux_series: np.ndarray,
    uy_series: np.ndarray,
    orders: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute S_p for a probe time series.

    Parameters
    ----------
    ux_series, uy_series : array, shape (T, P)
        Probe time series of velocity components at P points, T time steps.
    orders : sequence of int
        Orders p for the structure function.

    Returns
    -------
    lags_dt : ndarray, shape (M,)
        Δt in multiples of base dt (1..T-1 will be returned; user scales by dt).
    S : ndarray, shape (len(orders), M)
        Structure function values.
    """
    T, P = ux_series.shape
    max_m = max(1, T - 1)
    lags = np.arange(1, max_m + 1, dtype=np.int64)
    orders_arr = np.asarray(list(orders), dtype=np.int64)
    S = np.zeros((orders_arr.size, lags.size), dtype=np.float64)

    # Compute vector-speed increments for each lag
    for j, m in enumerate(lags):
        dux = ux_series[m:, :] - ux_series[:-m, :]
        duy = uy_series[m:, :] - uy_series[:-m, :]
        dmag = np.sqrt(dux * dux + duy * duy)
        # Average over time (axis 0) and probes (axis 1)
        for k, p in enumerate(orders_arr):
            if p == 1:
                S[k, j] = float(np.mean(dmag))
            elif p == 2:
                S[k, j] = float(np.mean(dmag * dmag))
            else:
                S[k, j] = float(np.mean(np.power(dmag, p)))
    return lags.astype(np.float64), S


def measure_temporal_structure_function(
    velocity_process,
    *,
    N: int,
    dt: float,
    n_steps: int,
    n_points: int = 64,
    seed: int = 0,
    orders: Sequence[int] = (1, 2),
    warmup_steps: int = 0,
) -> TemporalSFResult:
    """
    Probe a time-varying velocity process to estimate temporal structure functions.

    Parameters
    ----------
    velocity_process : object
        Must implement `get_velocity() -> (ux, uy)` and `step(dt)`.
    N : int
        Grid size (assumed square) of the velocity field.
    dt : float
        Sampling timestep.
    n_steps : int
        Number of stored samples (after warmup) to collect. The total number of
        calls to `step` will be warmup_steps + n_steps - 1.
    n_points : int, optional
        Number of probe points (random distinct grid points), default 64.
    seed : int, optional
        RNG seed for selecting probe points.
    orders : sequence of int, optional
        Temporal SF orders to compute (default: (1, 2)).
    warmup_steps : int, optional
        Steps to advance (without recording) before sampling to decorrelate
        initial conditions.

    Returns
    -------
    TemporalSFResult
        lags, S, orders, dt, and the sampled points.
    """
    points = _choose_random_points(N, n_points, seed)

    # Warm up the process if requested
    for _ in range(max(0, int(warmup_steps))):
        velocity_process.step(dt)

    # Record probe time series
    T = int(n_steps)
    ux0, uy0 = velocity_process.get_velocity()
    P = points.shape[0]
    ux_series = np.empty((T, P), dtype=np.float64)
    uy_series = np.empty((T, P), dtype=np.float64)

    def sample_at_points(ux: np.ndarray, uy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals_x = ux[points[:, 0], points[:, 1]]
        vals_y = uy[points[:, 0], points[:, 1]]
        return vals_x, vals_y

    ux_series[0, :], uy_series[0, :] = sample_at_points(ux0, uy0)
    for t in range(1, T):
        velocity_process.step(dt)
        ux, uy = velocity_process.get_velocity()
        ux_series[t, :], uy_series[t, :] = sample_at_points(ux, uy)

    lags_idx, S = _compute_temporal_structure_from_series(ux_series, uy_series, orders)
    lags_dt = lags_idx * float(dt)
    return TemporalSFResult(lags=lags_dt, orders=np.asarray(list(orders)), S=S, dt=float(dt), points=points)


__all__ = [
    "TemporalSFResult",
    "measure_temporal_structure_function",
]

