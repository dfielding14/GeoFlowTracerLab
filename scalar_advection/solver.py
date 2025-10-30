"""
Scalar advection–diffusion solver (ETDRK4 pseudo-spectral).

Extracted from the monolithic ``turbulent_scalar_sim`` script so it can be
reused independently.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fft import FFT_BACKEND, fft2, ifft2
from .fft import enable_fft_profiling, get_fft_profile
from .grid import SpectralGrid


@dataclass
class ScalarConfig:
    """Configuration for scalar advection-diffusion simulation.

    Parameters
    ----------
    integrator : str
        Time integrator to use: ``'etdrk4'`` (default), ``'rk4'``, or ``'heun'``.
    """

    t_end: Optional[float] = None
    t_advective_mult: float = 1.0
    dt: Optional[float] = None
    cfl: float = 0.5
    kappa: Optional[float] = None
    peclet: Optional[float] = 1000.0
    mean_grad: Tuple[float, float] = (0.0, 0.0)
    save_every: Optional[int] = None
    output_frames: bool = False
    frame_interval: Optional[float] = None
    save_to_disk: bool = False
    save_dir: Optional[str] = None
    integrator: str = "etdrk4"
    # Profiling controls
    profile: bool = False
    profile_fft: bool = False


@dataclass
class SimulationDiagnostics:
    """Data collected during scalar evolution."""

    snapshots: List[np.ndarray] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: float = 0.0
    kappa: float = 0.0
    n_steps: int = 0
    frames: Optional[List[np.ndarray]] = None
    grad_sq_integral: float = 0.0
    # Full time series diagnostics
    times_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_sq_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    dissipation_ts: np.ndarray = field(default_factory=lambda: np.array([]))


class ScalarAdvectionDiffusionSolver:
    """
    Pseudo-spectral ETDRK4 solver for passive scalar advection-diffusion.
    """

    def __init__(self, grid: SpectralGrid):
        self.grid = grid
        self.dtype = grid.dtype
        self.cdtype = grid.cdtype
        # Work buffers (allocated lazily)
        self._tmp_hat = None  # complex buffer for k-space products
        self._adv_buf = None  # real buffer for advection term in physical space
        # Precomputed complex wavenumber factors (for reuse in derivatives)
        self._ikx = (1j * self.grid.kx).astype(self.cdtype)
        self._iky = (1j * self.grid.ky).astype(self.cdtype)

    def _ensure_work_buffers(self) -> None:
        if self._tmp_hat is None or self._tmp_hat.shape != (self.grid.N, self.grid.N):
            self._tmp_hat = np.empty((self.grid.N, self.grid.N), dtype=self.cdtype)
        if self._adv_buf is None or self._adv_buf.shape != (self.grid.N, self.grid.N):
            self._adv_buf = np.empty((self.grid.N, self.grid.N), dtype=self.dtype)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def create_circle_initial_condition(
        self,
        radius: float = 0.25,
        center: Tuple[float, float] = (0.0, 0.0),
        val_in: float = 1.0,
        val_out: float = 0.0,
    ) -> np.ndarray:
        x = np.linspace(-self.grid.L / 2, self.grid.L / 2, self.grid.N, endpoint=False)
        y = np.linspace(-self.grid.L / 2, self.grid.L / 2, self.grid.N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="xy")
        r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return np.where(r <= radius, val_in, val_out).astype(self.dtype)

    def create_random_initial_condition(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta = rng.normal(0, 1, (self.grid.N, self.grid.N)).astype(self.dtype)
        return theta

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def compute_scalar_dissipation(self, theta: np.ndarray, kappa: float) -> float:
        # Accumulate in float64 to avoid overflow in square/add
        theta_hat = fft2(theta)
        theta_x = ifft2(1j * self.grid.kx * theta_hat).real.astype(np.float64, copy=False)
        theta_y = ifft2(1j * self.grid.ky * theta_hat).real.astype(np.float64, copy=False)
        val = np.mean(theta_x * theta_x + theta_y * theta_y, dtype=np.float64)
        return 2.0 * float(kappa) * float(val)

    @staticmethod
    def load_snapshots(snapshot_dir: str) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
        import glob

        metadata_file = os.path.join(snapshot_dir, "metadata.npy")
        if os.path.exists(metadata_file):
            metadata = np.load(metadata_file, allow_pickle=True).item()
        else:
            metadata = {}

        snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "theta_*.npy")))
        snapshots = []
        times = []
        for filepath in snapshot_files:
            filename = os.path.basename(filepath)
            time_str = filename.split("_t")[1].replace(".npy", "")
            times.append(float(time_str))
            snapshots.append(np.load(filepath))
        return snapshots, np.array(times), metadata

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def evolve(
        self,
        theta0: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        config: ScalarConfig,
        *,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, SimulationDiagnostics]:
        theta0 = np.asarray(theta0, dtype=self.dtype)
        ux = np.asarray(ux, dtype=self.dtype)
        uy = np.asarray(uy, dtype=self.dtype)

        dt, nsteps, kappa, t_end = self._resolve_time_controls(theta0, ux, uy, config)
        diagnostics = SimulationDiagnostics(dt=dt, kappa=kappa, n_steps=nsteps)
        if config.output_frames:
            diagnostics.frames = []
        # Enable FFT profiling if requested
        if getattr(config, "profile_fft", False):
            enable_fft_profiling(True)

        # Walltime accumulators
        nl_time_total = 0.0

        theta_hat = fft2(theta0).astype(self.cdtype)
        Llin = -kappa * self.grid.k2

        Gx, Gy = config.mean_grad
        if Gx != 0.0 or Gy != 0.0:
            F = -(ux * Gx + uy * Gy)
            F_hat = fft2(F).astype(self.cdtype)
        else:
            F_hat = None

        grad_sq_prev = self._mean_grad_sq(theta_hat)
        # Initialize time-series at t=0
        ts_times: List[float] = [0.0]
        ts_grad_sq: List[float] = [float(grad_sq_prev)]
        ts_eps: List[float] = [float(2.0 * kappa * grad_sq_prev)]

        integrator = (config.integrator or "etdrk4").lower()
        if integrator not in {"etdrk4", "rk4", "heun"}:
            raise ValueError("integrator must be 'etdrk4', 'rk4', or 'heun'")

        if integrator == "etdrk4":
            E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(Llin, dt)

        def rhs(theta_hat_state: np.ndarray) -> np.ndarray:
            return Llin * theta_hat_state + self._nonlinear_term(theta_hat_state, ux, uy, F_hat)

        snapshot_dir = None
        snapshot_count = 0
        if config.save_to_disk:
            snapshot_dir = config.save_dir or self._auto_snapshot_dir()
            os.makedirs(snapshot_dir, exist_ok=True)

        diagnostics.times = np.append(diagnostics.times, 0.0)
        if config.save_every is not None:
            if config.save_to_disk and snapshot_dir:
                np.save(os.path.join(snapshot_dir, "theta_00000_t0.0000.npy"), theta0)
            else:
                diagnostics.snapshots.append(theta0.copy())

        frame_step = None
        if config.output_frames and config.frame_interval is not None:
            frame_step = max(1, int(round(config.frame_interval / dt)))

        for n in range(1, nsteps + 1):
            if integrator == "etdrk4":
                Nv = self._nonlinear_term(theta_hat, ux, uy, F_hat)
                a_hat = E2 * theta_hat + Q * Nv
                Na = self._nonlinear_term(a_hat, ux, uy, F_hat)

                b_hat = E2 * theta_hat + Q * Na
                Nb = self._nonlinear_term(b_hat, ux, uy, F_hat)

                c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
                Nc = self._nonlinear_term(c_hat, ux, uy, F_hat)

                theta_hat = E * theta_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
            elif integrator == "rk4":
                t0 = time.perf_counter() if config.profile else None
                k1 = rhs(theta_hat)
                k2 = rhs(theta_hat + 0.5 * dt * k1)
                k3 = rhs(theta_hat + 0.5 * dt * k2)
                k4 = rhs(theta_hat + dt * k3)
                theta_hat = theta_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                if config.profile:
                    nl_time_total += (time.perf_counter() - t0)
            else:  # Heun / RK2
                t0 = time.perf_counter() if config.profile else None
                k1 = rhs(theta_hat)
                k2 = rhs(theta_hat + dt * k1)
                theta_hat = theta_hat + 0.5 * dt * (k1 + k2)
                if config.profile:
                    nl_time_total += (time.perf_counter() - t0)

            tnow = n * dt
            if config.save_every is not None and (n % config.save_every == 0 or n == nsteps):
                theta_snapshot = ifft2(theta_hat).real.astype(self.dtype)
                if config.save_to_disk and snapshot_dir:
                    filename = os.path.join(snapshot_dir, f"theta_{snapshot_count:05d}_t{tnow:.4f}.npy")
                    np.save(filename, theta_snapshot)
                    if snapshot_count == 0:
                        metadata = {
                            "N": self.grid.N,
                            "L": self.grid.L,
                            "dt": dt,
                            "kappa": kappa,
                            "peclet": config.peclet,
                            "mean_grad": config.mean_grad,
                            "t_end": config.t_end,
                            "save_every": config.save_every,
                        }
                        np.save(os.path.join(snapshot_dir, "metadata.npy"), metadata)
                    snapshot_count += 1
                else:
                    diagnostics.snapshots.append(theta_snapshot)
                diagnostics.times = np.append(diagnostics.times, tnow)

            if frame_step is not None and (n % frame_step == 0 or n == nsteps):
                diagnostics.frames.append(ifft2(theta_hat).real.astype(self.dtype))

            if verbose and n % max(1, nsteps // 10) == 0:
                print(f"  Step {n}/{nsteps} (t={tnow:.3f}/{t_end:.3f})")

            grad_sq_curr = self._mean_grad_sq(theta_hat)
            diagnostics.grad_sq_integral += 0.5 * (grad_sq_prev + grad_sq_curr) * dt
            grad_sq_prev = grad_sq_curr
            # Append time-resolved diagnostics
            ts_times.append(float(tnow))
            ts_grad_sq.append(float(grad_sq_curr))
            ts_eps.append(float(2.0 * kappa * grad_sq_curr))

        theta_final = ifft2(theta_hat).real.astype(self.dtype)

        # Finalize time series arrays
        diagnostics.times_ts = np.asarray(ts_times, dtype=self.dtype)
        diagnostics.grad_sq_ts = np.asarray(ts_grad_sq, dtype=self.dtype)
        diagnostics.dissipation_ts = np.asarray(ts_eps, dtype=self.dtype)

        if verbose:
            print(f"Simulation complete. Final time: {t_end:.3f}")
            if snapshot_dir and snapshot_count > 0:
                print(f"  Saved {snapshot_count} snapshots to: {snapshot_dir}/")

        # Attach profiling info if enabled
        if config.profile or config.profile_fft:
            prof = {"nonlinear_time_total": nl_time_total, "steps": nsteps}
            if config.profile_fft:
                prof["fft"] = get_fft_profile(reset=True)
            # attach as attribute to diagnostics for convenience
            diagnostics.profile = prof  # type: ignore[attr-defined]

        return theta_final, diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_time_controls(
        self,
        theta0: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        config: ScalarConfig,
    ) -> Tuple[float, int, float, float]:
        urms = np.sqrt(np.mean(ux**2 + uy**2))
        if urms <= 0:
            raise ValueError("Velocity RMS must be positive")

        if config.kappa is not None:
            kappa = config.kappa
        elif config.peclet is not None:
            kappa = urms * self.grid.L / config.peclet
        else:
            raise ValueError("Either kappa or peclet must be specified")

        if config.dt is not None:
            dt = config.dt
        else:
            denom = np.max(np.abs(ux)) + np.max(np.abs(uy)) + 1e-14
            dt_adv = config.cfl * self.grid.dx / denom
            dt_diff = config.cfl * self.grid.dx**2 / (4 * kappa + 1e-14)
            dt = float(min(dt_adv, dt_diff))

        if config.t_end is not None:
            t_end = config.t_end
        else:
            t_end = config.t_advective_mult * self.grid.L / urms

        nsteps = int(np.ceil(t_end / dt))
        dt = t_end / nsteps  # adjust to land exactly on t_end

        return dt, nsteps, float(kappa), float(t_end)

    # ------------------------------------------------------------------
    # Time-varying velocity evolution
    # ------------------------------------------------------------------
    def evolve_with_velocity_process(
        self,
        theta0: np.ndarray,
        velocity_process,
        config: ScalarConfig,
        *,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, SimulationDiagnostics]:
        """
        Evolve the scalar with a time-dependent velocity provided by a process
        object exposing ``get_velocity() -> (ux, uy)`` and ``step(dt)``.

        Supports integrators 'rk4' and 'heun'. For 'etdrk4', use a fixed
        velocity field via evolve().
        """
        theta0 = np.asarray(theta0, dtype=self.dtype)
        uxn, uyn = velocity_process.get_velocity()
        uxn = np.asarray(uxn, dtype=self.dtype)
        uyn = np.asarray(uyn, dtype=self.dtype)

        dt, nsteps, kappa, t_end = self._resolve_time_controls(theta0, uxn, uyn, config)
        diagnostics = SimulationDiagnostics(dt=dt, kappa=kappa, n_steps=nsteps)
        if config.output_frames:
            diagnostics.frames = []

        theta_hat = fft2(theta0).astype(self.cdtype)
        Llin = -kappa * self.grid.k2

        Gx, Gy = config.mean_grad

        def N_with_u(theta_hat_state: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
            theta_x = ifft2(1j * self.grid.kx * theta_hat_state).real
            theta_y = ifft2(1j * self.grid.ky * theta_hat_state).real
            adv = ux * theta_x + uy * theta_y
            N_hat = -fft2(adv).astype(self.cdtype)
            N_hat *= self.grid.dealias_mask
            if Gx != 0.0 or Gy != 0.0:
                F_hat = fft2(-(ux * Gx + uy * Gy)).astype(self.cdtype)
                N_hat += F_hat
            return N_hat

        grad_sq_prev = self._mean_grad_sq(theta_hat)
        ts_times: List[float] = [0.0]
        ts_grad_sq: List[float] = [float(grad_sq_prev)]
        ts_eps: List[float] = [float(2.0 * kappa * grad_sq_prev)]

        integrator = (config.integrator or "rk4").lower()
        if integrator not in {"rk4", "heun"}:
            raise ValueError("evolve_with_velocity_process supports 'rk4' and 'heun' only")

        snapshot_dir = None
        snapshot_count = 0
        if config.save_to_disk:
            snapshot_dir = config.save_dir or self._auto_snapshot_dir()
            os.makedirs(snapshot_dir, exist_ok=True)

        diagnostics.times = np.append(diagnostics.times, 0.0)
        if config.save_every is not None:
            theta_snapshot0 = ifft2(theta_hat).real.astype(self.dtype)
            if config.save_to_disk and snapshot_dir:
                np.save(os.path.join(snapshot_dir, "theta_00000_t0.0000.npy"), theta_snapshot0)
            else:
                diagnostics.snapshots.append(theta_snapshot0)

        frame_step = None
        if config.output_frames and config.frame_interval is not None:
            frame_step = max(1, int(round(config.frame_interval / dt)))

        for n in range(1, nsteps + 1):
            if integrator == "rk4":
                # t_n
                k1 = Llin * theta_hat + N_with_u(theta_hat, uxn, uyn)

                # t_n + dt/2
                velocity_process.step(dt / 2.0)
                uxm, uym = velocity_process.get_velocity()
                uxm = np.asarray(uxm, dtype=self.dtype)
                uym = np.asarray(uym, dtype=self.dtype)

                k2_state = theta_hat + 0.5 * dt * k1
                k2 = Llin * k2_state + N_with_u(k2_state, uxm, uym)

                k3_state = theta_hat + 0.5 * dt * k2
                k3 = Llin * k3_state + N_with_u(k3_state, uxm, uym)

                # t_n + dt
                velocity_process.step(dt / 2.0)
                uxn1, uyn1 = velocity_process.get_velocity()
                uxn1 = np.asarray(uxn1, dtype=self.dtype)
                uyn1 = np.asarray(uyn1, dtype=self.dtype)

                k4_state = theta_hat + dt * k3
                k4 = Llin * k4_state + N_with_u(k4_state, uxn1, uyn1)

                theta_hat = theta_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

                # Advance stored velocity to t_{n+1}
                uxn, uyn = uxn1, uyn1

            else:  # Heun
                k1 = Llin * theta_hat + N_with_u(theta_hat, uxn, uyn)
                velocity_process.step(dt)
                uxn1, uyn1 = velocity_process.get_velocity()
                uxn1 = np.asarray(uxn1, dtype=self.dtype)
                uyn1 = np.asarray(uyn1, dtype=self.dtype)
                k2_state = theta_hat + dt * k1
                k2 = Llin * k2_state + N_with_u(k2_state, uxn1, uyn1)
                theta_hat = theta_hat + 0.5 * dt * (k1 + k2)
                uxn, uyn = uxn1, uyn1

            tnow = n * dt

            if config.save_every is not None and (n % config.save_every == 0 or n == nsteps):
                theta_snapshot = ifft2(theta_hat).real.astype(self.dtype)
                if config.save_to_disk and snapshot_dir:
                    filename = os.path.join(snapshot_dir, f"theta_{snapshot_count:05d}_t{tnow:.4f}.npy")
                    np.save(filename, theta_snapshot)
                    if snapshot_count == 0:
                        metadata = {
                            "N": self.grid.N,
                            "L": self.grid.L,
                            "dt": dt,
                            "kappa": kappa,
                            "peclet": config.peclet,
                            "mean_grad": config.mean_grad,
                            "t_end": config.t_end,
                            "save_every": config.save_every,
                            "velocity": "time_varying_fourier_fractional",
                            "beta": getattr(getattr(velocity_process, 'beta', None), '__float__', lambda: None)() if hasattr(velocity_process, 'beta') else None,
                        }
                        np.save(os.path.join(snapshot_dir, "metadata.npy"), metadata)
                    snapshot_count += 1
                else:
                    diagnostics.snapshots.append(theta_snapshot)
                diagnostics.times = np.append(diagnostics.times, tnow)

            if frame_step is not None and (n % frame_step == 0 or n == nsteps):
                diagnostics.frames.append(ifft2(theta_hat).real.astype(self.dtype))

            if verbose and n % max(1, nsteps // 10) == 0:
                print(f"  Step {n}/{nsteps} (t={tnow:.3f}/{t_end:.3f})")

            grad_sq_curr = self._mean_grad_sq(theta_hat)
            diagnostics.grad_sq_integral += 0.5 * (grad_sq_prev + grad_sq_curr) * dt
            grad_sq_prev = grad_sq_curr
            ts_times.append(float(tnow))
            ts_grad_sq.append(float(grad_sq_curr))
            ts_eps.append(float(2.0 * kappa * grad_sq_curr))

        theta_final = ifft2(theta_hat).real.astype(self.dtype)

        diagnostics.times_ts = np.asarray(ts_times, dtype=self.dtype)
        diagnostics.grad_sq_ts = np.asarray(ts_grad_sq, dtype=self.dtype)
        diagnostics.dissipation_ts = np.asarray(ts_eps, dtype=self.dtype)

        if verbose:
            print(f"Simulation complete. Final time: {t_end:.3f}")
            if snapshot_dir and snapshot_count > 0:
                print(f"  Saved {snapshot_count} snapshots to: {snapshot_dir}/")

        return theta_final, diagnostics

    def _nonlinear_term(
        self,
        theta_hat: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        F_hat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Ensure scratch buffers
        self._ensure_work_buffers()

        # theta_x = ifft2(i kx theta_hat).real
        self._tmp_hat[...] = theta_hat
        self._tmp_hat *= self._ikx
        theta_x = ifft2(self._tmp_hat).real

        # adv_buf = ux * theta_x (in-place)
        np.multiply(ux, theta_x, out=self._adv_buf)

        # theta_y = ifft2(i ky theta_hat).real
        self._tmp_hat[...] = theta_hat
        self._tmp_hat *= self._iky
        theta_y = ifft2(self._tmp_hat).real

        # adv_buf += uy * theta_y (reusing theta_y as a temp if safe)
        np.multiply(uy, theta_y, out=theta_y)
        np.add(self._adv_buf, theta_y, out=self._adv_buf)

        N_hat = -fft2(self._adv_buf)
        N_hat *= self.grid.dealias_mask
        if F_hat is not None:
            N_hat = N_hat + F_hat
        return N_hat.astype(self.cdtype, copy=False)

    def _etdrk4_coeffs(self, Llin: np.ndarray, dt: float, M: int = 16) -> Tuple[np.ndarray, ...]:
        E = np.exp(Llin * dt)
        E2 = np.exp(Llin * dt / 2.0)

        j = np.arange(1, M + 1)
        r = np.exp(1j * np.pi * (j - 0.5) / M)
        LR = Llin[..., None] * dt + r

        Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1))
        f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=-1))
        f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=-1))
        f3 = dt * np.real(
            np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=-1)
        )
        return E, E2, Q, f1, f2, f3

    @staticmethod
    def _auto_snapshot_dir() -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join("snapshots", f"run_{timestamp}_fft{FFT_BACKEND.lower()}")

    def _mean_grad_sq(self, theta_hat: np.ndarray) -> float:
        # Accumulate in float64 to reduce overflow risk
        theta_x = ifft2(1j * self.grid.kx * theta_hat).real.astype(np.float64, copy=False)
        theta_y = ifft2(1j * self.grid.ky * theta_hat).real.astype(np.float64, copy=False)
        return float(np.mean(theta_x * theta_x + theta_y * theta_y, dtype=np.float64))


__all__ = ["ScalarConfig", "SimulationDiagnostics", "ScalarAdvectionDiffusionSolver"]
