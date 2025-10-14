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
    method: str = "spectral"
    integrator: str = "etdrk4"


@dataclass
class SimulationDiagnostics:
    """Data collected during scalar evolution."""

    snapshots: List[np.ndarray] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: float = 0.0
    kappa: float = 0.0
    n_steps: int = 0
    frames: Optional[List[np.ndarray]] = None


class ScalarAdvectionDiffusionSolver:
    """
    Pseudo-spectral ETDRK4 solver for passive scalar advection-diffusion.
    """

    def __init__(self, grid: SpectralGrid):
        self.grid = grid
        self.dtype = grid.dtype
        self.cdtype = grid.cdtype

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
        theta_hat = fft2(theta)
        theta_x = ifft2(1j * self.grid.kx * theta_hat).real
        theta_y = ifft2(1j * self.grid.ky * theta_hat).real
        return 2.0 * kappa * float(np.mean(theta_x**2 + theta_y**2))

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
        self.ux = np.asarray(ux, dtype=self.dtype)
        self.uy = np.asarray(uy, dtype=self.dtype)
        self.config = config
        self.verbose = verbose

        self._resolve_time_controls(theta0, ux, uy, config)
        self.diagnostics = SimulationDiagnostics(dt=self.dt,
                                            kappa=self.kappa,
                                            n_steps=self.nsteps)
        if config.output_frames:
            self.diagnostics.frames = []
        
        if config.method.lower() == "spectral":
            return self.spectral_solver(theta0)
        elif config.method.lower() == "finite_volume":
            raise NotImplementedError("Finite volume method not yet implemented")
        else:
            raise ValueError("Method must be 'spectral' or 'finite_volume'")
        

    def spectral_solver(
        self,
        theta0:np.ndarray
    ) -> Tuple[np.ndarray, SimulationDiagnostics]:

        theta_hat = fft2(theta0).astype(self.cdtype)
        Llin = -self.kappa * self.grid.k2

        Gx, Gy = self.config.mean_grad
        if Gx != 0.0 or Gy != 0.0:
            F = -(self.ux * Gx + self.uy * Gy)
            F_hat = fft2(F).astype(self.cdtype)
        else:
            F_hat = None

        integrator = (self.config.integrator or "etdrk4").lower()
        if integrator not in {"etdrk4", "rk4", "heun"}:
            raise ValueError("integrator must be 'etdrk4', 'rk4', or 'heun'")

        if integrator == "etdrk4":
            E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(Llin, self.dt)

        def rhs(theta_hat_state: np.ndarray) -> np.ndarray:
            return Llin * theta_hat_state + self._nonlinear_term(theta_hat_state, self.ux, self.uy, F_hat)

        snapshot_dir = None
        snapshot_count = 0
        if self.config.save_to_disk:
            snapshot_dir = self.config.save_dir or self._auto_snapshot_dir()
            os.makedirs(snapshot_dir, exist_ok=True)

        self.diagnostics.times = np.append(self.diagnostics.times, 0.0)
        if self.config.save_every is not None:
            if self.config.save_to_disk and snapshot_dir:
                np.save(os.path.join(snapshot_dir, "theta_00000_t0.0000.npy"), theta0)
            else:
                self.diagnostics.snapshots.append(theta0.copy())

        frame_step = None
        if self.config.output_frames and self.config.frame_interval is not None:
            frame_step = max(1, int(round(self.config.frame_interval / self.dt)))

        for n in range(1, self.nsteps + 1):
            if integrator == "etdrk4":
                Nv = self._nonlinear_term(theta_hat, self.ux, self.uy, F_hat)
                a_hat = E2 * theta_hat + Q * Nv
                Na = self._nonlinear_term(a_hat, self.ux, self.uy, F_hat)

                b_hat = E2 * theta_hat + Q * Na
                Nb = self._nonlinear_term(b_hat, self.ux, self.uy, F_hat)

                c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
                Nc = self._nonlinear_term(c_hat, self.ux, self.uy, F_hat)

                theta_hat = E * theta_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
            elif integrator == "rk4":
                k1 = rhs(theta_hat)
                k2 = rhs(theta_hat + 0.5 * self.dt * k1)
                k3 = rhs(theta_hat + 0.5 * self.dt * k2)
                k4 = rhs(theta_hat + self.dt * k3)
                theta_hat = theta_hat + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:  # Heun / RK2
                k1 = rhs(theta_hat)
                k2 = rhs(theta_hat + self.dt * k1)
                theta_hat = theta_hat + 0.5 * self.dt * (k1 + k2)

            tnow = n * self.dt
            if self.config.save_every is not None and (n % self.config.save_every == 0 or n == self.nsteps):
                theta_snapshot = ifft2(theta_hat).real.astype(self.dtype)
                if self.config.save_to_disk and snapshot_dir:
                    filename = os.path.join(snapshot_dir, f"theta_{snapshot_count:05d}_t{tnow:.4f}.npy")
                    np.save(filename, theta_snapshot)
                    if snapshot_count == 0:
                        metadata = {
                            "N": self.grid.N,
                            "L": self.grid.L,
                            "dt": self.dt,
                            "kappa": self.kappa,
                            "peclet": self.config.peclet,
                            "mean_grad": self.config.mean_grad,
                            "t_end": self.config.t_end,
                            "save_every": self.config.save_every,
                        }
                        np.save(os.path.join(snapshot_dir, "metadata.npy"), metadata)
                    snapshot_count += 1
                else:
                    self.diagnostics.snapshots.append(theta_snapshot)
                self.diagnostics.times = np.append(self.diagnostics.times, tnow)

            if frame_step is not None and (n % frame_step == 0 or n == self.nsteps):
                self.diagnostics.frames.append(ifft2(theta_hat).real.astype(self.dtype))

            if self.verbose and n % max(1, self.nsteps // 10) == 0:
                print(f"  Step {n}/{self.nsteps} (t={tnow:.3f}/{self.t_end:.3f})")

        theta_final = ifft2(theta_hat).real.astype(self.dtype)

        if self.verbose:
            print(f"Simulation complete. Final time: {self.t_end:.3f}")
            if snapshot_dir and snapshot_count > 0:
                print(f"  Saved {snapshot_count} snapshots to: {snapshot_dir}/")

        return theta_final, self.diagnostics

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
        self.kappa = kappa

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

        self.t_end = t_end
        self.nsteps = int(np.ceil(t_end / dt))
        self.dt = t_end / self.nsteps  # adjust to land exactly on t_end

    def _nonlinear_term(
        self,
        theta_hat: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        F_hat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        theta_x = ifft2(1j * self.grid.kx * theta_hat).real
        theta_y = ifft2(1j * self.grid.ky * theta_hat).real
        adv = ux * theta_x + uy * theta_y
        N_hat = -fft2(adv).astype(self.cdtype)
        N_hat *= self.grid.dealias_mask
        if F_hat is not None:
            N_hat += F_hat
        return N_hat

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


__all__ = ["ScalarConfig", "SimulationDiagnostics", "ScalarAdvectionDiffusionSolver"]
