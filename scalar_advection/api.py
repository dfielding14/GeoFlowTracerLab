"""
High-level, user-friendly API for the scalar advection toolkit.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .grid import SpectralGrid
from .velocity import VelocityConfig, VelocityFieldGenerator
from .solver import ScalarAdvectionDiffusionSolver, ScalarConfig, SimulationDiagnostics
from .spectra import (
    kinetic_energy_spectrum,
    plot_energy_spectrum,
    scalar_power_spectrum,
    plot_scalar_spectrum,
)
from .structure import (
    pair_increment_pdf,
    plot_pair_increment_pdf,
    plot_structure_functions,
    structure_functions,
    structure_functions_from_pair_pdf,
)
from .fractal import box_counting, BoxCountingResult
from .fft import set_fftw_threads, warm_fft_cache as _warm_fft_cache


class ScalarAdvectionAPI:
    """
    Facade that bundles velocity generation, scalar evolution, and diagnostics.
    """

    def __init__(self, N: int = 256, L: float = 1.0, dtype=np.float64, warm_cache: bool = False):
        self.grid = SpectralGrid(N=N, L=L, dtype=dtype)
        self.velocity_generator = VelocityFieldGenerator(self.grid)
        self.solver = ScalarAdvectionDiffusionSolver(self.grid)
        if warm_cache:
            self.warm_fft_cache()

    # ------------------------------------------------------------------
    # Velocity generation
    # ------------------------------------------------------------------
    def generate_velocity(self, config: VelocityConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
        config = config or VelocityConfig()
        ux, uy = self.velocity_generator.generate(config)
        if ux.dtype != self.grid.dtype:
            ux = ux.astype(self.grid.dtype, copy=False)
            uy = uy.astype(self.grid.dtype, copy=False)
        return ux, uy

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def circle_initial_condition(
        self,
        radius: float = 0.25,
        center: Tuple[float, float] = (0.0, 0.0),
        val_in: float = 1.0,
        val_out: float = 0.0,
    ) -> np.ndarray:
        return self.solver.create_circle_initial_condition(radius, center, val_in, val_out)

    def random_initial_condition(self, seed: int | None = None) -> np.ndarray:
        return self.solver.create_random_initial_condition(seed=seed)

    # ------------------------------------------------------------------
    # Scalar evolution
    # ------------------------------------------------------------------
    def evolve_scalar(
        self,
        theta0: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        config: ScalarConfig | None = None,
        *,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, SimulationDiagnostics]:
        config = config or ScalarConfig()
        return self.solver.evolve(theta0, ux, uy, config, verbose=verbose)

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------
    def scalar_spectrum(
        self,
        field: np.ndarray,
        *,
        subtract_mean: bool = True,
        subtract_mean_gradient: bool = False,
        mean_grad: Tuple[float, float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return scalar_power_spectrum(
            field,
            self.grid,
            subtract_mean=subtract_mean,
            subtract_mean_gradient=subtract_mean_gradient,
            mean_grad=mean_grad,
        )

    def velocity_spectrum(self, ux: np.ndarray, uy: np.ndarray, *, n_bins: int = 48) -> Dict[str, np.ndarray]:
        return kinetic_energy_spectrum(ux, uy, n_bins=n_bins)

    def velocity_structure_functions(self, ux: np.ndarray, uy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        return structure_functions((ux, uy), **kwargs)

    def scalar_structure_functions(self, theta: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        return structure_functions(theta, **kwargs)

    def velocity_pair_pdf(self, ux: np.ndarray, uy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        return pair_increment_pdf((ux, uy), **kwargs)

    def scalar_dissipation(self, theta: np.ndarray, kappa: float) -> float:
        return self.solver.compute_scalar_dissipation(theta, kappa)
    def box_counting(self, mask: np.ndarray, **kwargs) -> BoxCountingResult:
        return box_counting(mask, **kwargs)

    # ------------------------------------------------------------------
    # Visualization shortcuts (pass-through)
    # ------------------------------------------------------------------
    @staticmethod
    def plot_structure_functions(result: Dict[str, np.ndarray], **kwargs) -> None:
        plot_structure_functions(result, **kwargs)

    @staticmethod
    def plot_pair_increment_pdf(result: Dict[str, np.ndarray], **kwargs) -> None:
        plot_pair_increment_pdf(result, **kwargs)

    @staticmethod
    def plot_energy_spectrum(spec: Dict[str, np.ndarray], **kwargs) -> None:
        plot_energy_spectrum(spec, **kwargs)

    @staticmethod
    def plot_scalar_spectrum(k: np.ndarray, E: np.ndarray, **kwargs) -> None:
        plot_scalar_spectrum(k, E, **kwargs)

    # ------------------------------------------------------------------
    # FFT utilities
    # ------------------------------------------------------------------
    @staticmethod
    def set_fft_threads(n: int) -> None:
        set_fftw_threads(n)

    def warm_fft_cache(self) -> None:
        _warm_fft_cache((self.grid.N, self.grid.N), dtype=self.grid.dtype)

    # ------------------------------------------------------------------
    # Convenience driver
    # ------------------------------------------------------------------
    def quick_simulation(
        self,
        velocity_config: VelocityConfig | None = None,
        scalar_config: ScalarConfig | None = None,
        *,
        mean_gradient_initial: bool = False,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, SimulationDiagnostics]:
        """
        Mirror of the original quick-simulation helper.
        """
        velocity_config = velocity_config or VelocityConfig()
        scalar_config = scalar_config or ScalarConfig()

        ux, uy = self.generate_velocity(velocity_config)

        if mean_gradient_initial and scalar_config.mean_grad != (0.0, 0.0):
            theta0 = np.zeros((self.grid.N, self.grid.N), dtype=self.grid.dtype)
        else:
            theta0 = self.circle_initial_condition()

        theta_final, diagnostics = self.evolve_scalar(theta0, ux, uy, scalar_config, verbose=verbose)
        return theta0, theta_final, diagnostics


__all__ = [
    "ScalarAdvectionAPI",
    "VelocityConfig",
    "ScalarConfig",
    "SimulationDiagnostics",
    "kinetic_energy_spectrum",
    "scalar_power_spectrum",
    "plot_scalar_spectrum",
    "structure_functions",
    "structure_functions_from_pair_pdf",
    "pair_increment_pdf",
    "box_counting",
]
