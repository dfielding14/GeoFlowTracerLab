"""
Scalar advection toolkit providing modular velocity generation, solver, and statistics.
"""

from .api import ScalarAdvectionAPI
from .velocity import VelocityConfig, VelocityFieldGenerator, generate_velocity_field, generate_divfree_field
from .solver import ScalarAdvectionDiffusionSolver, ScalarConfig, SimulationDiagnostics
from .spectra import kinetic_energy_spectrum, plot_energy_spectrum, scalar_power_spectrum, plot_scalar_spectrum
from .structure import (
    pair_increment_pdf,
    plot_pair_increment_pdf,
    plot_structure_functions,
    structure_functions,
    structure_functions_from_pair_pdf,
)
from .fft import FFT_BACKEND, set_fftw_threads, warm_fft_cache
from .fractal import box_counting

__all__ = [
    "ScalarAdvectionAPI",
    "VelocityConfig",
    "VelocityFieldGenerator",
    "generate_velocity_field",
    "generate_divfree_field",
    "ScalarAdvectionDiffusionSolver",
    "ScalarConfig",
    "SimulationDiagnostics",
    "scalar_power_spectrum",
    "plot_scalar_spectrum",
    "kinetic_energy_spectrum",
    "plot_energy_spectrum",
    "structure_functions",
    "plot_structure_functions",
    "pair_increment_pdf",
    "plot_pair_increment_pdf",
    "structure_functions_from_pair_pdf",
    "FFT_BACKEND",
    "set_fftw_threads",
    "warm_fft_cache",
    "box_counting",
]
