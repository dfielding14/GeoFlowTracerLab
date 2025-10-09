<div align="center">

# Scalar Advection Toolkit

**Author:** Drummond B. Fielding (DBF)

**A pseudo-spectral laboratory for tracer transport, spectra, structure functions, and fractal diagnostics.**

</div>

## 1. Motivation

This repository grew out of the _Tracers, Regularity, and Computation_ working group (Simons Initiative on the Geometry of Flows, a.k.a. AstroGMT).  Our goal is to simulate passive-scalar advection–diffusion in model turbulent flows while extracting diagnostics relevant to geometric measure theory: spectra, structure functions, and box-counting dimensions of filamentary structures.

## 2. Governing equations

We evolve the dimensionless passive-scalar perturbation `θ(x, y, t)` on a 2-D periodic square domain `[-L/2, L/2)^2`.  With mean gradient forcing:

$$
\frac{\partial \theta}{\partial t} + \mathbf{u} \cdot \nabla \theta
    = \kappa \nabla^2 \theta - \mathbf{u} \cdot \mathbf{G},
$$

where $\mathbf{G} = (G_x, G_y)$ is the imposed large-scale gradient.  The full scalar is $\Theta = \mathbf{G}·\mathbf{x} + \theta$, but we only ever use $\theta$.  Velocity fields are synthetic, either Fourier-based (random phases shaped to a prescribed spectrum) or wavelet-based (divergence-free streamfunction built via scale coefficients).

## 3. Numerical method

| Component                    | Description                                                         |
|-----------------------------|---------------------------------------------------------------------|
| Spatial discretisation      | Pseudo-spectral FFT grid with 2/3 dealiasing                        |
| Time stepping               | Default ETDRK4 (exponential time differencing RK4). Alternatives: classical RK4 or Heun (RK2) for quick previews (`ScalarConfig.integrator`). |
| FFT backend                 | NumPy FFT or pyFFTW (auto-detected). Thread count adjustable via `ScalarAdvectionAPI.set_fft_threads`. |
| Mean-gradient forcing term  | Added in Fourier space as a source term `-u·G`.                      |
| Diagnostics                 | Spectra (`scalar_power_spectrum`, `kinetic_energy_spectrum`), structure functions, pair-increment PDFs, box-counting dimension. |

### 3.1 ETDRK4 overview

We decompose the equation into linear `Lθ = κ∇²θ` and nonlinear `N(θ) = -u·∇θ - u·G`.  ETDRK4 integrates

$$
\theta^{n+1} = e^{L \Delta t} \theta^n +
   \varphi_1(L \Delta t) N(\theta^n) +
   \varphi_2(L \Delta t) [N(a) + N(b)] +
   \varphi_3(L \Delta t) N(c),
$$

where `a, b, c` are RK-like stages.  The `φ` functions are approximated via contour integrals (Kassam & Trefethen 2005).  In Fourier space the linear operator is diagonal, so exponentials are cheap.

## 4. Code organisation

```
scalar_advection/
  api.py          High-level façade (ScalarAdvectionAPI)
  velocity.py     Velocity generators (Fourier + wavelet)
  solver.py       Scalar solver (ETDRK4, RK4, Heun)
  spectra.py      Spectral utilities and plotting helpers
  structure.py    Structure functions, pair-increment PDFs
  fractal.py      Box-counting dimension
  fft.py          FFT backend selection (NumPy / pyFFTW)
  binning.py      Shell/bin utilities
  fitting.py      Log-log power-law fitting utilities
examples/
  01_velocity_generation.ipynb
  02b_mean_gradient_forcing.ipynb
  03_flow_statistics.ipynb
scripts/
  profile_wavelet_generation.py  (Command-line timing probe)
README.md
```

## 5. Core API

```python
from scalar_advection import (
    ScalarAdvectionAPI, VelocityConfig, ScalarConfig,
    scalar_power_spectrum, kinetic_energy_spectrum,
    structure_functions, structure_functions_from_pair_pdf,
    pair_increment_pdf, plot_scalar_spectrum, plot_structure_functions,
    box_counting, set_fftw_threads,
)
```

### 5.1 ScalarAdvectionAPI

```python
api = ScalarAdvectionAPI(
    N=512,
    L=1.0,
    dtype=np.float64,
    warm_cache=True,  # prime FFTW plan cache
)
```

- `generate_velocity(VelocityConfig)` → `(ux, uy)`
- `circle_initial_condition(...)`, `random_initial_condition(...)`
- `evolve_scalar(theta0, ux, uy, ScalarConfig)` → `(theta_final, diagnostics)`
- `scalar_spectrum(theta, subtract_mean=..., subtract_mean_gradient=...)`
- `velocity_spectrum(ux, uy)`
- `scalar_structure_functions(theta, orders=...)`
- `velocity_structure_functions((ux, uy), ...)`
- `velocity_pair_pdf((ux, uy), ...)`
- `scalar_dissipation(theta, kappa)`
- `box_counting(mask, ...)`
- `set_fft_threads(n)`, `warm_fft_cache()`

### 5.2 Configuration dataclasses

```python
VelocityConfig(
    beta=5/3, urms=1.0, seed=None, f_sol=1.0,
    kmin=None, kmax=None, taper_width=0.0,
    precision='auto',  # 'float32', 'float64', or auto-match solver
)

ScalarConfig(
    peclet=2000, t_end=1.0, dt=None,
    mean_grad=(1.0, 0.0),
    save_every=None, output_frames=False,
    integrator='etdrk4',  # 'rk4', 'heun'
)
```

### 5.3 Diagnostics & plotting

```python
k, E = api.scalar_spectrum(theta_final)
ScalarAdvectionAPI.plot_scalar_spectrum(
    k, E, annotate_fit=True,
    fit_min_k=5, fit_max_k=k.max()/2,
)

orders = (1, 2, 3, 4)
sf = api.scalar_structure_functions(theta_final, orders=orders)
plot_structure_functions(sf, fit_min_r=4, fit_max_r=64)

pdf = api.velocity_pair_pdf((ux, uy), kind='mag')
plot_pair_increment_pdf(pdf, fit_min_points=..., ...)

mask = (theta_final > 0)
box = api.box_counting(mask, fit_min_size=2, fit_max_size=64)
print('Fractal dimension ≈', box.slope)
```

`plot_*` helpers expose keyword arguments `fit_min_*` / `fit_max_*` to clamp the log-log fit range.

## 6. Workflow (see notebooks)

1. **Velocity generation** – compare Fourier vs wavelet fields (01).
2. **Mean-gradient forced run** – 512² ETDRK4 run, spectra, structure functions, dissipation, box-counting (02b).
3. **Flow statistics** – structure functions, pair PDFs, spectra for both velocity and scalar fields (03).

Each notebook begins with:

```python
ScalarAdvectionAPI.set_fft_threads(8)
```

so pyFFTW (if available) uses multiple threads; `warm_cache=True` primes plans.

## 7. Performance tips

| Tip                                 | Reason                                                         |
|-------------------------------------|----------------------------------------------------------------|
| Install `pyfftw`                    | 2–3× faster FFTs; set `FFTW_THREADS`.                         |
| Warm cache once per resolution      | Avoid pyFFTW plan-build overhead.                             |
| Use `dtype=np.float32` for previews | Halves memory and FFT cost.                                   |
| Reduce saved snapshots              | Decrease `save_every`.                                        |
| Use `integrator='rk4'` for previews | Cheaper time stepping if ETDRK4 stiffness handling not needed.|

## 8. Fractal diagnostics

`scalar_advection/fractal.py` performs simple box-counting on binary masks.  For periodic fields, tile the mask (`np.tile(mask, (2, 2))`) before calling the routine to avoid edge breaks.  The notebook demonstrates using the `θ=0` contour to estimate the interface dimension in a mean-gradient forced run.

## 9. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

To verify the FFT backend:

```bash
python - <<'PY'
from scalar_advection import FFT_BACKEND
print('FFT backend:', FFT_BACKEND)
PY
```

If `pyfftw` is absent, the toolkit falls back to NumPy FFTs automatically.

---

_Maintained by Drummond Fielding’s Tracers, Regularity, and Computation subgroup (AstroGMT / Simons Initiative on the Geometry of Flows)._
