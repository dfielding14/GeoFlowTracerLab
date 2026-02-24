# Repository Guidelines

## Project Structure & Module Organization
- `scalar_advection/`: core Python package (API, solver, velocity generators, spectral/statistical diagnostics, FFT wrapper, plotting helpers).
- `examples/`: Jupyter workflow entry points (`01_velocity_generation.ipynb`, `02b_mean_gradient_forcing.ipynb`, etc.).
- `experiments/`: executable scripts for production runs and diagnostics (`run_single_alpha_kappa.py`, `analyze_alpha_kappa_run.py`, `run_wavelet_scalar_experiment.py`).
- `scripts/`: campaign orchestration and HPC utilities (`generate_alpha_kappa_campaign.py`, `run_single_alpha_kappa.sbatch`, `benchmark_threads.py`).
- `docs/`: design notes and HOWTO references (`TEMPORAL_DIAGNOSTICS.md`, `PROFILING.md`, `STRUCTURE_FUNCTIONS.md`).
- `campaigns/`, `experimental_results/`, `velocity_structure_functions_*`: campaign outputs and generated artifacts (ignored by git).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create environment.
- `pip install -r requirements.txt`: install Python dependencies (`numpy`, `scipy`, `matplotlib`, `pyfftw`, `numba`, etc.).
- `python - <<'PY'\nfrom scalar_advection import ScalarAdvectionAPI\nprint(ScalarAdvectionAPI().generate_velocity().__class__.__name__)\nPY`: quick import smoke check.
- `python experiments/run_single_alpha_kappa.py --alpha 1/3 --kappa 1e-3 --grid 256 --t-end 0.1 --n-save 8 --output-root experimental_results/alpha_kappa_runs`: run one short local simulation.
- `python scripts/generate_alpha_kappa_campaign.py` then `sbatch scripts/run_single_alpha_kappa.sbatch`: generate and launch parameter sweeps.

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation, snake_case for functions/variables/modules, PascalCase for classes/dataclasses.
- Follow existing style: type hints, NumPy-first APIs, explicit CLI flags, concise docstrings.
- Keep file/function names descriptive (`run_*`, `analyze_*`, `*_sf`, `*_functions`, `*frame*`).
- No project-wide formatter/linter config is present; keep changes consistent with current local style.

## Testing Guidelines
- No dedicated `tests/` directory or test runner config (`pytest`, `tox`, `unittest`, or `ruff` config) is present.
- Use deterministic smoke checks for new logic:
  - import-level check,
  - one small `ScalarAdvectionAPI` scalar evolution (`N=64` or `N=128`),
  - one lightweight experiment script run with tiny `n-save` and local output directory.
- Store/attach generated outputs only in ignored folders.

## Commit & Pull Request Guidelines
- Git history uses mostly Conventional-Commit-like prefixes (`feat:`, `perf:`, `refactor:`, `docs:`, `chore:`, `viz:`, `exp:`, `stability:`). Continue that convention.
- PRs should include:
  - summary of changed equations/parameters and scientific intent,
  - exact command(s) run for validation,
  - sample output locations (`campaigns/`, `experimental_results/`, plots),
  - any seeds/config toggles changed.
- Prefer small, reviewable commits with complete metadata in scripts when behavior changes.

## Security & Configuration Tips
- Keep `experimental_results/` and large binaries out of commits.
- For reproducibility, always set random seeds (`--velocity-seed`, etc.) for physics-altering scripts.
- Use explicit thread settings (`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`) before large runs to control resource usage.
