#!/bin/bash
#SBATCH --job-name=alphaKappa
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=168:00:00
#SBATCH --output=alpha_kappa_%j.out
#SBATCH --error=alpha_kappa_%j.err

set -euo pipefail

# ----------------------------------------------------------------------
# Rusty environment setup (modules 2.4 + gcc 13 + python 3.11)
# ----------------------------------------------------------------------
module purge
module load modules/2.4-20250724
module load gcc/13.3.0
module load python/3.11.11
module load fftw/3.3.10

VENV_PATH=${VENV_PATH:-$HOME/.venvs/geoflow}
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
PYTHON_BIN="${VENV_PATH}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${N_WORKERS:-128}}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-$OMP_NUM_THREADS}
TMP_BASE=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$TMP_BASE/mpl}
mkdir -p "$MPLCONFIGDIR"


: "${ALPHA:?Need ALPHA}"
: "${KAPPA:?Need KAPPA}"
GRID=${GRID:-4096}
T_END=${T_END:-20}
N_SAVE=${N_SAVE:-200}
DT_SAVE=${DT_SAVE:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-experimental_results/alpha_kappa_runs}
CFL=${CFL:-0.6}
INTEGRATOR=${INTEGRATOR:-etdrk4}
DTYPE=${DTYPE:-float64}
N_WORKERS=${N_WORKERS:-128}
VEL_SEED=${VEL_SEED:-1}
LAM_MIN=${LAM_MIN:-2}
LAM_MAX=${LAM_MAX:-$GRID}
TAG=${TAG:-}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-500}

DT_SAVE_ARGS=()
if [ -n "${DT_SAVE}" ]; then
  DT_SAVE_ARGS=(--dt-save "${DT_SAVE}")
fi
TAG_ARGS=()
if [ -n "${TAG}" ]; then
  TAG_ARGS=(--tag "${TAG}")
fi

cd "$SLURM_SUBMIT_DIR"

srun "$PYTHON_BIN" -u experiments/run_single_alpha_kappa.py \
    --alpha "${ALPHA}" \
    --kappa "${KAPPA}" \
    --grid "${GRID}" \
    --t-end "${T_END}" \
    --n-save "${N_SAVE}" \
    "${DT_SAVE_ARGS[@]}" \
    --cfl "${CFL}" \
    --integrator "${INTEGRATOR}" \
    --dtype "${DTYPE}" \
    --n-workers "${N_WORKERS}" \
    --velocity-seed "${VEL_SEED}" \
    --velocity-urms 1.0 \
    --lam-min "${LAM_MIN}" \
    --lam-max "${LAM_MAX}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --quiet \
    "${TAG_ARGS[@]}" \
    --output-root "${OUTPUT_ROOT}"
