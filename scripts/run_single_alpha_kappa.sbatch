#!/bin/bash
#SBATCH --job-name=alphaKappa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --output=alpha_kappa_%j.out
#SBATCH --error=alpha_kappa_%j.err

# Optional: module load commands go here, e.g.
# module load python/3.11

set -euo pipefail

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

srun python -u experiments/run_single_alpha_kappa.py \
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
