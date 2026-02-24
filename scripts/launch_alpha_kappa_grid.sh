#!/usr/bin/env bash
# Submit one job per (alpha, kappa) pair using run_single_alpha_kappa.sbatch.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT="${SCRIPT_DIR}/run_single_alpha_kappa.sbatch"

# Edit these lists to change the sweep.
ALPHAS=(0.1666666667 0.3333333333 0.5 0.6666666667)
KAPPAS=(1e-4 3e-4 1e-3 3e-3)

# Common solver controls (edit as needed).
GRID=2048
T_END=0.25
N_SAVE=16
DT_SAVE=""
OUTPUT_ROOT="experimental_results/alpha_kappa_batch"
CFL=0.7
INTEGRATOR="rk4"
DTYPE="float32"
N_WORKERS=32
VEL_SEED=1

for alpha in "${ALPHAS[@]}"; do
  for kappa in "${KAPPAS[@]}"; do
    echo "Submitting alpha=${alpha}, kappa=${kappa}"
    sbatch \
      --export=ALL,ALPHA=${alpha},KAPPA=${kappa},GRID=${GRID},T_END=${T_END},N_SAVE=${N_SAVE},DT_SAVE=${DT_SAVE},OUTPUT_ROOT=${OUTPUT_ROOT},CFL=${CFL},INTEGRATOR=${INTEGRATOR},DTYPE=${DTYPE},N_WORKERS=${N_WORKERS},VEL_SEED=${VEL_SEED} \
      "${SBATCH_SCRIPT}"
  done
done
