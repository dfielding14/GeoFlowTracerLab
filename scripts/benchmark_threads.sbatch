#!/bin/bash
#SBATCH --job-name=threadBench
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_threads_%j.out
#SBATCH --error=benchmark_threads_%j.err

set -euo pipefail

module purge
module load modules/2.4-20250724
module load gcc/13.3.0
module load python/3.11.11
module load fftw/3.3.10

VENV_PATH=${VENV_PATH:-$HOME/.venvs/geoflow}
source "${VENV_PATH}/bin/activate"

cd "$SLURM_SUBMIT_DIR"

python -u scripts/benchmark_threads.py
