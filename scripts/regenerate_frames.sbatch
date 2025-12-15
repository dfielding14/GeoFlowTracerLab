#!/bin/bash
#SBATCH --job-name=regen_frames
#SBATCH --output=logs/regen_frames_%j.out
#SBATCH --error=logs/regen_frames_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cca

# Regenerate theta_velocity_frames with correct cmasher colormaps
# Usage: sbatch --export=RUN_DIR=/path/to/run scripts/regenerate_frames.sbatch

set -euo pipefail

echo "=========================================="
echo "REGENERATE FRAMES JOB"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "Started at: $(date)"
echo "=========================================="

# Check required environment variable
if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR environment variable not set"
    echo "Usage: sbatch --export=RUN_DIR=/path/to/run scripts/regenerate_frames.sbatch"
    exit 1
fi
echo "RUN_DIR: ${RUN_DIR}"

# Project paths
PROJECT_DIR="/mnt/ceph/users/dfielding/GeoFlowTracerLab"
# Allow MOVIE_OUTPUT_DIR to be passed via environment, with default fallback
MOVIE_OUTPUT_DIR="${MOVIE_OUTPUT_DIR:-${PROJECT_DIR}/campaigns/alpha_kappa_2048/movies}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "MOVIE_OUTPUT_DIR: ${MOVIE_OUTPUT_DIR}"

# Setup environment
echo ""
echo "Setting up environment..."
module purge
module load modules/2.4-20250724
module load gcc/13.3.0 python/3.11.11 fftw/3.3.10 ffmpeg
echo "Modules loaded successfully"

echo "Activating virtualenv..."
VENV_PATH="${HOME}/.venvs/geoflow"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify required packages
echo ""
echo "Verifying required packages..."
python -c "import cmasher; print(f'  cmasher: {cmasher.__version__}')" || { echo "ERROR: cmasher not installed"; exit 1; }
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import matplotlib; print(f'  matplotlib: {matplotlib.__version__}')"
echo "All required packages available"

# Verify ffmpeg
echo ""
echo "Verifying ffmpeg..."
ffmpeg -version | head -1
echo "ffmpeg available"

# Create directories
echo ""
echo "Creating output directories..."
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${MOVIE_OUTPUT_DIR}"
echo "Directories ready"

# Verify run directory exists and has required files
echo ""
echo "Verifying run directory..."
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: Run directory does not exist: ${RUN_DIR}"
    exit 1
fi
echo "  run_config.json:        $([ -f "${RUN_DIR}/run_config.json" ] && echo "OK" || echo "MISSING")"
echo "  fields/velocity_fields: $([ -f "${RUN_DIR}/fields/velocity_fields.npz" ] && echo "OK" || echo "MISSING")"
echo "  fields/snapshots:       $([ -d "${RUN_DIR}/fields/snapshots" ] && echo "OK" || echo "MISSING")"

# Count snapshots
SNAPSHOT_COUNT=$(ls -1 "${RUN_DIR}/fields/snapshots/"*.npz 2>/dev/null | wc -l)
echo "  Snapshot count:         ${SNAPSHOT_COUNT}"

echo ""
echo "=========================================="
echo "RUNNING REGENERATION SCRIPT"
echo "=========================================="
echo ""

python "${PROJECT_DIR}/scripts/regenerate_frames_and_movie.py" \
    --run-dir "${RUN_DIR}" \
    --movie-output-dir "${MOVIE_OUTPUT_DIR}" \
    --fps 16 \
    --delete-old

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "JOB COMPLETE"
echo "=========================================="
echo "Exit code:   ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
