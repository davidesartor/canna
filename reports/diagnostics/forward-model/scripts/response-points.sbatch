#!/bin/bash
# Each band's response_points against a 16384-point jaxgb reference (reports/diagnostics/forward-model/scripts/response_points.py).
# The record goes to reports/diagnostics/forward-model/data/response-points.{json,npz}. Submit from the repo root;
# a first argument `smoke` cuts the counts:
#     sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/diagnostics/forward-model/scripts/response-points.sbatch

#SBATCH --job-name=canna-response-points
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
DIR=reports/diagnostics/forward-model
mkdir -p .slurm-logs "$DIR/data"
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# a stray LISA_* variable would change config.load's band
for v in ${!LISA_@}; do unset "$v"; done
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "commit    : $(git rev-parse HEAD)"
uv run --frozen python --version
uv run --frozen python "$DIR/scripts/response_points.py" "$DIR/data" "${1:-}"
echo "done      : $(date)"
