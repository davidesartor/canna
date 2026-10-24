#!/bin/bash
# Strub et al. 2022's faint source on CPU (strub_source.py beside this file), one array task per chain:
# task 0 the noiseless injection, task 1 noise seed 0 (Strub's prior, then the Fisher box), tasks
# 2-49 the 48 noise realisations. Chains go to outputs/strub-source/task-<k>.npz; strub_summary.sbatch
# reads them. Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_CPU --array=1 --time=00:30:00 reports/diagnostics/strub-source/scripts/strub_chains.sbatch --burn 400 --keep 400 --chunk 200
#     sbatch $CANNA_SBATCH_CPU reports/diagnostics/strub-source/scripts/strub_chains.sbatch
# Further flags go to `strub_source.py chain` (--nwalker, --temps, --burn, --keep, --chunk, --thin).

#SBATCH --job-name=canna-strub-source
#SBATCH --output=.slurm-logs/%x-%A_%a.out
#SBATCH --error=.slurm-logs/%x-%A_%a.err
#SBATCH --array=0-49%20
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
mkdir -p .slurm-logs
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
for v in ${!LISA_@}; do unset "$v"; done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_ARRAY_JOB_ID:-none}_${TASK}"
echo "commit    : $(git rev-parse HEAD)"
echo "cpus      : $(nproc)"
uv run --frozen python --version
uv run --frozen python reports/diagnostics/strub-source/scripts/strub_source.py chain --task "$TASK" "$@"
echo "done      : $(date)"
