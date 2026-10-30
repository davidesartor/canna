#!/bin/bash
# Data-dependence probe of a run's checkpoint (data_dependence.py beside this file), then the page
# (build_page.py). The record goes to reports/architecture/data-dependence/data/<tag>.json (tag
# defaults to <run>); it runs on the commit the run trained at (slurm/run-src.sh), with the run's
# own eval settings (stray LISA_* variables are cleared; a third argument `training` probes the
# training weights instead of the average). From the repo root, after `source slurm/<cluster>.env`;
# an XS or S run fits a CPU node, a larger one wants a GPU:
#     sbatch $CANNA_SBATCH_CPU reports/architecture/data-dependence/scripts/data_dependence.sbatch <run> [tag] [training]
#     sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/architecture/data-dependence/scripts/data_dependence.sbatch <run> [tag] [training]

#SBATCH --job-name=canna-lisa-data-dependence
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/architecture/data-dependence
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
[[ -z "${3:-}" ]] || export LISA_WEIGHTS="$3"
source slurm/run-src.sh
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source slurm/gpu.sh
else
    export JAX_PLATFORMS=cpu
fi
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
# --extra cuda on either: a sync without it would strip the shared .venv's cuda packages
uv run --extra cuda --frozen python "$HERE/scripts/data_dependence.py" "${2:-$LISA_RUN}"
uv run --frozen --no-sync python "$HERE/scripts/build_page.py"
