#!/bin/bash
# The joint MCMC of the paper's four narrowband eval windows (joint_mcmc.py beside this file), on
# CPU nodes: one array task per (window, seed), windows 9 5 2 7 x seeds 0 1. The chains are cached
# under outputs/mcmc/joint-<problem id>/, keyed on the problem (sky prior included) and the eval
# seed. Submit from the repo root after `source slurm/<cluster>.env` (unity or trex);
# time the likelihood first (one task, prints and stops):
#     sbatch $CANNA_SBATCH_CPU --array=0 --time=00:30:00 reports/mcmc/scripts/joint_mcmc.sbatch narrow-B --bench
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/joint_mcmc.sbatch narrow-B
# Further flags go to joint_mcmc.py (--burn, --keep, --thin, --nwalker, --force).

#SBATCH --job-name=canna-joint-mcmc
#SBATCH --output=.slurm-logs/%x-%A_%a.out
#SBATCH --error=.slurm-logs/%x-%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
mkdir -p .slurm-logs
export LISA_CONFIG="$1"; CONFIG="$1"; shift
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
WINDOWS=(9 5 2 7)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
WINDOW="${WINDOWS[$((TASK / 2))]}"
SEED=$((TASK % 2))

echo "host      : $(hostname)"
echo "job       : ${SLURM_ARRAY_JOB_ID:-none}_${TASK}"
echo "config    : ${CONFIG} window ${WINDOW} seed ${SEED} $*"
echo "cpus      : $(nproc)"
echo "python    : ${PYTHON}"
"$PYTHON" --version

"$PYTHON" reports/mcmc/scripts/joint_mcmc.py --window "$WINDOW" --seed "$SEED" "$@"
