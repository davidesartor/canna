#!/bin/bash
# The joint reference's log L timed on one eval injection, with the truth and the start checked against
# the prior (joint_mcmc.py beside this file), on CPU. The chains themselves are slurm/lisa-mcmc.sbatch.
# Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/joint_mcmc.sbatch narrow-B
# Further flags go to joint_mcmc.py (--injection, --nwalker).

#SBATCH --job-name=canna-joint-mcmc-bench
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
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

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG} $*"
echo "cpus      : $(nproc)"
echo "python    : ${PYTHON}"
"$PYTHON" --version

"$PYTHON" reports/mcmc/scripts/joint_mcmc.py "$@"
