#!/bin/bash
# Burn-in, kept length and thinning from the saved long chains (analyse_convergence.py beside this file),
# on CPU, once every task of long_chain.sbatch has saved its trace: the recommendation to data/convergence/summary.json,
# the curves to data/convergence/curves.npz; then export.sbatch refreshes the traces the page shows. Submit from the repo root:
#     source slurm/unity.env
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/analyse.sbatch narrow-B --bench --windows 15
#     sbatch $CANNA_SBATCH_CPU --dependency=afterok:<chains job id> reports/mcmc/scripts/analyse.sbatch narrow-B
# LISA_<FIELD> variables pass through and must match the chains'.

#SBATCH --job-name=canna-mcmc-convergence-analyse
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
mkdir -p .slurm-logs
export LISA_CONFIG="$1"; CONFIG="$1"; shift
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "host      : $(hostname)"
echo "date      : $(date -Is)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG} $*"
echo "python    : ${PYTHON}"
"$PYTHON" --version

"$PYTHON" reports/mcmc/scripts/analyse_convergence.py "$@"
