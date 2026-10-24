#!/bin/bash
# One long cold chain of the joint narrowband MCMC per (window, seed), kept from iteration 0 (long_chain.py
# beside this file), on CPU nodes: one array task per (window, seed), windows 9 5 2 7 x seeds 0-3. The
# traces go to outputs/mcmc/convergence/traces/joint-<problem id>/window<j>_seed<k>.npz (~175 MB each).
# Submit from the repo root; smoke-test first (10 000 iterations of window 9, seeds 0 1, into
# outputs/mcmc/convergence/bench/; the log prints ms/iteration, which sets the full run's --time):
#     source slurm/unity.env
#     sbatch $CANNA_SBATCH_CPU --array=0-1 --time=00:30:00 reports/mcmc/scripts/long_chain.sbatch narrow-B --bench
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/long_chain.sbatch narrow-B
# Further flags go to long_chain.py (--iterations, --thin, --nwalker, --force).

#SBATCH --job-name=canna-mcmc-convergence
#SBATCH --output=.slurm-logs/%x-%A_%a.out
#SBATCH --error=.slurm-logs/%x-%A_%a.err
#SBATCH --array=0-15%4
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
WINDOW="${WINDOWS[$((TASK / 4))]}"
SEED=$((TASK % 4))

echo "host      : $(hostname)"
echo "date      : $(date -Is)"
echo "job       : ${SLURM_ARRAY_JOB_ID:-none}_${TASK}"
echo "config    : ${CONFIG} window ${WINDOW} seed ${SEED} $*"
echo "cpus      : $(nproc)"
echo "python    : ${PYTHON}"
"$PYTHON" --version

"$PYTHON" reports/mcmc/scripts/long_chain.py --window "$WINDOW" --seed "$SEED" "$@"
