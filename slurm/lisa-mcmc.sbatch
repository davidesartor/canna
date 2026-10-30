#!/bin/bash
# The joint MCMC of a problem's eval injections on CPU nodes, one array task per (injection, seed) chain,
# cached in outputs/mcmc/joint-<problem id>/, which `mcmc pack` turns into reports/mcmc/data/eval-<id>.npz:
#     sbatch $CANNA_SBATCH_CPU --array=0-$(( $(LISA_CONFIG=narrow-B uv run --frozen python -m canna.lisa.mcmc status --count) - 1 )) \
#         slurm/lisa-mcmc.sbatch narrow-B

#SBATCH --job-name=canna-mcmc
#SBATCH --output=.slurm-logs/%x-%A_%a.out
#SBATCH --error=.slurm-logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
export LISA_CONFIG="$1"; shift
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0}"
echo "config    : ${LISA_CONFIG} $*"
echo "cpus      : $(nproc)"
uv run --frozen python -m canna.lisa.mcmc run \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
