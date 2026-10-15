#!/bin/bash
# The narrowband MCMC of a problem's eval sources (canna.lisa.mcmc), on CPU nodes: one array
# task per (source, seed) chain. The chains depend on the problem and the eval seed only, so
# they are cached under outputs/mcmc/eval-<problem id>/ and every run of that problem reuses
# them; eval.py draws them on its corner pages and lisa-local.sbatch tabulates the flow
# against them. Training jobs submit this themselves when the chains are missing
# (slurm/mcmc-ensure.sh); by hand, from the repo root:
#     sbatch --array=0-$(( $(.venv313/bin/python -m canna.lisa.mcmc status --config XS-fd --count) - 1 )) \
#         slurm/lisa-mcmc.sbatch XS-fd
# Needs the Python 3.13 venv with jexplore (.venv313; research log section 7).

#SBATCH --job-name=canna-mcmc
#SBATCH --output=.slurm-logs/%x-%A_%a.out
#SBATCH --error=.slurm-logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu2022
#SBATCH --account=lisa
# no --qos: the default one (cpu_2022_1280) is the CPU partition's own

set -euo pipefail
mkdir -p .slurm-logs
CONFIG="$1"; shift
export JAX_PLATFORMS=cpu
echo "host      : $(hostname)"
echo "job       : ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0}"
echo "config    : ${CONFIG} $*"
echo "cpus      : $(nproc)"
.venv313/bin/python -m canna.lisa.mcmc run --config "$CONFIG" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
