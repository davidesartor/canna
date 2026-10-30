#!/bin/bash
# Time the parts of a LISA training step (bench_step.py beside this file) on this checkout's
# src. Flags after the config go to the script:
#     sbatch $CANNA_SBATCH_GPU reports/benchmarks/step-time-memory/scripts/step.sbatch narrow-B --patches 4 8 16

#SBATCH --job-name=canna-lisa-bench
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
export LISA_CONFIG="$1"; shift
mkdir -p .slurm-logs
source slurm/gpu.sh
echo "bench     : ${LISA_CONFIG} $*"
srun uv run --extra cuda --frozen python reports/benchmarks/step-time-memory/scripts/bench_step.py "$@"
