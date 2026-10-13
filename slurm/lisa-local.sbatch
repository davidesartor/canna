#!/bin/bash
# Train, score (averaged weights x and training weights y) and evaluate one run in a single
# job, on the node's local /tmp, then copy the run folder home to outputs/lisa-<config>. For
# runs that fit one 24 h slot and whose checkpoints the home quota cannot hold while
# training (schedule-free checkpoints carry x, z and the optimizer moments). Submit from the
# repo root:
#     sbatch slurm/lisa-local.sbatch XS-fd
#     sbatch slurm/lisa-local.sbatch B-fd --n_random 50     # flags go to the scorecards
# No resume: a job that dies loses its run.

#SBATCH --job-name=canna-lisa-local
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
#SBATCH --qos=gpu_all

set -euo pipefail
mkdir -p .slurm-logs
CONFIG="$1"; shift
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export UV_SYSTEM_CERTS=1
SCRATCH="/tmp/canna-${SLURM_JOB_ID:-local}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG} $*"
df -h /tmp | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
RUN="uv run --extra cuda --frozen python -m canna.lisa"

srun $RUN.train --config "$CONFIG" --output_dir "$SCRATCH"
srun $RUN.scorecard --config "$CONFIG" --output_dir "$SCRATCH" "$@" || true
srun $RUN.scorecard --config "$CONFIG" --output_dir "$SCRATCH" --weights training "$@" || true
srun $RUN.eval --config "$CONFIG" --output_dir "$SCRATCH" || true
mkdir -p outputs
cp -r "$SCRATCH/lisa-$CONFIG" outputs/
echo "[kept] outputs/lisa-$CONFIG: $(ls outputs/lisa-$CONFIG | tr '\n' ' ')"
