#!/bin/bash
# Train and score one variant in the node's local /tmp, keeping only the scorecard, the loss
# plot and the log: for short comparison runs that the home quota could not hold. Submit from
# a repo root, with a tag and the flags that make the variant:
#     PYTHON=~/canna/.venv/bin/python sbatch slurm/lisa-compare.sbatch A XS-fd --total_steps 150000
# Results go to outputs/compare/<tag>/. The checkpoints are discarded with the job.

#SBATCH --job-name=canna-lisa-compare
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
#SBATCH --qos=gpu_all

set -euo pipefail
mkdir -p .slurm-logs
TAG="$1"; CONFIG="$2"; shift 2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
SCRATCH="/tmp/canna-${SLURM_JOB_ID:-local}"
KEEP="outputs/compare/${TAG}"
mkdir -p "$SCRATCH" "$KEEP"
trap 'rm -rf "$SCRATCH"' EXIT
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "variant   : ${TAG}: ${CONFIG} $*"
df -h /tmp | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

srun "$PYTHON" -m canna.lisa.train --config "$CONFIG" --output_dir "$SCRATCH" "$@"
srun "$PYTHON" -m canna.lisa.scorecard --config "$CONFIG" --output_dir "$SCRATCH" "$@"
srun "$PYTHON" -m canna.lisa.scorecard --config "$CONFIG" --output_dir "$SCRATCH" --weights training "$@" || true
cp "$SCRATCH"/lisa-"$CONFIG"/*.npz "$SCRATCH"/lisa-"$CONFIG"/losses.pdf "$KEEP"/
echo "[kept] $KEEP: $(ls "$KEEP" | tr '\n' ' ')"
