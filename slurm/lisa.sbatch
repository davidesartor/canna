#!/bin/bash
# Train the LISA galactic-binary flow. Submit from the repo root:
#     sbatch slurm/lisa.sbatch            # config B
#     sbatch slurm/lisa.sbatch XS         # some other config
# Resuming is automatic: the run picks up from the latest checkpoint in
# outputs/lisa-<config>/checkpoints, so re-submitting after a time limit
# continues rather than restarting.

#SBATCH --job-name=canna-lisa
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
# gpu_a100 / gpu_h100 cap at 1 day; gpu_std at 3; gpu_aivip_max at 10 on the same
# A100 nodes. Resuming works, so a 1-day chunk plus re-submission is equivalent to a
# long allocation -- switch to gpu_aivip_max only if you are entitled to it.

set -euo pipefail
mkdir -p .slurm-logs

CONFIG="${1:-B}"

# one JAX process, one GPU; let it grow into the card rather than preallocating
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --extra cuda: without it uv resolves plain jax and this trains on the CPU.
# --frozen: use the lockfile as-is, so the job never needs to reach the network --
# compute nodes commonly cannot. Run `uv sync --extra cuda` once on a login/visu
# node first, so the venv is already built when the job starts.
srun uv run --extra cuda --frozen python -m canna.lisa.train --config "${CONFIG}"
