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
# the default QOS (cpu_2022_1280) is rejected by the GPU partitions, which take
# only normal or gpu_all; check `sacctmgr show assoc user=$USER format=qos` if
# gpu_all is refused too
#SBATCH --qos=gpu_all
# gpu_a100 / gpu_h100 cap at 1 day; gpu_std at 3; gpu_aivip_max at 10 on the same
# A100 nodes. Resuming works, so a 1-day chunk plus re-submission is equivalent to a
# long allocation -- switch to gpu_aivip_max only if you are entitled to it.

set -euo pipefail
mkdir -p .slurm-logs

CONFIG="${1:-B}"

# one JAX process, one GPU; let it grow into the card rather than preallocating
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# XLA folds the cuDNN attention kernel into a CUDA graph, and on these nodes building
# that graph fails outright -- cudaGraphAddMemsetNode returns "invalid argument", which
# killed job 10876803 two minutes in. An empty list switches command buffers off; the
# same kernels still run, they are just launched one at a time instead of replayed as a
# graph, and at batch 256 the launch overhead is lost in the noise.
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# the CNES egress proxy re-signs TLS, so uv's bundled roots reject it; this points
# uv at the system trust store, which carries the CNES CA
export UV_SYSTEM_CERTS=1   # UV_NATIVE_TLS on uv < 0.12

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# --extra cuda: without it uv resolves plain jax and this trains on the CPU.
# --frozen: use the lockfile as-is, so the job never needs to reach the network --
# compute nodes commonly cannot. Run `uv sync --extra cuda` once on a login/visu
# node first, so the venv is already built when the job starts.
srun uv run --extra cuda --frozen python -m canna.lisa.train --config "${CONFIG}"
