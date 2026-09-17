#!/bin/bash
# Evaluate a trained LISA flow against its latest checkpoint. Submit from the repo root:
#     sbatch slurm/lisa-eval.sbatch            # config B
#     sbatch slurm/lisa-eval.sbatch XS         # some other config
# Writes ten corner plots -- one per SNR decile, flow against a Fisher approximation --
# to outputs/lisa-<config>/corner.
#
# This reads the same checkpoint directory the training job writes, and that manager
# keeps only the latest step, so a checkpoint can be replaced underneath a concurrent
# eval. Orbax writes to a temporary directory and renames, so a torn read is unlikely
# rather than impossible; if the restore does fail, just resubmit.

#SBATCH --job-name=canna-lisa-eval
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
#SBATCH --qos=gpu_all

set -euo pipefail
mkdir -p .slurm-logs

CONFIG="${1:-B}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# same cuDNN CUDA-graph capture failure as the training job: building the graph dies
# with cudaGraphAddMemsetNode "invalid argument" on these nodes, so switch command
# buffers off and let the kernels launch one at a time
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

export UV_SYSTEM_CERTS=1   # UV_NATIVE_TLS on uv < 0.12

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

srun uv run --extra cuda --frozen python -m canna.lisa.eval --config "${CONFIG}"
