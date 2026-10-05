#!/bin/bash
# Evaluate a trained LISA flow against its latest checkpoint. Submit from the repo root:
#     sbatch slurm/lisa-eval.sbatch            # config B
#     sbatch slurm/lisa-eval.sbatch XS         # some other config
#     sbatch slurm/lisa-eval.sbatch XS --output_dir outputs/500k   # flags go to eval
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

# compiled float32 LisaFlow comes out wrong on these GPUs at XLA's default GEMM autotuning
# level (4): one autotuned matmul is garbage inside the full program, though right on its
# own (F20). Level 3 still autotunes and matches the CPU. A float32 run also asks for true
# fp32 matmuls instead of the TF32 the GPU uses by default; bf16 runs are left as they are.
if [[ " ${*:2} " == *" --dtype float32 "* || " ${*:2} " == *" --dtype=float32 "* ]]; then
    XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=3"
    export JAX_DEFAULT_MATMUL_PRECISION=highest
fi

export UV_SYSTEM_CERTS=1   # UV_NATIVE_TLS on uv < 0.12

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
echo "XLA_FLAGS : ${XLA_FLAGS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

srun uv run --extra cuda --frozen python -m canna.lisa.eval --config "${CONFIG}" "${@:2}"
