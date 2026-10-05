#!/bin/bash
# Score a trained LISA flow source by source: f0 width against the ideal, which sources it
# found, and whether the truth falls inside its 68% / 95% intervals. Submit from the repo root:
#     sbatch slurm/lisa-scorecard.sbatch XS-late
#     sbatch slurm/lisa-scorecard.sbatch XS --output_dir outputs/500k   # flags go to scorecard
# Covers the ten eval injections (same keys and draws as eval.py) plus 200 random ones,
# prints a summary to the .out log and writes outputs/lisa-<config>/scorecard.npz.
# Same checkpoint caveat as lisa-eval.sbatch: it reads whatever step is latest.

#SBATCH --job-name=canna-lisa-scorecard
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

# compiled float32 LisaFlow can come out wrong on these GPUs, depending on which matmul
# kernels XLA's autotuner picks (F20). Autotune level 3 fixed the laptop but not the A100,
# whose fp32 posteriors were still 2.4x too narrow (F22), so a float32 run switches GEMM
# autotuning off altogether. It also asks for true fp32 matmuls instead of the TF32 the
# GPU uses by default. bf16 runs are left as they are. The CPU scorecard
# (lisa-scorecard-cpu.sbatch) stays the fp32 reference until an A100 run matches it.
if [[ " ${*:2} " == *" --dtype float32 "* || " ${*:2} " == *" --dtype=float32 "* ]]; then
    XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=0"
    export JAX_DEFAULT_MATMUL_PRECISION=highest
fi

export UV_SYSTEM_CERTS=1   # UV_NATIVE_TLS on uv < 0.12

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
echo "XLA_FLAGS : ${XLA_FLAGS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

srun uv run --extra cuda --frozen python -m canna.lisa.scorecard --config "${CONFIG}" "${@:2}"
