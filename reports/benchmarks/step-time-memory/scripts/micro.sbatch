#!/bin/bash
# Micro-batch and gradient-accumulation sweep of every band x size LISA training step
# (reports/benchmarks/step-time-memory/scripts/sweep_micro.py), all in one process. Records go to
# reports/benchmarks/step-time-memory/data/<tag>.jsonl. Submit from the repo root
# after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1080_ti:1 \
#         reports/benchmarks/step-time-memory/scripts/micro.sbatch 1080ti --bands full
#     sbatch ... reports/benchmarks/step-time-memory/scripts/micro.sbatch smoke --bands narrow --sizes XS --max-micro 4 --max-batch 16
# Flags go to sweep_micro.py; LISA_<FIELD> variables pass through to every config. Runs in 2 h
# chunks: 5 min before the limit the job requeues itself (same id, args and log) and the sweep
# resumes from its records, up to BENCH_CHUNKS chunks (default 6).

#SBATCH --job-name=canna-lisa-micro
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p .slurm-logs reports/benchmarks/step-time-memory/data
# one arena of 95% of the card, taken up front: freed points coalesce back into it, where
# arenas grown on demand would pile up per point and run out of memory early
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export UV_SYSTEM_CERTS=1

# float32 LisaFlow compiles wrong on these GPUs at XLA's default GEMM autotuning (F20, F22),
# so a run with a float32 backbone (any LISA_BF16 but backbone) switches autotuning off and asks for true fp32 matmuls
# instead of TF32 (F23). bf16-backbone runs are left as they are.
if [[ "${LISA_BF16:-none}" != backbone ]]; then
    XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=0"
    export JAX_DEFAULT_MATMUL_PRECISION=highest
fi
# BENCH_SRC: the frozen copy of src/ submit_step.sh made, so every chunk measures the same network
export PYTHONPATH="$PWD/${BENCH_SRC:-src}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "src       : ${PYTHONPATH}"
echo "sweep     : $*"
echo "chunk     : $((${SLURM_RESTART_COUNT:-0} + 1)) of ${BENCH_CHUNKS:-6}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

TAG="$1"
shift
# near the limit: stop the sweep and requeue, unless this was the last chunk
requeue() {
    kill -TERM "$CHILD" || true
    wait "$CHILD" || true
    if ((${SLURM_RESTART_COUNT:-0} + 1 < ${BENCH_CHUNKS:-6})); then
        echo "time limit near, requeueing"
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit 0
}
trap requeue USR1
uv run --extra cuda --frozen python reports/benchmarks/step-time-memory/scripts/sweep_micro.py "reports/benchmarks/step-time-memory/data/$TAG.jsonl" "$@" &
CHILD=$!
wait "$CHILD"
