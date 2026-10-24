#!/bin/bash
# Per-epoch and per-restart overhead of every band x size LISA run at its default batch, one job
# per chip (reports/benchmarks/training-overhead/scripts/bench_overhead.py), on the CPUs a training job
# gets (slurm/lisa.sbatch). Each config runs in its own process, as a restart does, and compiles
# once. Records go to
# reports/benchmarks/training-overhead/data/<tag>.jsonl; checkpoints are written to the repo's
# outputs/ (where runs keep them) and to the node's local disk, then removed. Submit with
# submit_overhead.sh, or from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch --export=ALL,BENCH_SRC=outputs/bench-src-<stamp>,LISA_BF16=backbone $CANNA_SBATCH_GPU \
#         --gres=gpu:a100:1 reports/benchmarks/training-overhead/scripts/overhead.sbatch a100-bf16backbone
# Configs after the tag (default: all ten); LISA_<FIELD> variables (LISA_BF16, LISA_BATCH_SIZE)
# pass through to every config. 5 min before the limit the job requeues itself (same id, args and
# log) and skips the configs already recorded, up to BENCH_CHUNKS chunks (default 3).

#SBATCH --job-name=canna-lisa-overhead
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p .slurm-logs reports/benchmarks/training-overhead/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# float32 LisaFlow compiles wrong on these GPUs at XLA's default GEMM autotuning (F20, F22),
# so a run with a float32 backbone (any LISA_BF16 but backbone) switches autotuning off and asks for true fp32 matmuls
# instead of TF32 (F23). bf16-backbone runs are left as they are.
if [[ "${LISA_BF16:-none}" != backbone ]]; then
    XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=0"
    export JAX_DEFAULT_MATMUL_PRECISION=highest
fi
export UV_SYSTEM_CERTS=1
# BENCH_SRC: the frozen copy of src/ submit_overhead.sh made, so every chunk measures the same network
export PYTHONPATH="$PWD/${BENCH_SRC:-src}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOCAL="${TMPDIR:-/tmp}/canna-overhead-${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : ${PYTHON}"
echo "src       : ${PYTHONPATH}"
echo "local     : ${LOCAL}"
echo "chunk     : $((${SLURM_RESTART_COUNT:-0} + 1)) of ${BENCH_CHUNKS:-3}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
df -h outputs "$LOCAL"

TAG="$1"
shift
CONFIGS=("$@")
((${#CONFIGS[@]})) || CONFIGS=(narrow-XS narrow-S narrow-B narrow-L narrow-XL full-XS full-S full-B full-L full-XL)
OUT="reports/benchmarks/training-overhead/data/$TAG.jsonl"

# what the training job's `srun uv run --extra cuda --frozen` adds before Python starts
start=$(date +%s.%N)
srun uv run --extra cuda --frozen python -c pass
export BENCH_UV_S=$(awk "BEGIN { print $(date +%s.%N) - $start }")
echo "uv run    : ${BENCH_UV_S} s"

# near the limit: stop the bench, drop its half-written checkpoint and requeue, unless this was
# the last chunk
requeue() {
    kill -TERM "$CHILD" || true
    wait "$CHILD" || true
    rm -rf outputs/bench-checkpoint-*-"$CHILD" outputs/bench-history-*-"$CHILD"
    if ((${SLURM_RESTART_COUNT:-0} + 1 < ${BENCH_CHUNKS:-3})); then
        echo "time limit near, requeueing"
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit 0
}
trap requeue USR1
# one process per config, as each training restart is: no device memory or autotuning carried over
for CONFIG in "${CONFIGS[@]}"; do
    BENCH_T0=$(date +%s.%N) "$PYTHON" reports/benchmarks/training-overhead/scripts/bench_overhead.py "$OUT" \
        --configs "$CONFIG" --targets outputs "$LOCAL" &
    CHILD=$!
    wait "$CHILD"
done
