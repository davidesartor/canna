#!/bin/bash
# The scorecard on a CPU node: a reference that shares no GPU code path. T3' ran here
# (F21), when compiled float32 LisaFlow was still wrong on the GPUs (F18). That is XLA's
# matmul kernel choice (F20). Autotune level 3 fixed the laptop GPU but not the A100 (F22),
# so this stays the fp32 reference until an A100 run matches it; then prefer the GPU one for
# speed (~1.6 s against ~88 s per injection) and keep this one as the cross-check.
# Submit from the repo root:
#     sbatch slurm/lisa-scorecard-cpu.sbatch XS-late --dtype float32 --n_random 40
# Flags after the config go to scorecard. It writes outputs/lisa-<config>/scorecard_<dtype>_cpu.npz,
# next to the GPU ones. Uses no GPU time; the CPU is ~20-50x slower than an A100, hence
# --n_random to score fewer random injections (the ten eval ones always come first).

#SBATCH --job-name=canna-lisa-scorecard-cpu
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=cpu2022
#SBATCH --account=lisa
# no --qos: the default one (cpu_2022_1280) is the CPU partition's own

set -euo pipefail
mkdir -p .slurm-logs

CONFIG="${1:-B}"

# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu

export UV_SYSTEM_CERTS=1   # UV_NATIVE_TLS on uv < 0.12

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
echo "cpus      : $(nproc)"

# --extra cuda although this runs on the CPU: the venv is shared with the GPU jobs, and
# syncing it without the extra would strip the CUDA packages they need
srun uv run --extra cuda --frozen python -m canna.lisa.scorecard --config "${CONFIG}" "${@:2}"
