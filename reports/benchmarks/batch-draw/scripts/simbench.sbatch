#!/bin/bash
# Throughput of the LISA simulator on one device (bench_simulator.py beside this file), with the
# trainer's XLA settings (slurm/gpu.sh). Records go to
# reports/benchmarks/batch-draw/data/<tag>.json; the tag cpu runs on the CPU. Sizes after
# the tag. Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:a100:1 reports/benchmarks/batch-draw/scripts/simbench.sbatch a100
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1080_ti:1 reports/benchmarks/batch-draw/scripts/simbench.sbatch 1080ti
#     sbatch $CANNA_SBATCH_CPU --mem=64G reports/benchmarks/batch-draw/scripts/simbench.sbatch cpu 16 64 256 1024 4096

#SBATCH --job-name=canna-lisa-simbench
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

set -euo pipefail
TAG="$1"; shift
mkdir -p .slurm-logs reports/benchmarks/batch-draw/data
source slurm/gpu.sh
if [[ "$TAG" == cpu ]]; then export JAX_PLATFORMS=cpu; fi
srun uv run --extra cuda --frozen python reports/benchmarks/batch-draw/scripts/bench_simulator.py \
    "reports/benchmarks/batch-draw/data/$TAG.json" "$@"
