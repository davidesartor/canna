#!/bin/bash
# Every band x size LISA training step at one batch, as the trainer runs it (gpu.sh's
# environment, one process per config so each peak is its own): bench_micro.py beside this file,
# retried at 2, 4, then 8 micro-batches on out-of-memory. Records go to data/<tag>.jsonl.
# Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:a100:1 --export=ALL,LISA_BF16=backbone reports/benchmarks/step-time-memory/scripts/sizes.sbatch a100-bf16backbone-b256
#     BANDS=narrow SIZES=XS sbatch ... sizes.sbatch smoke --steps 5     # smoke test
# BATCH (observations, default 256) and BENCH_SRC (a frozen copy of src/, default src) are read
# from the environment; flags after the tag go to bench_micro.py (default --steps 20).

#SBATCH --job-name=canna-lisa-sizes
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p .slurm-logs reports/benchmarks/step-time-memory/data
source slurm/gpu.sh
export PYTHONPATH="$PWD/${BENCH_SRC:-src}"
TAG="$1"
shift
(($#)) || set -- --steps 20
echo "src       : ${PYTHONPATH}"
echo "sweep     : bands ${BANDS:-narrow full}, sizes ${SIZES:-XS S B L XL}, batch ${BATCH:-256}, $*"

for band in ${BANDS:-narrow full}; do
    for size in ${SIZES:-XS S B L XL}; do
        for micro in 1 2 4 8; do
            if LISA_CONFIG="$band-$size" srun uv run --extra cuda --frozen python \
                reports/benchmarks/step-time-memory/scripts/bench_micro.py \
                "reports/benchmarks/step-time-memory/data/$TAG.jsonl" \
                --batch "${BATCH:-256}" --micro "$micro" "$@"; then
                break
            fi
            echo "$band-$size: failed at $micro micro-batches"
        done
    done
done
