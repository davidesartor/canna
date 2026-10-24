#!/bin/bash
# Per-window wall time of the flow's posterior sampler (bench_sampling.py beside this file) for the
# given configs at 256 and 1024 draws, on this checkout's src, to data/<tag>.jsonl; then the saved
# MCMC wall times to data/mcmc.json (collect_mcmc.py). Submit from the repo root:
#     sbatch --partition=gpu --account=pi_mehuber_umass_edu --qos=normal --gres=gpu:a100:1 \
#         reports/benchmarks/sampling-cost/scripts/sampling.sbatch a100 narrow-XS narrow-S narrow-B narrow-L narrow-XL

#SBATCH --job-name=canna-sampling-cost
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p .slurm-logs reports/benchmarks/sampling-cost/data
unset "${!LISA_@}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
source slurm/gpu.sh
module list 2>&1 || true
echo "python    : $(command -v "$PYTHON") ($("$PYTHON" --version))"
echo "commit    : $(git rev-parse --short HEAD) $(git status --short src | wc -l) changed in src"

TAG="$1"
shift
"$PYTHON" reports/benchmarks/sampling-cost/scripts/collect_mcmc.py
"$PYTHON" reports/benchmarks/sampling-cost/scripts/bench_sampling.py \
    "reports/benchmarks/sampling-cost/data/$TAG.jsonl" --configs "$@"
