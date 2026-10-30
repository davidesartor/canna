#!/bin/bash
# Health of the cached joint reference chains (collect_chains.py beside this file), on CPU:
# reports/mcmc/data/joint-<problem id>/chains.json, which the paper's MCMC
# numbers and figure read. Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/collect.sbatch narrow-B

#SBATCH --job-name=canna-joint-collect
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
mkdir -p .slurm-logs
export LISA_CONFIG="$1"; CONFIG="$1"; shift
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG}"
"$PYTHON" --version

"$PYTHON" reports/mcmc/scripts/collect_chains.py
