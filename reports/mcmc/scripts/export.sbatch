#!/bin/bash
# Thinned reference and Fisher draws of every eval window, and the long-chain traces (export_draws.py
# beside this file), on CPU: reports/mcmc/data/draws.json and
# data/convergence/traces.json, then the page that inlines them. Extra args pass to export_draws.py.
# Submit from the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/export.sbatch narrow-B

#SBATCH --job-name=canna-reference-export
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

"$PYTHON" reports/mcmc/scripts/export_draws.py "$@"
"$PYTHON" reports/mcmc/scripts/build_page.py
