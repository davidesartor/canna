#!/bin/bash
# LisaGB's prior drawn per band on CPU (reports/prior/scripts/draw_prior.py).
# Also the galactic sky prior's density by quadrature in three frames (4M points each, hence 8G),
# the windows and sources per run, and the sky-averaged snr constant C on an f0 grid under an
# isotropic and the galactic sky.
# The record goes to reports/prior/data/prior.json; scripts/build_{frequency,amplitude,sky}.py then rebuild the pages. Submit from
# the repo root after `source slurm/<cluster>.env` (unity or trex):
#     sbatch $CANNA_SBATCH_CPU reports/prior/scripts/prior.sbatch

#SBATCH --job-name=canna-lisa-prior
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
DIR=reports/prior
mkdir -p .slurm-logs "$DIR/data"
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
# a stray LISA_* variable would change config.load's bands
for v in ${!LISA_@}; do unset "$v"; done
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : $(command -v "$PYTHON") $("$PYTHON" --version)"
"$PYTHON" "$DIR/scripts/draw_prior.py" "$DIR/data/prior.json"
