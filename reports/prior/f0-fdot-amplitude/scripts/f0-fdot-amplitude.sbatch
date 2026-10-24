#!/bin/bash -l
# LisaGB's (f0, fdot, A) prior drawn per band on CPU (reports/prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py):
# every source's parameters, snr and z at the prior's frame, plus the prior's curves, then the prior cut at each visible snr.
# The draws go to reports/prior/f0-fdot-amplitude/data/{narrow,full}{,-cut}.npz and f0-fdot-amplitude.json; scripts/build_page.py then
# rebuilds the page. Submit from the repo root, optionally with fewer windows for a smoke test:
#     sbatch $CANNA_SBATCH_CPU reports/prior/f0-fdot-amplitude/scripts/f0-fdot-amplitude.sbatch [windows]

#SBATCH --job-name=canna-lisa-prior-f0-fdot-amplitude
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
DIR=reports/prior/f0-fdot-amplitude
mkdir -p .slurm-logs "$DIR/data"
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
# a stray LISA_* variable would change config.load's bands
for v in ${!LISA_@}; do unset "$v"; done
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : $(command -v "$PYTHON") $("$PYTHON" --version)"
"$PYTHON" "$DIR/scripts/draw_f0_fdot_amplitude.py" "$DIR/data" "$@"
"$PYTHON" "$DIR/scripts/build_page.py"
