#!/bin/bash -l
# How much of each prior source's above-noise power R response bins keep, against a 4096-point jaxgb reference, on CPU (scripts/draw_span.py).
# The record goes to reports/prior/response-span/data/; scripts/build_page.py then rebuilds the page.
# An optional argument sets the sources, a multiple of 32 (default 65536, each with 16 A draws; a smoke test takes 32). Submit from the repo root:
#     sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/response-span/scripts/span.sbatch [sources]

#SBATCH --job-name=canna-lisa-prior-response-span
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
DIR=reports/prior/response-span
mkdir -p .slurm-logs "$DIR/data"
export PYTHONUNBUFFERED=1
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
"$PYTHON" "$DIR/scripts/draw_span.py" "$DIR/data" "$@"
"$PYTHON" "$DIR/scripts/build_page.py"
