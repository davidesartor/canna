#!/bin/bash -l
# The orientation prior and the helicity chart per band on CPU (scripts/draw_polarization.py):
# angles, z±, round trips, the four-fold degeneracy, the phase reference's Fisher check and the
# Fisher matrix near the poles. The record goes to data/polarization.json; scripts/build_page.py
# then rebuilds the page. Submit from the repo root (WINDOWS=512 for a smoke test):
#     sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/polarization/scripts/polarization.sbatch

#SBATCH --job-name=canna-lisa-polarization
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail
DIR=reports/prior/polarization
mkdir -p .slurm-logs "$DIR/data"
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# a stray LISA_* variable would change config.load's bands
for v in ${!LISA_@}; do unset "$v"; done
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : $(command -v "$PYTHON") $("$PYTHON" --version)"
OUT="$DIR/data/polarization.json"
[ -n "${WINDOWS:-}" ] && OUT="$DIR/data/smoke.json"
"$PYTHON" "$DIR/scripts/draw_polarization.py" "$OUT" ${WINDOWS:-}
[ -n "${WINDOWS:-}" ] || "$PYTHON" "$DIR/scripts/build_page.py"
