#!/bin/bash -l
# The sky page's draws (CPU, ~1 min). Submit from the repo root:
#   sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/sky/scripts/sky.sbatch
# Smoke test: --export=ALL,SKY_DRAWS=65536,SKY_WINDOWS=256
#SBATCH --job-name=canna-prior-sky
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
DIR=reports/prior/sky
mkdir -p .slurm-logs "$DIR/data"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
for v in ${!LISA_@}; do unset "$v"; done
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="$DIR/data/sky.json"
[[ -n "${SKY_DRAWS:-}" ]] && OUT="$DIR/data/sky-smoke.json"
echo "host      : $(hostname)"; echo "date      : $(date)"; echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : $(command -v "$PYTHON") $("$PYTHON" --version)"
"$PYTHON" "$DIR/scripts/draw_sky.py" "$OUT"
[[ -n "${SKY_DRAWS:-}" ]] || "$PYTHON" "$DIR/scripts/build_page.py"
