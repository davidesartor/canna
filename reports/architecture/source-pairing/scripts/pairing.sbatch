#!/bin/bash -l
# The source-pairing page's path lengths (CPU, network-free). Submit from the repo root:
#   sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/architecture/source-pairing/scripts/pairing.sbatch
# Smoke test: --export=ALL,PAIRING_WINDOWS=2048
#SBATCH --job-name=canna-source-pairing
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
DIR=reports/architecture/source-pairing
mkdir -p .slurm-logs "$DIR/data"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
for v in ${!LISA_@}; do unset "$v"; done
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="$DIR/data/pairing.json"
[[ -n "${PAIRING_WINDOWS:-}" ]] && OUT="$DIR/data/pairing-smoke.json"
echo "host      : $(hostname)"; echo "date      : $(date)"; echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : $(command -v "$PYTHON") $("$PYTHON" --version)"
"$PYTHON" "$DIR/scripts/pairing.py" "$OUT"
[[ -n "${PAIRING_WINDOWS:-}" ]] || "$PYTHON" "$DIR/scripts/build_page.py"
