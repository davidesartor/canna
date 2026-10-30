#!/bin/bash
# Rank calibration of eval dirs (calibration_probe.py beside this file), then the page (build_page.py).
# With no dirs: every mock flow. Writes reports/posterior-eval/calibration/data/<setting>.json;
# CPU only, on the working tree's src/. A new run drops in by its eval dirs (each needs a scorecard.npz with validation).
# From the repo root, after `source slurm/unity.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/calibration/scripts/calibration.sbatch
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/calibration/scripts/calibration.sbatch outputs/runs/<run>/eval/<tag> [...]

#SBATCH --job-name=canna-lisa-calibration
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/calibration
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python --version
uv run --frozen python -u "$HERE/scripts/calibration_probe.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
echo "date      : $(date)"
