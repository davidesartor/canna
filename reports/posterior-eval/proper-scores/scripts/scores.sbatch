#!/bin/bash
# Proper scores as skill against the mock ladder (scores_probe.py beside this file), then the page (build_page.py).
# CPU only, numpy on saved validation files; writes reports/posterior-eval/proper-scores/data/<run>-<ode_steps>.json
# per eval dir (the first is the paired baseline) and data/mocks-<band>.json per --mocks band. From the repo root,
# after `source slurm/unity.env`; a real narrow-B or full-B run drops in by naming its eval dirs:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/proper-scores/scripts/scores.sbatch \
#         outputs/runs/<run>/eval/<tag-a> outputs/runs/<run>/eval/<tag-b> --mocks narrow full

#SBATCH --job-name=canna-lisa-scores
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/proper-scores
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python --version
uv run --frozen python "$HERE/scripts/scores_probe.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
echo "date      : $(date)"
