#!/bin/bash
# Mock flows of one band (mock_eval.py beside this file), then the page (build_page.py).
# Writes reports/posterior-eval/mock-flows/data/<band>/<mock>/; CPU only, on the working tree's src/.
# From the repo root, after `source slurm/unity.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/mock-flows/scripts/mock.sbatch <band> [--mocks a,b] [--n-validation N] [--n-random N]
# All seven mocks of a band take ~40 min on 4 CPUs (narrow); split them over two jobs with --mocks to stay under the hour.

#SBATCH --job-name=canna-lisa-mocks
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/mock-flows
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python "$HERE/scripts/mock_eval.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
echo "date      : $(date)"
