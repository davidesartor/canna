#!/bin/bash
# Data-space tests of eval dirs (data_space_checks.py beside this file), then the page (build_page.py).
# CPU only, reads the eval dirs' scorecard.npz (validation samples and scores); writes data/<setting>.json.
# From the repo root, with one bare sbatch per setting:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/data-space-checks/scripts/data_space_checks.sbatch <setting> <band> [--smoke] <eval-dir>...
# mocks:  ... mocks-narrow narrow reports/posterior-eval/mock-flows/data/narrow/{fisher,wide,narrow,biased,prior,phases-prior,loud-floor}
# a run:  ... <run>-narrow-B-32 narrow outputs/runs/<run>/eval/<tag>

#SBATCH --job-name=canna-data-space
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/data-space-checks/scripts
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
SETTING="$1"
BAND="$2"
shift 2
uv run --frozen python "$HERE/data_space_checks.py" --setting "$SETTING" --band "$BAND" "$@"
uv run --frozen python "$HERE/build_page.py"
echo "date      : $(date)"
