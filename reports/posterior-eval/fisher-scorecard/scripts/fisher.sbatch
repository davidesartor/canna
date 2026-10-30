#!/bin/bash
# Fisher scorecard probe (fisher_probe.py beside this file), then the page (build_page.py); numpy only, no model.
# Writes reports/posterior-eval/fisher-scorecard/data/<setting>.json. Default: every mock-flows setting.
# A real run drops in with its eval dirs (any with a scorecard.npz; draws.npz adds the flow-over-Fisher card). From the repo root:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/fisher-scorecard/scripts/fisher.sbatch outputs/runs/<run>/eval/<tag> [...]

#SBATCH --job-name=canna-fisher-scorecard
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
HERE=reports/posterior-eval/fisher-scorecard/scripts
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python "$HERE/fisher_probe.py" "$@"
uv run --frozen python "$HERE/build_page.py"
echo "date      : $(date)"
