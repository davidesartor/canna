#!/bin/bash
# Strub et al. 2022's faint source: its snr over window sizes and frames, then the chains of
# strub_chains.sbatch summarised into reports/diagnostics/strub-source/data/strub_source.json,
# then the page. Submit from the repo root after `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_CPU reports/diagnostics/strub-source/scripts/strub_summary.sbatch

#SBATCH --job-name=canna-strub-summary
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
mkdir -p .slurm-logs
export JAX_PLATFORMS=cpu
for v in ${!LISA_@}; do unset "$v"; done
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "commit    : $(git rev-parse HEAD)"
uv run --frozen python --version
uv run --frozen python reports/diagnostics/strub-source/scripts/strub_source.py summarise "$@"
uv run --frozen python reports/diagnostics/strub-source/scripts/build_page.py
