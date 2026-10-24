#!/bin/bash
# Each eval source's draw widths, the truth's rank among them and the narrowband MCMC's widths (draw_widths.py beside this file),
# on CPU, then the page (build_page.py). The record goes to
# reports/posterior-eval/flow-vs-reference/data/widths/<run>.json, from the run's newest eval at its
# config's weights and ode_steps; MCMC widths only where the narrowband cache exists (the narrow band). Submit from the repo root after
# `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/widths.sbatch <run>

#SBATCH --job-name=canna-lisa-widths
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/flow-vs-reference
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
export JAX_PLATFORMS=cpu
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
echo "run       : ${LISA_RUN}"
uv run --frozen python --version
uv run --frozen python "$HERE/scripts/draw_widths.py"
uv run --frozen python "$HERE/scripts/build_page.py"
