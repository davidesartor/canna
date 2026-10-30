#!/bin/bash
# LisaFlow's anatomy (anatomy.py beside this file) for the given configs (default: all ten), appended to
# reports/architecture/network-anatomy/data/anatomy.jsonl (configs already there are skipped), then the page.
# Abstract shapes only, so a CPU node; from the repo root, after `source slurm/unity.env`:
#     sbatch $CANNA_SBATCH_CPU reports/architecture/network-anatomy/scripts/anatomy.sbatch [config ...]

#SBATCH --job-name=canna-network-anatomy
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/architecture/network-anatomy
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
echo "commit    : $(git rev-parse --short HEAD) $(git status --porcelain src | wc -l) changed files in src/"
uv run --frozen python -u "$HERE/scripts/anatomy.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
