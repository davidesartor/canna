#!/bin/bash
# The eval injections' Fisher precision per band on CPU (fisher_diagonal.py beside this file),
# then the page. The record goes to reports/mcmc/data/fisher-diagonal.json. Submit from
# the repo root:
#     sbatch --partition=cpu --account=pi_mehuber_umass_edu --qos=normal reports/mcmc/scripts/fisher.sbatch

#SBATCH --job-name=canna-fisher-forecast
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
DIR=reports/mcmc
mkdir -p .slurm-logs "$DIR/data"
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# a stray LISA_* variable would change config.load's band
for v in ${!LISA_@}; do unset "$v"; done
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "commit    : $(git rev-parse HEAD)"
uv run --frozen python "$DIR/scripts/fisher_diagonal.py" "$DIR/data/fisher-diagonal.json"
uv run --frozen python "$DIR/scripts/build_page.py"
echo "done      : $(date)"
