#!/bin/bash
# The forward model's checks on CPU (reports/diagnostics/forward-model/scripts/forward_model.py): jaxgb's TDI X
# against Babak et al.'s, the sky-averaged sensitivity's slope, and 3000 pure-noise windows; scripts/build_forward_model.py then rebuilds the page.
# The record goes to reports/diagnostics/forward-model/data/forward-model.json. Submit from the repo root;
# a first argument `smoke` cuts the counts:
#     sbatch --partition=cpu --account=pi_mehuber_umass_edu --qos=normal reports/diagnostics/forward-model/scripts/forward-model.sbatch

#SBATCH --job-name=canna-forward-model
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail
DIR=reports/diagnostics/forward-model
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
uv run --frozen python "$DIR/scripts/forward_model.py" "$DIR/data" "${1:-}"
echo "done      : $(date)"
