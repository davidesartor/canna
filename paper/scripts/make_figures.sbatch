#!/bin/bash
# make_figures.sh on a CPU node; arguments pass through. From the repo root:
#     sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal paper/scripts/make_figures.sbatch
#SBATCH --job-name=paper-figures
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1
for v in ${!LISA_@}; do unset "$v"; done
hostname; date
bash paper/scripts/make_figures.sh "$@"
