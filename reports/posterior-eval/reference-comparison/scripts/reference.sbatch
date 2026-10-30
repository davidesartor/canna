#!/bin/bash
# Flow draws against the MCMC reference (reference_probe.py beside this file), on CPU, then the page (build_page.py).
# Settings are name=eval_dir pairs; the records go to reports/posterior-eval/reference-comparison/data/<name>.json.
# A run needs its --run; the mocks use the narrow-B config. From the repo root, after `source slurm/unity.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch --run <run> <run>-32=outputs/runs/<run>/eval/<epoch>-average-32
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch mock-fisher=reports/posterior-eval/mock-flows/data/narrow/fisher ...

#SBATCH --job-name=canna-reference-comparison
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/reference-comparison
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
echo "args      : $*"
uv run --frozen python --version
uv run --frozen python -u "$HERE/scripts/reference_probe.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
