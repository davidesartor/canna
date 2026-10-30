#!/bin/bash
# Posterior widths against the ODE integrator (convergence_probe.py beside this file), then the page
# (build_page.py). The record goes to reports/posterior-eval/sampler-convergence/data/<run>.json; it runs on
# the commit the run trained at (slurm/run-src.sh) with the run's eval settings (stray LISA_* cleared).
# Words after the run go to the probe (--draws, --steps, --schedules, --euler). From the repo root, after
# `source slurm/<cluster>.env`; a B run wants a GPU for the full grid (~1.5k network calls per draw):
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch <run>
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch <run> --draws 32 --steps 4,8,16,32,64,128

#SBATCH --job-name=canna-lisa-sampler-convergence
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/sampler-convergence
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
shift
source slurm/run-src.sh
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source slurm/gpu.sh
    EXTRA=(--extra cuda)
else
    export JAX_PLATFORMS=cpu
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    EXTRA=()
fi
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run "${EXTRA[@]}" --frozen python --version
uv run "${EXTRA[@]}" --frozen python -u "$HERE/scripts/convergence_probe.py" "$@"
uv run --frozen python "$HERE/scripts/build_page.py"
