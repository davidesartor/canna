#!/bin/bash
# The flow on misspecified windows (robustness_probe.py beside this file), then the page (build_page.py).
# The record goes to reports/posterior-eval/robustness/data/<run>.json; it runs on the working
# tree's src with the run's eval settings (stray LISA_* cleared). Words after the run go
# to the probe (--draws, --injections, --ode-steps, --cases). From the repo root, after `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/posterior-eval/robustness/scripts/robustness.sbatch <run>
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/robustness/scripts/robustness.sbatch <run> --draws 64

#SBATCH --job-name=canna-lisa-robustness
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/robustness
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
shift
export PYTHONPATH="$PWD/src"
export LISA_BF16="$(sed -n 's/^bf16: *//p' "outputs/runs/$LISA_RUN/config.yaml" | tr -d "'\"")"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source slurm/gpu.sh
else
    export JAX_PLATFORMS=cpu
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
"$PYTHON" --version
"$PYTHON" -u "$HERE/scripts/robustness_probe.py" "$@"
"$PYTHON" "$HERE/scripts/build_page.py"
