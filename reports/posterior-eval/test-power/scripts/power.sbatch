#!/bin/bash
# Size and power of the validation tests (test_power.py beside this file), then the page (build_page.py).
# `toy` writes data/toy.json on CPU; a run reads its newest scorecard.npz validation samples at the commit it
# trained at, needs a GPU and writes data/<run>-<ode_steps>.json; LISA_<FIELD>=value pairs pick the eval.
# From the repo root after `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/test-power/scripts/power.sbatch toy
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1 --time=03:00:00 --mem=32G reports/posterior-eval/test-power/scripts/power.sbatch <run> LISA_ODE_STEPS=128

#SBATCH --job-name=canna-lisa-power
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/test-power/scripts
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export PYTHONUNBUFFERED=1
if [[ "$1" != toy ]]; then
    export LISA_RUN="$1"
    for ARG in "${@:2}"; do export "$ARG"; done
    source slurm/run-src.sh
fi
source slurm/gpu.sh
uv run --frozen --extra cuda python "$HERE/test_power.py" "$([[ "$1" == toy ]] && echo toy || echo lisa)"
uv run --frozen --extra cuda python "$HERE/build_page.py"
