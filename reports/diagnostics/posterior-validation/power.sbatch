#!/bin/bash
# Size and power of the validation tests (test_power.py beside this file), then the page (build_page.py).
# `toy` writes data/toy.json; a run reads its newest validation.npz (from the scorecard) at the commit it trained at and
# writes data/<run>-<ode_steps>.json; LISA_<FIELD>=value pairs pick the eval. From the repo root after
# `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_GPU reports/diagnostics/posterior-validation/power.sbatch toy
#     sbatch $CANNA_SBATCH_GPU reports/diagnostics/posterior-validation/power.sbatch <run> LISA_ODE_STEPS=128

#SBATCH --job-name=canna-lisa-power
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail
HERE=reports/diagnostics/posterior-validation
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
