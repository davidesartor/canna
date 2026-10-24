#!/bin/bash
# Velocity probe of a run's checkpoint (velocity_probe.py beside this file), then the page
# (build_page.py). The record goes to reports/posterior-eval/flow-vs-reference/data/probe/<tag>.json
# (tag defaults to <run>; the page shows each run's newest-epoch record); it runs on the commit
# the run trained at (slurm/run-src.sh),
# with the run's own eval settings (stray LISA_* variables are cleared; a third argument `training`
# probes the training weights instead of the average). From the repo root, after
# `source slurm/<cluster>.env`; an XS or S run fits a CPU node, a larger one wants a GPU:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/probe.sbatch <run> [tag] [training]
#     sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/posterior-eval/flow-vs-reference/scripts/probe.sbatch <run> [tag] [training]
# For a job still training, live.sh beside this file probes a snapshot of its checkpoint instead.

#SBATCH --job-name=canna-lisa-probe
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/flow-vs-reference
mkdir -p .slurm-logs "$HERE/data/probe"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
[[ -z "${3:-}" ]] || export LISA_WEIGHTS="$3"
source slurm/run-src.sh
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source slurm/gpu.sh
    EXTRA=(--extra cuda)
else
    export JAX_PLATFORMS=cpu
    EXTRA=()
fi
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
OUT="$HERE/data/probe/${2:-$LISA_RUN}.json"
uv run "${EXTRA[@]}" --frozen python "$HERE/scripts/velocity_probe.py" "$OUT"
uv run --frozen python "$HERE/scripts/build_page.py"
