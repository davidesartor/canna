#!/bin/bash
# The learned loss weights of a run, from its history.npz and last checkpoint. From the repo root:
#     sbatch $CANNA_SBATCH_CPU reports/architecture/learned-loss-weights/scripts/probe.sbatch <run>
# B-size runs on a GPU: source slurm/unity.env; sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/architecture/learned-loss-weights/scripts/probe.sbatch <run>
#SBATCH --job-name=canna-loss-weights
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
HERE=reports/architecture/learned-loss-weights
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
source slurm/run-src.sh
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    source slurm/gpu.sh
    EXTRA=(--extra cuda)
else
    export JAX_PLATFORMS=cpu
    EXTRA=()
fi
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"; echo "job       : ${SLURM_JOB_ID:-none}"; echo "date      : $(date)"
uv run "${EXTRA[@]}" --frozen python "$HERE/scripts/loss_weights_probe.py" "$HERE/data/$LISA_RUN.json"
uv run --frozen python "$HERE/scripts/build_page.py"
