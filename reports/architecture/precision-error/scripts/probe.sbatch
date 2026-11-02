#!/bin/bash
# Per-block and end-to-end error of each bf16 mode against a float32 forward pass on a run's params, on the working tree's src.
# From the repo root, with PYTHON=<venv python> exported (default .venv/bin/python):
#     sbatch $CANNA_SBATCH_CPU reports/architecture/precision-error/scripts/probe.sbatch <run>
# On a GPU, where bf16 accumulates as it does in training: sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/architecture/precision-error/scripts/probe.sbatch <run>
#SBATCH --job-name=canna-precision-error
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
HERE=reports/architecture/precision-error
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
export LISA_BF16="$(sed -n 's/^bf16: *//p' "outputs/runs/$LISA_RUN/config.yaml" | head -1 | tr -d "'\"")"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # the float32 reference needs gpu.sh's float32 flags (no autotune, highest matmul precision) whatever the run trained in
    RUN_BF16="$LISA_BF16"
    export LISA_BF16=none
    source slurm/gpu.sh
    export LISA_BF16="$RUN_BF16"
else
    export JAX_PLATFORMS=cpu
fi
export PYTHONUNBUFFERED=1
echo "host      : $(hostname)"; echo "job       : ${SLURM_JOB_ID:-none}"; echo "date      : $(date)"
"$PYTHON" "$HERE/scripts/precision_probe.py" "$HERE/data/$LISA_RUN-${JAX_PLATFORMS:-gpu}.json"
"$PYTHON" "$HERE/scripts/build_page.py"
