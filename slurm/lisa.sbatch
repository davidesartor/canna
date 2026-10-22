#!/bin/bash
# Train a LISA run; five minutes before the time limit it requeues itself, so one submission trains to the end.
#     sbatch $CANNA_SBATCH_GPU slurm/lisa.sbatch [<config>] [LISA_<FIELD>=value ...]   # a new run (default full-B)
#     sbatch $CANNA_SBATCH_GPU slurm/lisa.sbatch <config> LISA_RUN=<run>              # resume it, on its commit
# L and XL hold 6 and 14 GiB checkpoints: pass --mem=64G.

#SBATCH --job-name=canna-lisa
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
export LISA_CONFIG="${1:-full-B}"
shift || true
if (($#)); then export "$@"; fi

# the run's name: given, kept from this job's first start (a requeue reruns the script), or fresh
NAME_FILE=".slurm-logs/run-${SLURM_JOB_ID:-local}"
if [[ -z "${LISA_RUN:-}" && -f "$NAME_FILE" ]]; then LISA_RUN=$(cat "$NAME_FILE"); fi
if [[ -z "${LISA_RUN:-}" ]]; then
    LISA_RUN=$(uv run --frozen --no-sync python src/canna/lisa/config.py "$LISA_CONFIG")
fi
export LISA_RUN
echo "$LISA_RUN" > "$NAME_FILE"

if [[ -f "${LISA_OUTPUT_DIR:-outputs}/runs/$LISA_RUN/config.yaml" ]]; then source slurm/run-src.sh; fi
source slurm/gpu.sh
echo "run       : $LISA_RUN ($LISA_CONFIG $*)"

# training runs in the background so bash can take the time-limit signal and requeue
trap 'echo "[requeue] time limit near, requeueing ${SLURM_JOB_ID}"; scontrol requeue "$SLURM_JOB_ID"' USR1
srun uv run --extra cuda --frozen python -m canna.lisa.train &
wait $!
