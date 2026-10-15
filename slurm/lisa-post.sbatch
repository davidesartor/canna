#!/bin/bash
# Everything after training, for a run trained in place by lisa.sbatch links (outputs/lisa-<config>,
# on scratch): the scorecards of x and y, the eval's corner pages (with the reference MCMC drawn
# on them), the eval draws of x and y, and the flow-against-MCMC tables. The same steps
# lisa-local.sbatch runs after its training. slurm/lisa-chain.sh queues it after the last link;
# by hand, from the repo root:
#     sbatch slurm/lisa-post.sbatch B-fd-rope-long --n_random 50   # flags go to the scorecards

#SBATCH --job-name=canna-lisa-post
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
#SBATCH --qos=gpu_all

set -euo pipefail
mkdir -p .slurm-logs
CONFIG="$1"; shift
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export UV_SYSTEM_CERTS=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG} $*"
RUN="uv run --extra cuda --frozen python -m canna.lisa"
PY="uv run --extra cuda --frozen python"
DIR="outputs/lisa-$CONFIG"

srun $RUN.scorecard --config "$CONFIG" "$@" || true
srun $RUN.scorecard --config "$CONFIG" --weights training "$@" || true
srun $RUN.eval --config "$CONFIG" || true
srun $PY lisa_checks/eval_draws.py "$CONFIG" outputs "$DIR/eval_draws.npz" || true
srun $PY lisa_checks/eval_draws.py "$CONFIG" outputs "$DIR/eval_draws_training.npz" --weights training || true
for DRAWS in eval_draws.npz eval_draws_training.npz; do
    srun $RUN.mcmc report --config "$CONFIG" --run "$DIR" --draws "$DRAWS" --no_pages || true
done
echo "[post] $DIR: $(ls "$DIR" | tr '\n' ' ')"
