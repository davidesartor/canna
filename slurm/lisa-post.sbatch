#!/bin/bash
# Run one post-training module of canna.lisa (scorecard, eval, mcmc report) on a run, on the commit it trained at.
# Words after the run go to `python -m canna.lisa.<...>`; LISA_<FIELD>=value pairs are exported (eval fields only):
#     sbatch $CANNA_SBATCH_GPU slurm/lisa-post.sbatch <run> scorecard
#     sbatch $CANNA_SBATCH_GPU --time=02:00:00 slurm/lisa-post.sbatch <run> eval LISA_WEIGHTS=training
#     sbatch $CANNA_SBATCH_GPU --time=01:00:00 slurm/lisa-post.sbatch <run> mcmc report
# It reads the latest checkpoint, which a live training job can replace mid-read: if the restore fails, resubmit.

#SBATCH --job-name=canna-lisa-post
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
export LISA_RUN="$1"
shift
ARGS=()
for ARG in "$@"; do
    if [[ "$ARG" == LISA_*=* ]]; then export "$ARG"; else ARGS+=("$ARG"); fi
done
source slurm/run-src.sh
source slurm/gpu.sh
srun uv run --extra cuda --frozen python -m "canna.lisa.${ARGS[0]}" "${ARGS[@]:1}"
