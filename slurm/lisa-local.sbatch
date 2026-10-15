#!/bin/bash
# Train, score (averaged weights x and training weights y) and evaluate one run in a single
# job, on the node's local /tmp, then copy the run folder home to outputs/lisa-<config>. For
# runs that fit one 24 h slot and whose checkpoints the home quota cannot hold while
# training (schedule-free checkpoints carry x, z and the optimizer moments). Submit from the
# repo root:
#     sbatch slurm/lisa-local.sbatch XS-fd
#     sbatch slurm/lisa-local.sbatch B-fd --n_random 50     # flags go to the scorecards
# No resume: a job that dies loses its run. The posterior draws of the ten eval injections
# (eval_draws.npz for x, eval_draws_training.npz for y) are written for the MCMC comparison.
# The optimizer state is kept: schedule-free's averaged weights x, the model the scorecards
# evaluate, live in it, and dropping it lost x for three runs (F43). KEEP_OPT_STATE=0 drops it
# (y only, ~0.4 GB instead of ~0.9). The copy home holds the checkpoint, the histories, the
# scorecards, the corner pages and the draws. KEEP_AS=<name> stores the run as outputs/<name>
# instead of outputs/lisa-<config>, so a rerun of a config does not merge into an older run.
# At its start the job submits the problem's reference MCMC if it has none (slurm/mcmc-ensure.sh);
# the corner pages then show it, and mcmc_compare.txt tabulates the flow against it.

#SBATCH --job-name=canna-lisa-local
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=24:00:00
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
SCRATCH="/tmp/canna-${SLURM_JOB_ID:-local}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "config    : ${CONFIG} $*"
df -h /tmp | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
RUN="uv run --extra cuda --frozen python -m canna.lisa"

# the reference MCMC of the eval sources, on CPU nodes, if this problem has none yet: it is
# done long before the eval, which draws it on the corner pages
slurm/mcmc-ensure.sh "$CONFIG" || true

srun $RUN.train --config "$CONFIG" --output_dir "$SCRATCH"
srun $RUN.scorecard --config "$CONFIG" --output_dir "$SCRATCH" "$@" || true
srun $RUN.scorecard --config "$CONFIG" --output_dir "$SCRATCH" --weights training "$@" || true
srun $RUN.eval --config "$CONFIG" --output_dir "$SCRATCH" || true
PY="uv run --extra cuda --frozen python"
srun $PY lisa_checks/eval_draws.py "$CONFIG" "$SCRATCH" "$SCRATCH/lisa-$CONFIG/eval_draws.npz" || true
srun $PY lisa_checks/eval_draws.py "$CONFIG" "$SCRATCH" "$SCRATCH/lisa-$CONFIG/eval_draws_training.npz" --weights training || true
# the flow against the MCMC, per source and parameter (mcmc_compare*.txt in the run folder)
for DRAWS in eval_draws.npz eval_draws_training.npz; do
    srun $RUN.mcmc report --config "$CONFIG" --run "$SCRATCH/lisa-$CONFIG" --draws "$DRAWS" --no_pages || true
done
if [[ "${KEEP_OPT_STATE:-1}" != "1" ]]; then rm -rf "$SCRATCH/lisa-$CONFIG"/checkpoints/*/opt_state; fi
DEST="outputs/${KEEP_AS:-lisa-$CONFIG}"
mkdir -p outputs
if [[ -e "$DEST" ]]; then DEST="$DEST-$SLURM_JOB_ID"; fi
cp -r "$SCRATCH/lisa-$CONFIG" "$DEST"
echo "[kept] $DEST: $(ls "$DEST" | tr '\n' ' ')"
