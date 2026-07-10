#!/bin/bash
#SBATCH --job-name=flow_sanity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|v100
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Full 8-param flow-matching sanity check, generative posterior p(x|y).
# NOISE_SCALE=0.0 gives the noiseless case.
# 24h run, checkpoint/eval/corner every CHECKPOINT_EVERY_s. Knobs come from the
# environment (set via `sbatch --export`).
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-300}
RUN_SECONDS=${RUN_SECONDS:-86400}   # override for a short smoke test

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Start time   : $(date)"
echo "N_SOURCES     : ${N_SOURCES:-1}"
echo "NOISE_SCALE   : ${NOISE_SCALE:-0.0}"
echo "CHECK_EVERY   : ${CHECKPOINT_INTERVAL} s"
echo "============================================"

mkdir -p logs

echo "Launching flow_sanity.py (${RUN_SECONDS}s) ..."
uv run python flow_sanity.py --seconds "$RUN_SECONDS"
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
