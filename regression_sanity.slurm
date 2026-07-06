#!/bin/bash
#SBATCH --job-name=regr_2src
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Point-estimate regression, all 8 params supervised per source.
# NOISE_SCALE=0.0 gives the noiseless case.
# 24h run, eval every 15 min. Knobs come from the environment (set via
# `sbatch --export`).
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export CHECKPOINT_EVERY_s=${CHECKPOINT_EVERY_s:-60}   # checkpoint/eval/plot every minute

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Start time   : $(date)"
echo "N_SOURCES     : ${N_SOURCES:-1}"
echo "NOISE_SCALE   : ${NOISE_SCALE:-1.0}"
echo "CHECK_EVERY   : ${CHECKPOINT_EVERY_s} s"
echo "============================================"

mkdir -p logs

echo "Launching regression_sanity.py (24h) ..."
uv run python regression_sanity.py --seconds 86400
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
