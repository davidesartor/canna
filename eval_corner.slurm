#!/bin/bash
#SBATCH --job-name=eval_corner
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=1080ti
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Posterior corner-plot eval against a flow_sanity.py checkpoint. Draws
# N_POSTERIOR samples in EVAL_BATCH_SIZE-sized chunks (no big FFT batch), so
# it fits on a small GPU. Knobs come from the environment (set via
# `sbatch --export`); N_SOURCES/SIMPLIFIED/TAG_SUFFIX must match the
# checkpoint being evaluated.
# ---------------------------------------------------------------------------

set -uo pipefail

export N_EVAL=${N_EVAL:-3}
export N_POSTERIOR=${N_POSTERIOR:-1024}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
export SIMPLIFIED=${SIMPLIFIED:-0}

echo "============================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "Start time    : $(date)"
echo "N_SOURCES     : ${N_SOURCES:-1}"
echo "MIN_SNR       : ${MIN_SNR:-0.0}"
echo "N_EVAL        : ${N_EVAL}"
echo "N_POSTERIOR   : ${N_POSTERIOR}"
echo "EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}"
echo "SIMPLIFIED    : ${SIMPLIFIED}"
echo "TAG_SUFFIX    : ${TAG_SUFFIX:-}"
echo "============================================"

mkdir -p logs

echo "Launching eval_corner.py ..."
uv run python eval_corner.py
echo "python exit code: $?"

echo "============================================"
echo "Job finished  : $(date)"
echo "============================================"
