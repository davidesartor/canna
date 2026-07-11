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
# Full 8-param (or SIMPLIFIED=1 4-param) flow-matching sanity check, generative
# posterior p(x|y). MIN_SNR/MIN_SNR_FINAL bound the log-space SNR-floor
# curriculum ramp.
# 24h run, checkpointing every CHECKPOINT_INTERVAL_s (training only -- no
# posterior/corner-plot eval; run eval_corner.slurm separately against a
# checkpoint for that). Knobs come from the environment (set via
# `sbatch --export`).
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-300}
export RUN_SECONDS=${RUN_SECONDS:-86400}   # override for a short smoke test
export SIMPLIFIED=${SIMPLIFIED:-0}

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Start time   : $(date)"
echo "N_SOURCES     : ${N_SOURCES:-1}"
echo "MIN_SNR       : ${MIN_SNR:-100.0} -> ${MIN_SNR_FINAL:-1.0}"
echo "CHECK_EVERY   : ${CHECKPOINT_INTERVAL} s"
echo "SIMPLIFIED    : ${SIMPLIFIED}"
echo "============================================"

mkdir -p logs

echo "Launching flow_sanity.py (${RUN_SECONDS}s) SIMPLIFIED=${SIMPLIFIED} ..."
uv run python flow_sanity.py
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
