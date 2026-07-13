#!/bin/bash
#SBATCH --job-name=canna_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|v100
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Evaluate a trained checkpoint: flow posterior corner plots across SNR
# quantiles. Import-time toggles (SIMPLIFIED_PROBLEM, N_SOURCES, REMAT_LEVEL)
# must match the training run that produced the checkpoint, else the run tag /
# model dims won't match. Knobs come from the environment (`sbatch --export`).
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM:-0}
export N_SOURCES=${N_SOURCES:-1}

echo "============================================"
echo "Job ID           : $SLURM_JOB_ID"
echo "Node             : $SLURM_NODELIST"
echo "Start time       : $(date)"
echo "SIMPLIFIED_PROBLEM: ${SIMPLIFIED_PROBLEM}"
echo "N_SOURCES        : ${N_SOURCES}"
echo "============================================"

mkdir -p logs

echo "Launching scripts/eval.py SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM} N_SOURCES=${N_SOURCES} ..."
uv run python scripts/eval.py
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
