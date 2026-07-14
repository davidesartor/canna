#!/bin/bash
#SBATCH --job-name=canna_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Train the flow-matching posterior p(x|y). Training only -- no corner-plot
# eval; run slurm/eval.slurm separately against a checkpoint for that.
# SIMPLIFIED_PROBLEM=1 = reduced month/4-param problem; unset/0 = year/8-param.
# N_SOURCES (separate axis, default 1) sets the GB source count in [1,4].
# These two import-time toggles are the only env knobs; all other training
# settings (run length, batch size, LR, ...) are hard-coded in scripts/train.py.
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

echo "Launching scripts/train.py SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM} N_SOURCES=${N_SOURCES} ..."
uv run python scripts/train.py
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
