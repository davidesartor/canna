#!/bin/bash
#SBATCH --job-name=canna_tests
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Run the tests/ data-inspection suite: draws a real physics batch, asserts
# invariants, and writes text + figures to outputs/inspect/.
# SIMPLIFIED_PROBLEM selects the problem (run once per value for both).
# N_PROBE/PROBE_CHUNK/SEED set the draw. Knobs via `sbatch --export`.
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM:-0}
export N_SOURCES=${N_SOURCES:-1}
export N_PROBE=${N_PROBE:-64}
export PROBE_CHUNK=${PROBE_CHUNK:-16}

echo "============================================"
echo "Job ID            : $SLURM_JOB_ID"
echo "Node              : $SLURM_NODELIST"
echo "SIMPLIFIED_PROBLEM: ${SIMPLIFIED_PROBLEM}"
echo "N_SOURCES         : ${N_SOURCES}"
echo "N_PROBE           : ${N_PROBE} (chunk ${PROBE_CHUNK})"
echo "============================================"

uv run pytest ${PYTEST_ARGS:-tests/} -v
echo "pytest exit code: $?"

echo "Job finished : $(date)"
