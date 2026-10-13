#!/bin/bash
# Time the parts of a LISA training step on an A100 (lisa_checks/bench_step.py). Submit from
# a repo root:
#     sbatch slurm/lisa-bench.sbatch XS-fd --patches 4 8 16
# PYTHON overrides the interpreter, e.g. another clone's venv, so a scratch checkout can be
# timed without building its own (and the venv's editable install is left alone): the code
# comes from this checkout's src via PYTHONPATH.

#SBATCH --job-name=canna-lisa-bench
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --account=lisa
#SBATCH --qos=gpu_all

set -euo pipefail
mkdir -p .slurm-logs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-.venv/bin/python}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "python    : ${PYTHON}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
srun "$PYTHON" lisa_checks/bench_step.py "$@"
