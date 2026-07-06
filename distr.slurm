#!/bin/bash
#SBATCH --job-name=distr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --mem=30G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

set -uo pipefail

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Start time   : $(date)"
echo "N_SOURCES    : ${N_SOURCES:-1}"
echo "============================================"

mkdir -p logs

uv run python distr.py
echo "python exit code: $?"

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
