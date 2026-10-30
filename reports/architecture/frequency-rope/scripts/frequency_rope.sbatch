#!/bin/bash
# RoPE positions, periods, kernel, bf16 noise and Fisher f0 widths per band (frequency_rope.py beside this file), then the page
# (build_page.py). Writes reports/architecture/frequency-rope/data/rope.json; `--smoke` writes a
# small rope-smoke.json instead. Runs the working-tree src on CPU. From the repo root:
#     sbatch $CANNA_SBATCH_CPU reports/architecture/frequency-rope/scripts/frequency_rope.sbatch [--smoke]

#SBATCH --job-name=canna-frequency-rope
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/architecture/frequency-rope
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python "$HERE/scripts/frequency_rope.py" "$@"
[[ "${1:-}" == "--smoke" ]] || uv run --frozen python "$HERE/scripts/build_page.py"
echo "done      : $(date)"
