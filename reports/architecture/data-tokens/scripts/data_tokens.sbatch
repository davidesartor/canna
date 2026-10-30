#!/bin/bash
# Token layout and source spans per band (data_tokens.py beside this file), then the page
# (build_page.py). Writes reports/architecture/data-tokens/data/tokens.json; `--smoke` writes a
# small tokens-smoke.json instead. Runs the working-tree src on CPU. From the repo root:
#     sbatch $CANNA_SBATCH_CPU reports/architecture/data-tokens/scripts/data_tokens.sbatch [--smoke]

#SBATCH --job-name=canna-data-tokens
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/architecture/data-tokens
mkdir -p .slurm-logs "$HERE/data"
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
uv run --frozen python "$HERE/scripts/data_tokens.py" "$@"
[[ "${1:-}" == "--smoke" ]] || uv run --frozen python "$HERE/scripts/build_page.py"
echo "done      : $(date)"
