#!/bin/bash -l
# Build reports/diagnostics/source-viewer/source-viewer.html on CPU (build_source_viewer.py beside this
# file): the narrow-B and full-B problems and flows, each band's bank of prior windows, the JavaScript port
# checked against canna under node, then the page; stray LISA_* variables are cleared, so the
# bands are the stock configs.
# Submit from the repo root; on Unity, after `source slurm/unity.env`:
#     sbatch $CANNA_SBATCH_CPU --constraint=avx512 reports/diagnostics/source-viewer/scripts/source-viewer.sbatch
# Flags go to build_source_viewer.py (--no-check, --out page.html).

#SBATCH --job-name=canna-lisa-source-viewer
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export JAX_PLATFORMS=cpu

# Unity's node module lives on the uri/main branch, which loads on compute nodes only; the login
# shell (bash -l) is what defines `module`
if ! command -v node >/dev/null && type module >/dev/null 2>&1; then
    set +u
    module load uri/main nodejs/20.13.1-GCCcore-13.3.0
    set -u
fi
command -v node >/dev/null || { echo "node not found: the port check needs it" >&2; exit 1; }

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
echo "node      : $(node --version)"

uv run --frozen python reports/diagnostics/source-viewer/scripts/build_source_viewer.py "$@"
