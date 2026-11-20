#!/usr/bin/env bash
# Rebuild figures/*.pdf and numbers/*.tex from the fig_*/numbers_* scripts here.
#   scripts/make_figures.sh [--pdf] [fig_priors ...]    --pdf also compiles build/main.pdf
# The scripts import JAX on the CPU: run it in a job (make_figures.sbatch), not on a login node.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# the repo's .venv as it is: a CPU `uv run` would sync away its cuda extra
PY=../../.venv/bin/python
[[ -x $PY ]] || PY="uv run --frozen python"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONWARNINGS=ignore
# a fixed timestamp keeps the pdfs byte-identical unless a script or its inputs changed
export SOURCE_DATE_EPOCH=0

pdf=0
scripts=()
for arg in "$@"; do
    if [[ $arg == --pdf ]]; then pdf=1; else scripts+=("${arg%.py}.py"); fi
done
((${#scripts[@]})) || scripts=(fig_*.py numbers_*.py)

failed=()
for script in "${scripts[@]}"; do
    printf '%-26s' "$script"
    if log=$($PY "$script" 2>&1); then echo ok; else echo FAILED; echo "$log"; failed+=("$script"); fi
done

# the TikZ diagrams, only when a source changed (xelatex output is not byte-stable)
for src in ../figures/src/*.tex; do
    name=$(basename "$src" .tex)
    if [[ $src -nt ../figures/$name.pdf ]]; then
        (cd ../figures/src && xelatex -interaction=nonstopmode -halt-on-error "$name.tex" > /dev/null \
            && mv "$name.pdf" .. && rm -f "$name.aux" "$name.log")
        printf '%-26sok\n' "$name.tex"
    fi
done

if ((pdf)); then
    (cd .. && SOURCE_DATE_EPOCH= latexmk -interaction=nonstopmode -halt-on-error main.tex > /dev/null) \
        || { echo "latexmk failed: see paper/build/main.log" >&2; exit 1; }
    echo "build/main.pdf built"
fi

((${#failed[@]} == 0)) || { echo "failed: ${failed[*]}" >&2; exit 1; }
