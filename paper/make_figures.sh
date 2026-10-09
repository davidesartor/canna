#!/usr/bin/env bash
# Rebuild everything the paper reads that is not committed: figures/*.pdf (and the tables,
# which are committed but regenerated from the same data). Run after every pull.
#
#   ./make_figures.sh              all figures and tables
#   ./make_figures.sh --pdf        ... and then compile main.pdf with latexmk
#   ./make_figures.sh fig_band     only the named scripts (fig_*.py, tables.py)
#
# Python: $PYTHON if set, else the repository's existing .venv (left untouched), else
# `uv run --frozen`, which creates the environment from uv.lock on first use, else python3.
# Everything runs on the CPU and takes a few minutes. Needs pdflatex for the network
# diagram and latexmk for --pdf.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$PAPER")"

if [[ -n "${PYTHON:-}" ]]; then
    read -r -a PY <<< "$PYTHON"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
    PY=("$REPO/.venv/bin/python")
elif command -v uv > /dev/null; then
    PY=(uv run --frozen --project "$REPO" python)
else
    PY=(python3)
fi
echo "python: ${PY[*]}"

export JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONWARNINGS=ignore

BUILD_PDF=0
SCRIPTS=()
for arg in "$@"; do
    case "$arg" in
        --pdf) BUILD_PDF=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) SCRIPTS+=("${arg%.py}.py") ;;
    esac
done
if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    SCRIPTS=($(cd "$PAPER/scripts" && ls fig_*.py) tables.py)
fi

mkdir -p "$PAPER/figures"
cd "$PAPER/scripts"
for script in "${SCRIPTS[@]}"; do
    start=$SECONDS
    printf '%-22s' "$script"
    "${PY[@]}" "$script" > "/tmp/canna-paper-$script.log" 2>&1 || {
        echo "FAILED, log follows"; cat "/tmp/canna-paper-$script.log"; exit 1; }
    echo "ok ($((SECONDS - start)) s)"
done

# the network diagram is TikZ, compiled on its own
if command -v pdflatex > /dev/null; then
    printf '%-22s' "architecture.tex"
    (cd "$PAPER/figures/src" \
        && pdflatex -interaction=nonstopmode -halt-on-error architecture.tex > /dev/null \
        && mv architecture.pdf ../architecture.pdf \
        && rm -f architecture.aux architecture.log)
    echo "ok"
else
    echo "pdflatex not found: skipping figures/architecture.pdf" >&2
fi

if [[ $BUILD_PDF -eq 1 ]]; then
    (cd "$PAPER" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1) \
        && echo "main.pdf built" || { echo "latexmk failed: see $PAPER/main.log"; exit 1; }
fi
