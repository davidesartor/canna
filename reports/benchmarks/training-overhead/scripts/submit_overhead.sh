#!/bin/bash
# Submit overhead.sbatch, one job per chip: reports/benchmarks/training-overhead/scripts/submit_overhead.sh [gres types...] from the repo root after `source slurm/<cluster>.env`; LISA_BF16=<mode> tags the jobs and data files -bf16<mode>
set -euo pipefail
[[ -n "${CANNA_SBATCH_GPU:-}" ]] || {
    echo "CANNA_SBATCH_GPU unset: source slurm/unity.env or slurm/trex.env first" >&2; exit 1; }
CHIPS=("$@")
((${#CHIPS[@]})) || CHIPS=(l40s a100)
# freeze src/ so every job and chunk measures the network as it is now; pass BENCH_SRC to reuse a copy
SRC="${BENCH_SRC:-outputs/bench-src-$(date +%Y%m%d-%H%M%S)}"
if [[ ! -d "$SRC" ]]; then
    mkdir -p "$SRC"
    cp -r src/canna "$SRC/"
    find "$SRC" -name __pycache__ -prune -exec rm -rf {} +
    git rev-parse HEAD >"$SRC/HEAD"
    git diff HEAD -- src >"$SRC/src.diff"
fi
echo "frozen src: $SRC"
for CHIP in "${CHIPS[@]}"; do
    TAG="${CHIP//_/}"
    [[ "${LISA_BF16:-none}" == none ]] || TAG+="-bf16$LISA_BF16"
    [[ ! -e "reports/benchmarks/training-overhead/data/$TAG.jsonl" ]] || {
        echo "data/$TAG.jsonl exists: move it aside first" >&2; exit 1; }
    sbatch --export=ALL,BENCH_SRC="$SRC" $CANNA_SBATCH_GPU --gres="gpu:$CHIP:1" \
        --job-name="canna-lisa-overhead-$TAG" \
        reports/benchmarks/training-overhead/scripts/overhead.sbatch "$TAG"
done
