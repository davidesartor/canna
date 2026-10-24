#!/bin/bash
# Velocity probe of a job still training, on that job's own GPU: snapshot the run's checkpoint
# (retrying if a newer save deletes it mid-copy) and run velocity_probe.py beside this file as an
# overlapping step of the job. The record goes to the report's data/probe/<tag>.json; the snapshot
# to outputs/probes/, which a later live probe of the same run replaces. It runs on the commit
# the run trained at (slurm/run-src.sh). From the repo root:
#     reports/posterior-eval/flow-vs-reference/scripts/live.sh calm-otter-full-B 14891481 full-B-epoch79
set -euo pipefail
unset $(compgen -e LISA_) || true
export LISA_RUN=$1; JOB=$2; TAG=$3
source slurm/run-src.sh
HERE=reports/posterior-eval/flow-vs-reference
SNAPSHOT=outputs/probes/$LISA_RUN
SRC=$RUN_DIR/checkpoints
rm -rf "$SNAPSHOT"; mkdir -p "$SNAPSHOT/runs/$LISA_RUN/checkpoints" "$HERE/data/probe"
cp "$RUN_DIR/config.yaml" "$SNAPSHOT/runs/$LISA_RUN/"
for try in 1 2 3 4 5; do
    STEP=$(ls "$SRC" | grep -E '^[0-9]+$' | sort -n | tail -1)
    if cp -r "$SRC/$STEP" "$SNAPSHOT/runs/$LISA_RUN/checkpoints/" && [[ -d "$SRC/$STEP" ]]; then break; fi
    rm -rf "$SNAPSHOT/runs/$LISA_RUN/checkpoints"/*; sleep 2
done
echo "snapshot $SRC -> $SNAPSHOT"
# the run's XLA flags (slurm/gpu.sh reads its LISA_BF16)
source slurm/gpu.sh
srun --jobid="$JOB" --overlap --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
    env LISA_OUTPUT_DIR="$SNAPSHOT" \
    uv run --extra cuda --frozen python "$HERE/scripts/velocity_probe.py" "$HERE/data/probe/$TAG.json"
