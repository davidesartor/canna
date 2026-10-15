#!/bin/bash
# Submit the narrowband MCMC of a config's eval sources (slurm/lisa-mcmc.sbatch) unless its
# chains are cached or already queued. Called by the training jobs at their start, so the
# chains are ready long before the eval draws them; safe to call from several jobs of the same
# problem at once (a lock folder names the submitted job). Never fails its caller.
#     slurm/mcmc-ensure.sh XS-fd
CONFIG="$1"
PY=.venv313/bin/python
[[ -x "$PY" ]] || { echo "[mcmc] no .venv313: chains not submitted"; exit 0; }
export JAX_PLATFORMS=cpu
if "$PY" -m canna.lisa.mcmc status --config "$CONFIG" >/dev/null 2>&1; then
    echo "[mcmc] chains of $CONFIG cached"; exit 0
fi
DIR=$("$PY" -m canna.lisa.mcmc status --config "$CONFIG" --dir 2>/dev/null)
N=$("$PY" -m canna.lisa.mcmc status --config "$CONFIG" --count 2>/dev/null)
mkdir -p "$DIR"
# mkdir is atomic: of several jobs starting together, one takes the lock and submits
LOCK="$DIR/.submitted"
if ! mkdir "$LOCK" 2>/dev/null; then
    sleep 30  # let the holder write its job id
    if squeue -h -j "$(cat "$LOCK/job" 2>/dev/null)" 2>/dev/null | grep -q .; then
        echo "[mcmc] chains of $CONFIG already queued ($(cat "$LOCK/job"))"; exit 0
    fi
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0  # a stale lock: its job is gone
fi
JOB=$(sbatch --parsable --array=0-$((N - 1)) slurm/lisa-mcmc.sbatch "$CONFIG") || {
    echo "[mcmc] sbatch failed"; rm -rf "$LOCK"; exit 0; }
echo "$JOB" > "$LOCK/job"
echo "[mcmc] submitted $JOB: $N chains of $CONFIG -> $DIR"
