#!/bin/bash
#SBATCH --job-name=canna_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --partition=gpu,gpu-preempt

# ---------------------------------------------------------------------------
# Train the flow-matching posterior p(x|y). Training only -- no corner-plot
# eval; run slurm/eval.slurm separately against a checkpoint for that.
# SIMPLIFIED_PROBLEM=1 = reduced month/4-param problem; unset/0 = year/8-param.
# N_SOURCES (separate axis, default 1) sets the GB source count in [1,4].
# These two import-time toggles are the only env knobs; all other training
# settings (run length, batch size, LR, ...) are hard-coded in scripts/train.py.
#
# The run outlives the walltime by design: this job requeues itself under the
# same job id and train.py resumes from its checkpoint, both when it crashes and
# when it nears the wall -- slurm raises SIGUSR1 with 180s to go and train.py
# checkpoints and stops. It keeps coming back until train.py runs out of steps.
# ---------------------------------------------------------------------------

set -uo pipefail   # not -e: tolerate Python's nonzero teardown exit (120)

export SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM:-0}
export N_SOURCES=${N_SOURCES:-1}

# dying faster than this is a deterministic bug, not a fluke; 0 forces requeue (tests)
MIN_RUNTIME_TO_REQUEUE=${MIN_RUNTIME_TO_REQUEUE:-600}

requeue_wanted=0

requeue_and_wait_to_be_killed() {
    echo "requeueing $SLURM_JOB_ID to resume from checkpoint"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60   # slurmctld terminates us here; don't race it to the exit
}

on_walltime() {
    requeue_wanted=1
    echo "[walltime] wall is near -- asking train.py to checkpoint and stop"
    kill -USR1 "$train_pid" 2>/dev/null
}

echo "============================================"
echo "Job ID           : $SLURM_JOB_ID"
echo "Attempt          : $(( ${SLURM_RESTART_COUNT:-0} + 1 ))"
echo "Node             : $SLURM_NODELIST"
echo "Start time       : $(date)"
echo "SIMPLIFIED_PROBLEM: ${SIMPLIFIED_PROBLEM}"
echo "N_SOURCES        : ${N_SOURCES}"
echo "============================================"

echo "Launching scripts/train.py SIMPLIFIED_PROBLEM=${SIMPLIFIED_PROBLEM} N_SOURCES=${N_SOURCES} ..."
started_at=$SECONDS
# armed before there is anything to kill: an untrapped USR1 would take the shell down with it
train_pid=
trap on_walltime USR1

# backgrounded and waited on, or bash would defer the USR1 trap until training returns
uv run python scripts/train.py &
train_pid=$!
wait "$train_pid"
rc=$?

if [ "$requeue_wanted" -eq 1 ]; then
    wait "$train_pid" 2>/dev/null   # the trap cut the wait above short; let train.py finish saving
    echo "train.py checkpointed and stopped after $(( SECONDS - started_at ))s"
    requeue_and_wait_to_be_killed
    exit 0
fi

ran_for=$(( SECONDS - started_at ))
echo "python exit code: $rc (after ${ran_for}s)"

[ "$rc" -eq 120 ] && rc=0

if [ "$rc" -ne 0 ]; then
    if [ "$ran_for" -lt "$MIN_RUNTIME_TO_REQUEUE" ]; then
        echo "died after only ${ran_for}s -- not requeueing, fix it first"
    else
        requeue_and_wait_to_be_killed
    fi
fi

echo "============================================"
echo "Job finished : $(date)"
echo "============================================"
exit $rc
