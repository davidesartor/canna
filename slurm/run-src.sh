# Sourced by the lisa jobs that open an existing run: put the commit it trained at on PYTHONPATH,
# from a worktree outputs/src/<commit> made on first use, and default LISA_BF16 to the run's,
# so the XLA flags that follow match it. Needs LISA_RUN; from the repo root:
#     LISA_RUN=calm-otter-narrow-B; source slurm/run-src.sh
RUN_DIR="${LISA_OUTPUT_DIR:-outputs}/runs/${LISA_RUN:?set LISA_RUN=<run>}"
[[ -f "$RUN_DIR/config.yaml" ]] || { echo "[run] no $RUN_DIR/config.yaml" >&2; return 1; }
COMMIT=$(sed -n 's/^commit: *//p' "$RUN_DIR/config.yaml" | tr -d "'\"")
WORKTREE="$PWD/outputs/src/$COMMIT"
mkdir -p outputs/src
# one job makes the worktree; the others wait on the lock, then find it there
flock outputs/src/.lock bash -c \
    "[[ -d '$WORKTREE' ]] || git worktree add --detach '$WORKTREE' '$COMMIT'"
export PYTHONPATH="$WORKTREE/src"
export LISA_BF16="${LISA_BF16:-$(sed -n 's/^bf16: *//p' "$RUN_DIR/config.yaml" | tr -d "'\"")}"
echo "run       : $LISA_RUN at $COMMIT ($RUN_DIR)"
