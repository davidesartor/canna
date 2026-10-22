# F17 (2026-10-03): E3' relaunch hit the quota again; checkpoints drop the optimizer state
- After the cleanup, job 13645445 saved epoch 1 (`checkpoints/1`), then failed writing epoch
  2's `2.orbax-checkpoint-tmp` (`RESOURCE_EXHAUSTED`): the quota fits one 1.7 GB checkpoint,
  not the two orbax keeps during a save.
- **Speed:** ~2 min compile, then **60 s per 1000-step epoch** (epoch 1 at 2:05, epoch 2 at
  3:05), so 1M steps ≈ **17 h**, inside one 24 h job. The 57 ms/step estimate held.
- **Fix (2026-10-03):** `--save_opt_state/--no-save_opt_state`, false in `XS-late-768.yaml`.
  The optimizer state is 64% of a checkpoint (483 of 771 MB at XS), so a 768-wide one drops
  from ~1.7 to ~0.6 GB. `restore_from` keeps a fresh optimizer state when the checkpoint has
  none and says so in the log; `tests/lisa/test_checkpoint.py` covers both ways.
- **Cost:** a resume restarts muon's momentum and Adam's moments. That matters only at a 24 h
  boundary, which this run should not reach.
