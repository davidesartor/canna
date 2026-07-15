"""Crash resume: a run that dies and restores must match one that never died.

train.slurm requeues the job on a crash and train.py picks the checkpoint back up, so
the resumed weights have to agree with an uninterrupted run exactly -- which only holds
if the optimizer moments, step count, and target stats all come back, not just weights.
"""

import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flax import nnx, serialization

from canna import lisa, networks

from _bench import CONFIG_BY_NAME, build_model, wdm_shape

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
import train

CONFIG = CONFIG_BY_NAME["S"]  # smallest of the ladder; resume is width-independent
BATCH = 2
STEPS_BEFORE_CRASH = 3
STEPS_AFTER_CRASH = 3
DTYPE = jnp.float32

RunState = tuple[networks.MMDiT, nnx.Optimizer, train.TargetStats]


def build_run_state(seed: int = 0) -> RunState:
    """Flow, optimizer, and stats wired exactly as train.setup does, at the small config."""
    flow = build_model(CONFIG, dtype=DTYPE, seed=seed)
    optimizer = nnx.Optimizer(
        model=flow,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(train.LEARNING_RATE, weight_decay=train.WEIGHT_DECAY),
        ),
        wrt=nnx.Param,
    )
    return flow, optimizer, train.TargetStats()


def make_batch(step: int) -> train.TrainBatch:
    """A batch fixed by step index, so both runs see identical data."""
    keys = jr.split(jr.fold_in(jr.key(0), step), 6)
    u_shape = (BATCH, lisa.N_SOURCES, len(lisa.PARAMETER_NAMES))
    return train.TrainBatch(
        ut=jr.uniform(keys[0], u_shape, DTYPE),
        du=jr.normal(keys[1], u_shape, DTYPE),
        t=jr.uniform(keys[2], (BATCH,), DTYPE),
        y=jr.normal(keys[3], (BATCH, *wdm_shape()), DTYPE),
        u1=jr.uniform(keys[4], u_shape, DTYPE),
        y_clean=jr.normal(keys[5], (BATCH, *wdm_shape()), DTYPE),
    )


def run_steps(state: RunState, steps):
    flow, optimizer, stats = state
    for step in steps:
        train.train_step(
            flow,
            optimizer,
            stats,
            make_batch(step),
            train.aux_loss_weight_schedule(step),
        )


def params(state: RunState) -> dict:
    return nnx.to_pure_dict(nnx.state(state[0], nnx.Param))


def max_abs_diff(a: dict, b: dict) -> float:
    per_leaf = jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b)
    return float(jnp.max(jnp.stack(jax.tree.leaves(per_leaf))))


def test_resume_matches_uninterrupted(tmp_path):
    total = STEPS_BEFORE_CRASH + STEPS_AFTER_CRASH
    ckpt_path = str(tmp_path / "ckpt.msgpack")

    reference = build_run_state()
    run_steps(reference, range(total))

    # the run that dies mid-way and comes back off disk into freshly built state
    crashed = build_run_state()
    run_steps(crashed, range(STEPS_BEFORE_CRASH))
    train.save_checkpoint(*crashed, ckpt_path)

    resumed = build_run_state()
    start_step = train.restore_checkpoint(*resumed, ckpt_path)
    assert start_step == STEPS_BEFORE_CRASH, "step count did not survive the round trip"
    run_steps(resumed, range(start_step, total))

    # bit-identical, or something in the optimizer/stats state was silently lost
    err = max_abs_diff(params(reference), params(resumed))
    assert (
        err == 0.0
    ), f"resumed run diverged from uninterrupted run: max |a - b| = {err:.3e}"


def test_target_stats_survive_the_round_trip(tmp_path):
    """Welford stats normalize the loss, so losing them would quietly change training."""
    ckpt_path = str(tmp_path / "ckpt.msgpack")
    crashed = build_run_state()
    run_steps(crashed, range(STEPS_BEFORE_CRASH))
    train.save_checkpoint(*crashed, ckpt_path)

    resumed = build_run_state()
    train.restore_checkpoint(*resumed, ckpt_path)

    before, after = crashed[2].variance(), resumed[2].variance()
    assert float(jnp.max(jnp.abs(before - after))) == 0.0
    assert float(jnp.min(before)) > 0.0, "stats never accumulated; test proves nothing"


def test_restore_is_not_a_silent_no_op(tmp_path):
    """Guards the assertions above: untrained state must NOT already match a trained run."""
    trained = build_run_state()
    run_steps(trained, range(STEPS_BEFORE_CRASH))
    assert max_abs_diff(params(trained), params(build_run_state())) > 0.0


def test_setup_returns_zero_when_nothing_to_resume(tmp_path):
    *_, step = train.setup(str(tmp_path / "absent.msgpack"))
    assert step == 0


def test_legacy_checkpoint_is_rejected(tmp_path):
    """Pre-resume checkpoints hold bare weights; restoring one must raise, not resume at 0."""
    ckpt_path = str(tmp_path / "legacy.msgpack")
    flow, _, _ = build_run_state()
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(nnx.to_pure_dict(nnx.state(flow))))

    with pytest.raises(ValueError, match="predates the current checkpoint format"):
        train.restore_checkpoint(*build_run_state(), ckpt_path)


def test_save_leaves_no_partial_file(tmp_path):
    """The write goes via rename, so a reader never sees a half-written checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.msgpack")
    train.save_checkpoint(*build_run_state(), ckpt_path)

    assert os.path.exists(ckpt_path)
    assert os.listdir(tmp_path) == ["ckpt.msgpack"], "temporary file left behind"
