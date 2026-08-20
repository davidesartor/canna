"""The point trainer: config parsing, and the CLI's checkpoint/resume round trip."""

import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import yaml

import canna.point as point
from canna.point import NoisyPoint, PointFlow

CONFIG_ROOT = Path(point.__file__).parent / "configs"


def config():
    with open(CONFIG_ROOT / "XS.yaml") as f:
        return yaml.safe_load(f)


def run_train(out_dir, **flags):
    """Run the trainer CLI on the XS config and return its stdout."""
    cmd = [
        sys.executable,
        "-m",
        "canna.point.train",
        "--config",
        "XS",
        "--config_root",
        str(CONFIG_ROOT),
        "--output_dir",
        str(out_dir),
        "--dtype",
        "float32",
        "--no-muon",
    ]
    for name, value in flags.items():
        cmd += [f"--{name}", str(value)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def skeleton(seed=0):
    """The (problem, flow) pair train.py builds before it restores anything."""
    key_sample, key_network, _ = jr.split(jr.key(seed), 3)
    problem = NoisyPoint(**config()["problem"])
    p = problem.sample_physical(key_sample)
    flow = PointFlow(
        **config()["network"],
        x_shape=problem.physical_to_flow(p).shape,
        y_shape=problem.preprocess(problem.sample_observation(key_sample, p)).shape,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        key=key_network,
    )
    return problem, flow


def param_leaves(module):
    return jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array))


# --- the CLI, end to end ---------------------------------------------------


def test_a_cli_flag_overrides_the_run_config(tmp_path):
    """XS.yaml asks for 1000 steps in 100-step epochs; the flags must win"""
    out = run_train(tmp_path, total_steps=20, log_interval=10)
    assert "[epoch 2/2]" in out and "/10]" not in out


def test_a_run_writes_a_checkpoint_and_a_loss_curve(tmp_path):
    out = run_train(tmp_path, total_steps=20, log_interval=10)
    assert (tmp_path / "point-XS" / "checkpoint.eqx").exists()
    assert (tmp_path / "point-XS" / "losses.pdf").exists()
    assert "[epoch 2/2]" in out


def test_the_checkpoint_holds_trained_params_not_the_fresh_ones(tmp_path):
    run_train(tmp_path, total_steps=20, log_interval=10)
    _, flow = skeleton()
    restored = eqx.tree_deserialise_leaves(
        tmp_path / "point-XS" / "checkpoint.eqx", flow
    )
    assert any(
        not jnp.allclose(a, b)
        for a, b in zip(param_leaves(flow), param_leaves(restored))
    )


def test_rerunning_resumes_instead_of_retraining(tmp_path):
    """the finished epochs are the finite rows of the saved loss history"""
    run_train(tmp_path, total_steps=20, log_interval=10)
    _, flow = skeleton()
    after_first = eqx.tree_deserialise_leaves(
        tmp_path / "point-XS" / "checkpoint.eqx", flow
    )

    out = run_train(tmp_path, total_steps=20, log_interval=10)
    assert "[checkpoint] resuming at epoch 2" in out
    assert "[epoch " not in out

    after_second = eqx.tree_deserialise_leaves(
        tmp_path / "point-XS" / "checkpoint.eqx", flow
    )
    assert all(
        jnp.allclose(a, b)
        for a, b in zip(param_leaves(after_first), param_leaves(after_second))
    )


def test_a_fresh_output_dir_starts_at_epoch_zero(tmp_path):
    out = run_train(tmp_path, total_steps=10, log_interval=10)
    assert "resuming" not in out
    assert "[epoch 1/1]" in out


def test_the_seed_reaches_the_network_init(tmp_path):
    a, b = skeleton(seed=1), skeleton(seed=2)
    assert any(
        not jnp.allclose(u, v) for u, v in zip(param_leaves(a[1]), param_leaves(b[1]))
    )
