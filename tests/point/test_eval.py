"""sample_posterior: RK4 transport of prior draws, and where a trained flow puts them."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
import pytest
import yaml

import canna.point as point
from canna.point.eval import sample_posterior
from canna.point.train import TrainState

CONFIG_ROOT = Path(point.__file__).parent / "configs"


def args(**overrides):
    with open(CONFIG_ROOT / "XS.yaml") as f:
        base = yaml.safe_load(f)
    base.update(
        dtype="float32",
        muon=False,
        learning_rate=1e-3,
        weight_decay=1e-5,
        network=dict(hidden_dim=64, num_blocks=2),
        config="XS",
        config_root=CONFIG_ROOT,
        output_dir=Path("outputs"),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def untrained():
    return TrainState.from_config(args(seed=0))


@pytest.fixture(scope="module")
def trained():
    state = TrainState.from_config(args(seed=0))
    state, _ = state.train_epoch(batch_size=128, n_steps=500)
    return state


def observation(problem, key):
    key_p, key_o = jr.split(key)
    truth = problem.sample_physical(key_p)
    return truth, problem.preprocess(problem.sample_observation(key_o, truth))


def test_returns_one_point_per_prior_draw(untrained):
    problem = untrained.problem
    truth, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 256))
    post = sample_posterior(problem, untrained.flow, u0, y)
    assert post.shape == (256, truth.shape[-1])
    assert jnp.all(jnp.isfinite(post))


@pytest.mark.parametrize("ode_steps", [1, 4, 8])
def test_any_ode_step_count_stays_finite(untrained, ode_steps):
    problem = untrained.problem
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 32))
    post = sample_posterior(problem, untrained.flow, u0, y, ode_steps)
    assert post.shape == u0.shape and jnp.all(jnp.isfinite(post))


def test_untrained_flow_is_the_identity_transport(untrained):
    """zero-init Modulation makes a fresh velocity field depend on nothing but x"""
    problem = untrained.problem
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 16))
    a = sample_posterior(problem, untrained.flow, u0, y)
    _, other_y = observation(problem, jr.key(11))
    b = sample_posterior(problem, untrained.flow, u0, other_y)
    assert jnp.allclose(a, b)


def test_is_deterministic_for_one_set_of_prior_draws(untrained):
    problem = untrained.problem
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 32))
    assert jnp.allclose(
        sample_posterior(problem, untrained.flow, u0, y),
        sample_posterior(problem, untrained.flow, u0, y),
    )


def test_a_restored_flow_reproduces_the_saved_posterior(tmp_path, trained):
    manager = ocp.CheckpointManager(tmp_path)
    trained.save_to(manager, epoch=1, loss_hist=jnp.zeros((1, 1)))
    manager.wait_until_finished()

    fresh = TrainState.from_config(args(seed=999))
    fresh, *_ = fresh.restore_from(ocp.CheckpointManager(tmp_path))

    problem = trained.problem
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 64))
    ref = sample_posterior(problem, trained.flow, u0, y)
    got = sample_posterior(problem, fresh.flow, u0, y)
    assert jnp.allclose(ref, got)


def test_a_trained_flow_concentrates_near_the_truth(trained):
    problem = trained.problem
    key_p, key_o, key_u = jr.split(jr.key(5), 3)
    truth = problem.sample_physical(key_p)
    y = problem.preprocess(problem.sample_observation(key_o, truth))
    u0 = jax.vmap(problem.sample_point)(jr.split(key_u, 2000))
    post = sample_posterior(problem, trained.flow, u0, y, 8)
    physical = jax.vmap(problem.flow_to_physical)(post)
    assert jnp.allclose(jnp.mean(physical, axis=0), truth, atol=0.3)
    assert jnp.all(jnp.std(physical, axis=0) < 0.8)
