"""sample_posterior: RK4 transport of prior draws, and where a trained flow puts them."""

import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

import canna.point as point
from canna.point import NoisyPoint, PointFlow
from canna.point.eval import sample_posterior

CONFIG_ROOT = Path(point.__file__).parent / "configs"


def config():
    with open(CONFIG_ROOT / "XS.yaml") as f:
        return yaml.safe_load(f)


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


@pytest.fixture(scope="module")
def untrained():
    return skeleton()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train through the real CLI, then read the flow back out of its checkpoint."""
    out_dir = tmp_path_factory.mktemp("trained")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "canna.point.train",
            *("--config", "XS"),
            *("--config_root", str(CONFIG_ROOT)),
            *("--output_dir", str(out_dir)),
            *("--dtype", "float32"),
            *("--total_steps", "2000"),
            *("--log_interval", "500"),
            "--no-muon",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr[-2000:]

    problem, flow = skeleton()
    return problem, eqx.tree_deserialise_leaves(
        out_dir / "point-XS" / "checkpoint.eqx", flow
    )


def observation(problem, key):
    key_p, key_o = jr.split(key)
    truth = problem.sample_physical(key_p)
    return truth, problem.preprocess(problem.sample_observation(key_o, truth))


def test_returns_one_point_per_prior_draw(untrained):
    problem, flow = untrained
    truth, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_flow)(jr.split(jr.key(3), 256))
    post = sample_posterior(problem, flow, u0, y)
    assert post.shape == (256, truth.shape[-1])
    assert jnp.all(jnp.isfinite(post))


@pytest.mark.parametrize("ode_steps", [1, 4, 8])
def test_any_ode_step_count_stays_finite(untrained, ode_steps):
    problem, flow = untrained
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_flow)(jr.split(jr.key(3), 32))
    post = sample_posterior(problem, flow, u0, y, ode_steps)
    assert post.shape == u0.shape and jnp.all(jnp.isfinite(post))


def test_untrained_flow_is_the_identity_transport(untrained):
    """zero-init Modulation makes a fresh velocity field depend on nothing but x"""
    problem, flow = untrained
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_flow)(jr.split(jr.key(3), 16))
    a = sample_posterior(problem, flow, u0, y)
    _, other_y = observation(problem, jr.key(11))
    b = sample_posterior(problem, flow, u0, other_y)
    assert jnp.allclose(a, b)


def test_is_deterministic_for_one_set_of_prior_draws(untrained):
    problem, flow = untrained
    _, y = observation(problem, jr.key(1))
    u0 = jax.vmap(problem.sample_flow)(jr.split(jr.key(3), 32))
    assert jnp.allclose(
        sample_posterior(problem, flow, u0, y),
        sample_posterior(problem, flow, u0, y),
    )


def test_the_restored_flow_is_not_the_fresh_one(trained, untrained):
    """the checkpoint round trip must actually carry the training into eval"""
    _, restored = trained
    _, fresh = untrained

    def leaves(module):
        return jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array))

    assert any(not jnp.allclose(a, b) for a, b in zip(leaves(fresh), leaves(restored)))


def test_a_trained_flow_approaches_the_analytic_gaussian_posterior(trained):
    """Gaussian prior + Gaussian noise, so the exact posterior is in closed form."""
    problem, flow = trained
    key_p, key_o, key_u = jr.split(jr.key(5), 3)
    truth = problem.sample_physical(key_p)
    o = problem.sample_observation(key_o, truth)
    y = problem.preprocess(o)
    u0 = jax.vmap(problem.sample_flow)(jr.split(key_u, 2000))
    post = sample_posterior(problem, flow, u0, y, 8)
    physical = jax.vmap(problem.flow_to_physical)(post)

    gain = problem.cov @ jnp.linalg.inv(problem.cov + problem.noise_cov)
    exact_mean = gain @ o
    exact_std = jnp.sqrt(jnp.diag(problem.cov - gain @ problem.cov))
    assert jnp.allclose(jnp.mean(physical, axis=0), exact_mean, atol=0.3)
    assert jnp.allclose(jnp.std(physical, axis=0), exact_std, rtol=0.4)
