"""point.train_sample: geodesic/jacobian construction, key provenance, batch."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.point import NoisyPoint, TrainSample, train_sample


@pytest.fixture
def problem():
    return NoisyPoint()


def flow_target(problem, key):
    """x1, the endpoint the geodesic walks to, from the first of the 4 sub-keys."""
    key_p, *_ = jr.split(key, 4)
    return problem.physical_to_flow(problem.sample_physical(key_p))


def test_train_sample_returns_namedtuple_fields(problem):
    sample = train_sample(problem, jr.key(0))
    assert isinstance(sample, TrainSample)
    assert sample._fields == ("xt", "dx", "t", "y")


def test_train_sample_t_in_unit_interval(problem):
    for i in range(20):
        sample = train_sample(problem, jr.key(i))
        assert sample.t.shape == ()
        assert 0.0 <= float(sample.t) <= 1.0


def test_train_sample_shapes_match_dims():
    problem = NoisyPoint(dim=3)
    sample = train_sample(problem, jr.key(0))
    assert sample.xt.shape == (3,)
    assert sample.dx.shape == (3,)
    assert sample.y.shape == (3,)


def test_train_sample_endpoint_consistent_with_geodesic_derivative(problem):
    """Euclidean geodesics are straight lines, so dx = x1-x0 is
    t-independent and x1 == xt + (1-t)*dx must hold exactly."""
    key = jr.key(7)
    sample = train_sample(problem, key)
    reconstructed_x1 = sample.xt + (1.0 - sample.t) * sample.dx
    assert jnp.allclose(reconstructed_x1, flow_target(problem, key), atol=1e-4)


def test_train_sample_dx_matches_jacobian_of_geodesic(problem):
    """dx must literally be d/dt geodesic(t, x0, x1)."""
    key = jr.key(11)
    sample = train_sample(problem, key)
    x0 = sample.xt - sample.t * sample.dx
    x1 = flow_target(problem, key)
    jac = jax.jacfwd(lambda t: problem.geodesic(t, x0, x1))(sample.t)
    assert jnp.allclose(jac, sample.dx, atol=1e-4)


def test_train_sample_zero_noise_y_equals_the_flow_target():
    """At noise_std=0, o == p deterministically and preprocess is physical_to_flow,
    so the conditioning view collapses exactly onto x1."""
    problem = NoisyPoint(noise_std=0.0)
    key = jr.key(3)
    sample = train_sample(problem, key)
    assert jnp.allclose(sample.y, flow_target(problem, key), atol=1e-6)


def test_train_sample_two_keys_give_different_draws(problem):
    s1 = train_sample(problem, jr.key(100))
    s2 = train_sample(problem, jr.key(101))
    assert not jnp.allclose(s1.xt, s2.xt, atol=1e-8)


def test_train_sample_same_key_is_deterministic(problem):
    s1 = train_sample(problem, jr.key(42))
    s2 = train_sample(problem, jr.key(42))
    for a, b in zip(s1, s2):
        assert jnp.allclose(a, b, atol=1e-8)


def test_train_sample_vmaps_over_batch_of_keys(problem):
    keys = jr.split(jr.key(0), 5)
    batched = jax.vmap(partial(train_sample, problem))(keys)
    assert batched.xt.shape == (5, 2)
    assert batched.dx.shape == (5, 2)
    assert batched.t.shape == (5,)
    assert batched.y.shape == (5, 2)


def test_train_sample_vmap_matches_individual_calls(problem):
    keys = jr.split(jr.key(9), 4)
    batched = jax.vmap(partial(train_sample, problem))(keys)
    for i, key in enumerate(keys):
        single = train_sample(problem, key)
        assert jnp.allclose(batched.xt[i], single.xt, atol=1e-6)
        assert jnp.allclose(batched.dx[i], single.dx, atol=1e-6)


def test_flow_base_point_is_sample_point_of_key_x0(problem):
    """x0 (the flow's base point) is problem.sample_point(key_x0),
    the third of the 4 sub-keys split from the training key -- not a fixed reference
    distribution. Reconstruct x0 algebraically from xt/t/dx and compare."""
    key = jr.key(56)
    *_, key_x0, key_t = jr.split(key, 4)
    x0 = problem.sample_point(key_x0)
    sample = train_sample(problem, key)
    reconstructed_x0 = sample.xt - sample.t * sample.dx
    assert jnp.allclose(reconstructed_x0, x0, atol=1e-4)


def test_sample_point_used_as_base_is_approximately_standard_normal():
    """sample_point whitens a fresh prior draw, so the base distribution in
    point-space is ~N(0,I) regardless of the prior's own cov."""
    problem = NoisyPoint(dim=2, seed=5)
    keys = jr.split(jr.key(0), 4000)
    draws = jax.vmap(problem.sample_point)(keys)
    assert jnp.allclose(jnp.mean(draws, axis=0), jnp.zeros(2), atol=0.1)
    assert jnp.allclose(jnp.var(draws, axis=0), jnp.ones(2), atol=0.15)


def test_y_is_a_noisy_view_of_the_flow_target():
    """y conditions on a noisy observation of the same p the geodesic ends at, so it
    differs from x1 by whitened zero-mean noise."""
    noise_std = 0.6
    problem = NoisyPoint(seed=0, dim=2, noise_std=noise_std)
    keys = jr.split(jr.key(2), 4000)
    samples = jax.vmap(partial(train_sample, problem))(keys)
    targets = jax.vmap(lambda k: flow_target(problem, k))(keys)
    residual = samples.y - targets
    expected_cov = noise_std**2 * problem.whitening @ problem.whitening.T
    assert jnp.allclose(jnp.cov(residual, rowvar=False), expected_cov, atol=0.05)
    assert jnp.allclose(jnp.mean(residual, axis=0), jnp.zeros(2), atol=0.05)


def test_y_matches_second_subkey_key_o(problem):
    key = jr.key(77)
    key_p, key_o, *_ = jr.split(key, 4)
    p = problem.sample_physical(key_p)
    expected_y = problem.preprocess(problem.sample_observation(key_o, p))
    assert jnp.allclose(train_sample(problem, key).y, expected_y, atol=1e-6)


def test_t_matches_fourth_subkey_key_t(problem):
    key = jr.key(13)
    *_, key_t = jr.split(key, 4)
    assert jnp.allclose(train_sample(problem, key).t, jr.uniform(key_t, ()), atol=1e-8)


def test_train_sample_is_jit_compatible(problem):
    key = jr.key(13)
    jitted = eqx.filter_jit(partial(train_sample, problem))
    for a, b in zip(train_sample(problem, key), jitted(key)):
        assert jnp.allclose(a, b, atol=1e-6)


def test_train_sample_requests_exactly_one_noisy_observation():
    """y comes from a single sample_observation call."""
    n_calls = []

    class RecordingPoint(NoisyPoint):
        def sample_observation(self, key, p):
            n_calls.append(key)
            return super().sample_observation(key, p)

    train_sample(RecordingPoint(), jr.key(0))
    assert len(n_calls) == 1


def test_train_sample_fields_are_float64(problem):
    """canna/__init__.py enables x64; every TrainSample field should be float64."""
    sample = train_sample(problem, jr.key(0))
    for field in sample:
        assert field.dtype == jnp.float64


def test_t_is_sampled_half_open_zero_one(problem):
    """t = jr.uniform(key, ()) draws from the half-open [0, 1), so t
    should never realize exactly 1.0 across many draws."""
    ts = jnp.array([train_sample(problem, jr.key(i)).t for i in range(500)])
    assert jnp.all(ts < 1.0)
    assert jnp.all(ts >= 0.0)
