"""Problem.train_sample: geodesic/jacobian construction, x_target/y_target provenance, batch."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.problems import NoisyPoint, TrainSample


@pytest.fixture
def problem():
    return NoisyPoint()


def test_train_sample_returns_namedtuple_fields(problem):
    sample = problem.train_sample(jr.key(0))
    assert isinstance(sample, TrainSample)
    assert sample._fields == ("xt", "dx", "t", "y", "x_target", "y_target")


def test_train_sample_t_in_unit_interval(problem):
    for i in range(20):
        sample = problem.train_sample(jr.key(i))
        assert sample.t.shape == ()
        assert 0.0 <= float(sample.t) <= 1.0


def test_train_sample_shapes_match_dims():
    problem = NoisyPoint(dim=3)
    sample = problem.train_sample(jr.key(0))
    assert sample.xt.shape == (3,)
    assert sample.dx.shape == (3,)
    assert sample.x_target.shape == (3,)
    assert sample.y.shape == (3,)
    assert sample.y_target.shape == (3,)


def test_train_sample_endpoint_consistent_with_geodesic_derivative(problem):
    """Euclidean geodesics are straight lines, so dx = x1-x0 is
    t-independent and x_target == xt + (1-t)*dx must hold exactly."""
    sample = problem.train_sample(jr.key(7))
    reconstructed_x1 = sample.xt + (1.0 - sample.t) * sample.dx
    assert jnp.allclose(reconstructed_x1, sample.x_target, atol=1e-4)


def test_train_sample_dx_matches_jacobian_of_geodesic(problem):
    """dx must literally be d/dt geodesic(t, x0, x1)."""
    sample = problem.train_sample(jr.key(11))
    x0 = sample.xt - sample.t * sample.dx
    x1 = sample.x_target
    geo = problem.geometry
    jac = jax.jacfwd(lambda t: geo.geodesic(t, x0, x1))(sample.t)
    assert jnp.allclose(jac, sample.dx, atol=1e-4)


def test_train_sample_zero_noise_y_equals_y_target():
    """With noise_std=0, sample_observation is deterministic regardless of
    the noise key, so y and y_target (built from the same p) must coincide."""
    problem = NoisyPoint(noise_std=0.0)
    sample = problem.train_sample(jr.key(3))
    assert jnp.allclose(sample.y, sample.y_target, atol=1e-6)


def test_train_sample_x_target_is_valid_manifold_point(problem):
    sample = problem.train_sample(jr.key(4))
    p = problem.chart.backward(sample.x_target)
    assert jnp.allclose(problem.chart.forward(p), sample.x_target, atol=1e-4)


def test_train_sample_two_keys_give_different_draws(problem):
    s1 = problem.train_sample(jr.key(100))
    s2 = problem.train_sample(jr.key(101))
    assert not jnp.allclose(s1.xt, s2.xt, atol=1e-8)


def test_train_sample_same_key_is_deterministic(problem):
    s1 = problem.train_sample(jr.key(42))
    s2 = problem.train_sample(jr.key(42))
    for a, b in zip(s1, s2):
        assert jnp.allclose(a, b, atol=1e-8)


def test_train_sample_vmaps_over_batch_of_keys(problem):
    keys = jr.split(jr.key(0), 5)
    batched = jax.vmap(problem.train_sample)(keys)
    assert batched.xt.shape == (5, 2)
    assert batched.dx.shape == (5, 2)
    assert batched.t.shape == (5,)
    assert batched.y.shape == (5, 2)
    assert batched.x_target.shape == (5, 2)
    assert batched.y_target.shape == (5, 2)


def test_train_sample_vmap_matches_individual_calls(problem):
    keys = jr.split(jr.key(9), 4)
    batched = jax.vmap(problem.train_sample)(keys)
    for i, key in enumerate(keys):
        single = problem.train_sample(key)
        assert jnp.allclose(batched.xt[i], single.xt, atol=1e-6)
        assert jnp.allclose(batched.dx[i], single.dx, atol=1e-6)
        assert jnp.allclose(batched.x_target[i], single.x_target, atol=1e-6)


def test_x_target_is_chart_forward_of_key_p_split(problem):
    """train_sample splits key into 5 (key_p, key_o, key_o_target, key_x0, key_t);
    x_target must be chart.forward(sample_physical(key_p)), the first sub-key."""
    key = jr.key(55)
    key_p, _, _, _, _ = jr.split(key, 5)
    p = problem.sample_physical(key_p)
    sample = problem.train_sample(key)
    assert jnp.allclose(sample.x_target, problem.chart.forward(p), atol=1e-6)


def test_flow_base_point_is_sample_point_of_key_x0(problem):
    """x0 (the flow's base point) is problem.sample_point(key_x0),
    the fourth of the 5 sub-keys split from the training key -- not a fixed reference
    distribution. Reconstruct x0 algebraically from xt/t/dx and compare."""
    key = jr.key(56)
    _, _, _, key_x0, _ = jr.split(key, 5)
    x0 = problem.sample_point(key_x0)
    sample = problem.train_sample(key)
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


def test_y_conditions_on_noise_while_y_target_is_clean():
    """train_sample draws y from a noisy observation and y_target from a clean one of
    the same p, so the two differ and their residuals about x_target are uncorrelated.
    """
    problem = NoisyPoint(noise_std=1.0)
    keys = jr.split(jr.key(0), 3000)
    samples = jax.vmap(problem.train_sample)(keys)
    residual_y = samples.y - samples.x_target
    residual_y_target = samples.y_target - samples.x_target
    cross = jnp.mean(residual_y * residual_y_target, axis=0)
    assert jnp.all(jnp.abs(cross) < 0.1)
    assert not jnp.allclose(samples.y[0], samples.y_target[0], atol=1e-3)


def test_train_sample_requests_one_noisy_then_one_clean_observation():
    """y comes from a clean=False call, y_target from a clean=True one, in that order."""
    requested_clean_flags = []

    class RecordingPoint(NoisyPoint):
        def sample_observation(self, key, p, clean=False):
            requested_clean_flags.append(clean)
            return super().sample_observation(key, p, clean)

    RecordingPoint().train_sample(jr.key(0))
    assert requested_clean_flags == [False, True]


def test_zero_noise_collapses_y_and_y_target_to_x_target():
    """At noise_std=0 both conditioning views collapse exactly onto x_target,
    since preprocess is chart.forward and o == p deterministically."""
    problem = NoisyPoint(noise_std=0.0)
    sample = problem.train_sample(jr.key(3))
    assert jnp.allclose(sample.y, sample.x_target, atol=1e-6)
    assert jnp.allclose(sample.y_target, sample.x_target, atol=1e-6)


def test_train_sample_fields_are_float64(problem):
    """canna/__init__.py enables x64; every TrainSample field should be float64."""
    sample = problem.train_sample(jr.key(0))
    for field in sample:
        assert field.dtype == jnp.float64


def test_t_is_sampled_half_open_zero_one(problem):
    """t = jr.uniform(key, ()) draws from the half-open [0, 1), so t
    should never realize exactly 1.0 across many draws."""
    ts = jnp.array([problem.train_sample(jr.key(i)).t for i in range(500)])
    assert jnp.all(ts < 1.0)
    assert jnp.all(ts >= 0.0)
