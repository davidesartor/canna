"""Problem.sample_observation(clean=True) path: y_target provenance, split ordering."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.problems import NoisyPoint, NoisySinusoid


@pytest.fixture
def point_problem():
    return NoisyPoint(noise_std=1.0)


@pytest.fixture
def sinusoid_problem():
    return NoisySinusoid(n_sources=2)


@pytest.fixture
def key():
    return jr.key(0)


# --- clean path ignores its key argument ---


def test_point_clean_observation_invariant_to_key(point_problem):
    p = jnp.array([1.5, -2.5])
    keys = jr.split(jr.key(3), 50)
    draws = jax.vmap(lambda k: point_problem.sample_observation(k, p, clean=True))(keys)
    assert jnp.all(jnp.var(draws, axis=0) < 1e-12)


def test_sinusoid_clean_observation_invariant_to_key(sinusoid_problem, key):
    p = sinusoid_problem.sample_physical(key)
    keys = jr.split(jr.key(4), 20)
    draws = jax.vmap(lambda k: sinusoid_problem.sample_observation(k, p, clean=True))(
        keys
    )
    assert jnp.all(jnp.var(draws, axis=0) < 1e-10)


def test_key_o_target_does_not_affect_y_target(point_problem):
    """key_o_target is split off in train_sample but the clean branch never
    consumes it, so preprocess(sample_observation(_, p, clean=True)) is a pure
    function of p regardless of which key would have landed in that slot."""
    p = jnp.array([0.4, 3.3])
    y_targets = [
        point_problem.preprocess(point_problem.sample_observation(k, p, clean=True))
        for k in jr.split(jr.key(5), 10)
    ]
    for a, b in zip(y_targets, y_targets[1:]):
        assert jnp.allclose(a, b, atol=1e-6)


# --- clean bypasses noise magnitude entirely ---


def test_point_clean_observation_ignores_noise_std_magnitude():
    p = jnp.array([2.0, -3.0])
    key = jr.key(9)
    huge_noise = NoisyPoint(noise_std=1e8)
    zero_noise = NoisyPoint(noise_std=0.0)
    clean_huge = huge_noise.sample_observation(key, p, clean=True)
    clean_zero = zero_noise.sample_observation(key, p, clean=True)
    assert jnp.allclose(clean_huge, p, atol=1e-9)
    assert jnp.allclose(clean_huge, clean_zero, atol=1e-9)
    assert jnp.allclose(clean_huge, zero_noise.sample_observation(key, p), atol=1e-9)


def test_sinusoid_clean_observation_ignores_noise_level_magnitude(key):
    p = NoisySinusoid(n_sources=2).sample_physical(key)
    huge = NoisySinusoid(n_sources=2, noise_level=1e12)
    zero = NoisySinusoid(n_sources=2, noise_level=0.0)
    clean_huge = huge.sample_observation(key, p, clean=True)
    clean_zero = zero.sample_observation(key, p, clean=True)
    assert jnp.allclose(clean_huge, huge.clean_signal(p), atol=1e-6)
    assert jnp.allclose(clean_huge, clean_zero, atol=1e-6)
    assert jnp.allclose(clean_huge, zero.sample_observation(key, p), atol=1e-6)


# --- shape / dtype / batch parity with the noisy path ---


def test_point_clean_observation_preserves_dtype():
    p = jnp.array([1.0, 2.0], dtype=jnp.float64)
    o = NoisyPoint().sample_observation(jr.key(0), p, clean=True)
    assert o.dtype == jnp.float64


def test_point_clean_observation_batches_over_leading_p_axis():
    problem = NoisyPoint(noise_std=5.0)
    p_batched = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    o = problem.sample_observation(jr.key(0), p_batched, clean=True)
    assert o.shape == p_batched.shape
    assert jnp.allclose(o, p_batched, atol=1e-6)


def test_sinusoid_clean_observation_preserves_dtype(sinusoid_problem, key):
    p = sinusoid_problem.sample_physical(key)
    o = sinusoid_problem.sample_observation(key, p, clean=True)
    assert o.dtype == jnp.float64


def test_sinusoid_clean_observation_batches_over_leading_p_axis(sinusoid_problem, key):
    keys = jr.split(key, 4)
    p_batch = jnp.stack([sinusoid_problem.sample_physical(k) for k in keys])
    o = sinusoid_problem.sample_observation(key, p_batch, clean=True)
    assert o.shape[0] == 4
    assert o.shape[1:] == sinusoid_problem.clean_signal(p_batch[0]).shape
    assert jnp.allclose(o, sinusoid_problem.clean_signal(p_batch), atol=1e-5)


# --- clean flag under jax.jit / vmap ---


def test_point_clean_branch_matches_eager_under_jit(point_problem):
    p = jnp.array([0.7, -1.1])
    key = jr.key(11)
    jitted = eqx.filter_jit(point_problem.sample_observation)
    assert jnp.allclose(
        jitted(key, p, clean=True),
        point_problem.sample_observation(key, p, clean=True),
        atol=1e-6,
    )
    assert jnp.allclose(
        jitted(key, p, clean=False),
        point_problem.sample_observation(key, p, clean=False),
        atol=1e-6,
    )


def test_sinusoid_clean_branch_matches_eager_under_jit(sinusoid_problem, key):
    p = sinusoid_problem.sample_physical(key)
    jitted = eqx.filter_jit(sinusoid_problem.sample_observation)
    assert jnp.allclose(
        jitted(key, p, clean=True),
        sinusoid_problem.sample_observation(key, p, clean=True),
        atol=1e-6,
    )


def test_point_clean_flag_stays_static_when_passed_through_jit(point_problem):
    """clean is branched on with a bare `if clean:`, so it must reach the branch as a
    Python bool; filter_jit keeps the non-array flag static and retraces per value."""
    p = jnp.array([0.1, 0.2])
    key = jr.key(12)
    jitted = eqx.filter_jit(
        lambda k, p, clean: point_problem.sample_observation(k, p, clean)
    )
    assert jnp.allclose(jitted(key, p, True), p, atol=1e-6)
    assert not jnp.allclose(jitted(key, p, False), p, atol=1e-6)


def test_point_train_sample_is_jit_compatible(point_problem):
    """The clean=True/False literals inside train_sample are resolved at trace
    time (never passed in from outside), so jitting train_sample itself must
    not hit the tracer issue that a dynamically-passed clean flag would."""
    key = jr.key(13)
    jitted = eqx.filter_jit(point_problem.train_sample)
    eager = point_problem.train_sample(key)
    traced = jitted(key)
    for a, b in zip(eager, traced):
        assert jnp.allclose(a, b, atol=1e-6)


# --- 5-way split provenance: y <- key_o (2nd), y_target <- clean(key_o_target, 3rd), t <- key_t (5th) ---


def test_point_y_matches_second_subkey_key_o(point_problem):
    key = jr.key(77)
    key_p, key_o, key_o_target, key_x0, key_t = jr.split(key, 5)
    p = point_problem.sample_physical(key_p)
    expected_y = point_problem.preprocess(point_problem.sample_observation(key_o, p))
    sample = point_problem.train_sample(key)
    assert jnp.allclose(sample.y, expected_y, atol=1e-6)


def test_point_t_matches_fifth_subkey_key_t(point_problem):
    key = jr.key(13)
    *_, key_t = jr.split(key, 5)
    expected_t = jr.uniform(key_t, ())
    sample = point_problem.train_sample(key)
    assert jnp.allclose(sample.t, expected_t, atol=1e-8)


def test_sinusoid_y_matches_second_subkey_key_o(sinusoid_problem, key):
    key_p, key_o, key_o_target, key_x0, key_t = jr.split(key, 5)
    p = sinusoid_problem.sample_physical(key_p)
    expected_y = sinusoid_problem.preprocess(
        sinusoid_problem.sample_observation(key_o, p)
    )
    sample = sinusoid_problem.train_sample(key)
    assert jnp.allclose(sample.y, expected_y, atol=1e-4)


def test_sinusoid_y_target_matches_clean_signal_of_key_p(sinusoid_problem, key):
    key_p, *_ = jr.split(key, 5)
    p = sinusoid_problem.sample_physical(key_p)
    expected_y_target = sinusoid_problem.preprocess(sinusoid_problem.clean_signal(p))
    sample = sinusoid_problem.train_sample(key)
    assert jnp.allclose(sample.y_target, expected_y_target, atol=1e-4)


def test_sinusoid_clean_through_preprocess_matches_preprocess_of_clean_signal(
    sinusoid_problem, key
):
    p = sinusoid_problem.sample_physical(key)
    o_clean = sinusoid_problem.sample_observation(key, p, clean=True)
    lhs = sinusoid_problem.preprocess(o_clean)
    rhs = sinusoid_problem.preprocess(sinusoid_problem.clean_signal(p))
    assert jnp.allclose(lhs, rhs, atol=1e-6)


# --- y_target has zero variance across draws; residual y - y_target is pure noise ---


def test_point_y_target_zero_variance_across_keys_fixed_p(point_problem):
    p = jnp.array([1.0, -1.0])
    keys = jr.split(jr.key(6), 200)
    y_targets = jax.vmap(
        lambda k: point_problem.preprocess(
            point_problem.sample_observation(k, p, clean=True)
        )
    )(keys)
    assert jnp.all(jnp.var(y_targets, axis=0) < 1e-12)


def test_point_residual_y_minus_y_target_matches_noise_and_chart_scale():
    noise_std = 0.6
    problem = NoisyPoint(seed=0, dim=2, noise_std=noise_std)
    keys = jr.split(jr.key(2), 4000)
    samples = jax.vmap(problem.train_sample)(keys)
    residual = samples.y - samples.y_target
    scale = problem.chart.scale
    expected_cov = noise_std**2 * scale @ scale.T
    assert jnp.allclose(jnp.cov(residual, rowvar=False), expected_cov, atol=0.05)
    assert jnp.allclose(jnp.mean(residual, axis=0), jnp.zeros(2), atol=0.05)
