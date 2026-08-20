"""NoisyPoint: Gaussian point in R^D observed under additive Gaussian noise."""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

from canna.point import NoisyPoint


def test_noisy_point_default_dim_is_two():
    problem = NoisyPoint()
    assert problem.cov.shape == (2, 2)
    assert problem.noise_chol.shape == (2, 2)


def test_maps_are_flat():
    problem = NoisyPoint(dim=3)
    x0 = jnp.array([0.3, -1.2, 0.7])
    x1 = jnp.array([-0.5, 0.4, 2.0])
    assert jnp.allclose(problem.log_map(x0, x1), x1 - x0)
    assert jnp.allclose(problem.exp_map(x0, x1 - x0), x1)
    assert jnp.allclose(
        problem.exp_map(x0, 0.25 * problem.log_map(x0, x1)), x0 + 0.25 * (x1 - x0)
    )


def test_sample_physical_shape():
    problem = NoisyPoint(dim=3)
    p = problem.sample_physical(jr.key(0))
    assert p.shape == (3,)


def test_sample_flow_shape_matches_flow_dim():
    problem = NoisyPoint(dim=3)
    x = problem.sample_flow(jr.key(0))
    assert x.shape == (3,)


def test_sample_observation_unbiased():
    problem = NoisyPoint()
    p = jnp.array([3.0, -1.0])
    keys = jr.split(jr.key(0), 4000)
    obs = jax.vmap(lambda k: problem.sample_observation(k, p))(keys)
    assert jnp.allclose(jnp.mean(obs, axis=0), p, atol=0.1)


def test_sample_observation_noise_matches_noise_cov():
    problem = NoisyPoint(seed=4)
    p = jnp.array([0.0, 0.0])
    keys = jr.split(jr.key(1), 20000)
    obs = jax.vmap(lambda k: problem.sample_observation(k, p))(keys)
    assert jnp.allclose(jnp.cov(obs, rowvar=False), problem.noise_cov, atol=0.05)


def test_preprocess_preserves_shape():
    problem = NoisyPoint()
    o = jnp.array([1.0, 2.0])
    y = problem.preprocess(o)
    assert y.shape == o.shape


def test_preprocess_matches_physical_to_flow():
    """preprocess(o) == chart.forward(o), whitening the
    observation with the same Affine chart used for the physical->point map."""
    problem = NoisyPoint()
    o = jnp.array([3.3, -7.1])
    assert jnp.allclose(problem.preprocess(o), problem.physical_to_flow(o), atol=1e-6)


def test_preprocess_not_identity_for_nonzero_shift_scale():
    """Confirms preprocess actually transforms o when the
    chart's shift/scale are non-trivial (the random full covariance)."""
    problem = NoisyPoint()
    o = jnp.array([1.0, 1.0])
    assert not jnp.allclose(problem.preprocess(o), o, atol=1e-3)


def test_sample_flow_is_a_standard_normal_draw():
    """The flow's base distribution is N(0, I), independent of the prior's cov."""
    problem = NoisyPoint()
    key = jr.key(21)
    assert jnp.allclose(problem.sample_flow(key), jr.normal(key, (2,)), atol=1e-6)


def test_sample_observation_vmaps_over_a_batch_of_p():
    """One example per call; a batch is the caller's vmap."""
    problem = NoisyPoint()
    p_batched = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    o = jax.vmap(lambda p: problem.sample_observation(jr.key(0), p))(p_batched)
    assert o.shape == p_batched.shape
    # one key, so every row gets the same additive draw
    assert jnp.allclose(o - p_batched, o[0] - p_batched[0], atol=1e-6)


def test_the_additive_noise_is_independent_of_p():
    problem = NoisyPoint()
    key = jr.key(2)
    a = problem.sample_observation(key, jnp.array([2.0, -2.0])) - jnp.array([2.0, -2.0])
    b = problem.sample_observation(key, jnp.array([0.0, 7.0])) - jnp.array([0.0, 7.0])
    assert jnp.allclose(a, b, atol=1e-6)


# --------------------------------------------------- random full-covariance prior


def test_prior_covariance_is_spd():
    cov = NoisyPoint(dim=4).cov
    assert jnp.allclose(cov, cov.T, atol=1e-6)
    assert bool(jnp.all(jnp.linalg.eigvalsh(cov) > 0))


def test_prior_covariance_cholesky_succeeds():
    cov = NoisyPoint(dim=5).cov
    L = jnp.linalg.cholesky(cov)
    assert bool(jnp.all(jnp.isfinite(L)))
    assert jnp.allclose(L @ L.T, cov, atol=1e-5)


def test_prior_covariance_is_full_not_diagonal():
    cov = NoisyPoint(dim=3).cov
    off_diagonal = cov - jnp.diag(jnp.diag(cov))
    assert float(jnp.max(jnp.abs(off_diagonal))) > 1e-3


def test_chart_whitens_the_prior_to_identity_covariance():
    problem = NoisyPoint(dim=3, seed=1)
    keys = jr.split(jr.key(0), 20000)
    p = jax.vmap(problem.sample_physical)(keys)
    z = jax.vmap(problem.physical_to_flow)(p)
    empirical = jnp.cov(z, rowvar=False)
    assert jnp.allclose(empirical, jnp.eye(3), atol=0.05)


def test_chart_whitening_roundtrip_recovers_physical_point():
    problem = NoisyPoint(dim=4, seed=2)
    p = jnp.array([1.0, -2.0, 0.5, 3.0])
    assert jnp.allclose(
        problem.flow_to_physical(problem.physical_to_flow(p)), p, atol=1e-4
    )


def test_draws_are_centred_on_zero():
    keys = jr.split(jr.key(11), 20000)
    p = jax.vmap(NoisyPoint(dim=3).sample_physical)(keys)
    assert jnp.allclose(jnp.mean(p, axis=0), 0.0, atol=0.05)


def test_same_seed_is_deterministic():
    a, b = NoisyPoint(dim=3, seed=7), NoisyPoint(dim=3, seed=7)
    assert jnp.allclose(a.cov, b.cov, atol=0.0)


def test_different_seed_gives_a_different_covariance():
    a, b = NoisyPoint(dim=3, seed=0), NoisyPoint(dim=3, seed=1)
    assert not jnp.allclose(a.cov, b.cov, atol=1e-3)


def test_dim_sets_the_covariance_size():
    assert NoisyPoint(dim=6).cov.shape == (6, 6)


def test_empirical_correlation_matches_prior_covariance():
    problem = NoisyPoint(dim=2, seed=3)
    keys = jr.split(jr.key(4), 20000)
    p = jax.vmap(problem.sample_physical)(keys)
    empirical = jnp.cov(p, rowvar=False)
    assert jnp.allclose(empirical, problem.cov, atol=0.1)


# --- log_likelihood ---


def test_log_likelihood_matches_gaussian_logpdf_up_to_a_constant():
    """Body is the unnormalized quadratic (no -D/2 log(2*pi*sigma**2) term), so it equals
    the full Gaussian logpdf only up to an additive constant fixed by noise_cov.
    """
    problem = NoisyPoint(dim=3)
    cov = problem.noise_cov
    pairs = [
        (jnp.array([1.0, 2.0, -0.5]), jnp.array([0.7, 2.3, 0.1])),
        (jnp.array([0.0, 0.0, 0.0]), jnp.array([5.0, -5.0, 5.0])),
        (jnp.array([-3.0, 1.0, 1.0]), jnp.array([-3.0, 1.0, 1.0])),
    ]
    offsets = jnp.array(
        [
            jstats.multivariate_normal.logpdf(o, p, cov) - problem.log_likelihood(p, o)
            for o, p in pairs
        ]
    )
    assert jnp.allclose(offsets, offsets[0], atol=1e-6)


def test_log_likelihood_monotonically_decreases_with_distance():
    problem = NoisyPoint(dim=2)
    o, direction = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
    radii = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
    values = jax.vmap(lambda r: problem.log_likelihood(r * direction, o))(radii)
    assert bool(jnp.all(values[:-1] > values[1:]))


def test_log_likelihood_symmetric_in_o_and_p():
    problem = NoisyPoint(dim=3)
    o, p = jnp.array([1.0, 2.0, -3.0]), jnp.array([0.5, -1.0, 2.0])
    assert jnp.allclose(
        problem.log_likelihood(p, o), problem.log_likelihood(o, p), atol=1e-8
    )


def test_log_likelihood_is_maximal_at_a_perfect_match():
    problem = NoisyPoint(dim=2)
    o = jnp.array([1.0, -1.0])
    assert float(problem.log_likelihood(o, o)) == 0.0
    assert problem.log_likelihood(o + jnp.array([0.3, 0.0]), o) < 0.0


def test_log_likelihood_scalar_shape_no_batch():
    problem = NoisyPoint(dim=4)
    assert problem.log_likelihood(jnp.ones(4), jnp.zeros(4)).shape == ()


def test_log_likelihood_vmaps_over_a_batch_of_o():
    problem = NoisyPoint(dim=2)
    p = jnp.array([1.0, -1.0])
    o = jnp.array([[1.0, -1.0], [2.0, 0.0], [0.0, 0.0]])
    ll = jax.vmap(lambda o_i: problem.log_likelihood(p, o_i))(o)
    assert ll.shape == (3,)
    reference = jnp.stack([problem.log_likelihood(p, o[i]) for i in range(3)])
    assert jnp.allclose(ll, reference, atol=1e-6)


def test_log_likelihood_vmaps_over_a_batch_of_p():
    problem = NoisyPoint(dim=2)
    o = jnp.array([1.0, -1.0])
    p = jnp.array([[1.0, -1.0], [2.0, 0.0], [0.0, 0.0]])
    ll = jax.vmap(lambda p_i: problem.log_likelihood(p_i, o))(p)
    assert ll.shape == (3,)
    reference = jnp.stack([problem.log_likelihood(p[i], o) for i in range(3)])
    assert jnp.allclose(ll, reference, atol=1e-6)


def test_log_likelihood_dim_one_is_scalar_and_finite():
    problem = NoisyPoint(dim=1, seed=9)
    p = problem.sample_physical(jr.key(0))
    assert p.shape == (1,)
    ll = problem.log_likelihood(p, p)
    assert ll.shape == () and jnp.isfinite(ll)
