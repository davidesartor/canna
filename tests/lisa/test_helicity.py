"""The helicity parametrisation of the orientation (LisaGB.polarisation = "helicity", F49)."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa import LisaGB
from canna.lisa.train import fisher_metric, train_sample

XS = dict(n_sources=4, t_obs=63115200.0, response_points=16, f0_range=(1.0e-4, 12.0e-3))


def mismatch(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))


def test_round_trip_keeps_the_signal():
    problem = LisaGB(**XS, polarisation="helicity")
    f = problem.sample_f(jr.key(0))
    p = problem.sample_physical(jr.key(1), f)
    x = problem.physical_to_flow(p, f)
    assert x.shape == (4, 11)
    back = problem.flow_to_physical(x, f)
    # psi and phi0 come back as another of the four equivalent pairs; the rest is unchanged
    assert jnp.allclose(back[:, [0, 1, 2, 3, 4, 6]], p[:, [0, 1, 2, 3, 4, 6]], rtol=1e-5, atol=1e-6)
    assert mismatch(problem.clean_signal(p, f), problem.clean_signal(back, f)) < 1e-5


def test_the_symmetric_images_are_one_flow_point():
    problem = LisaGB(**XS, polarisation="helicity")
    f = problem.sample_f(jr.key(2))
    p = problem.sample_physical(jr.key(3), f)
    x = problem.physical_to_flow(p, f)
    for dpsi, dphi in [(np.pi, 0.0), (np.pi / 2, np.pi), (3 * np.pi / 2, np.pi)]:
        q = p.at[:, 5].add(dpsi).at[:, 7].add(dphi)
        assert jnp.allclose(problem.physical_to_flow(q, f), x, atol=1e-5)


def test_face_on_measures_phi_minus_only():
    """At sin iota = +1 the data see phi0 - 2 psi alone, at -1 phi0 + 2 psi alone."""
    problem = LisaGB(**XS, polarisation="helicity")
    f = problem.sample_f(jr.key(4))
    x = problem.sample_flow(jr.key(5), f)
    for c, free, fixed in [(1.0, slice(7, 9), slice(9, 11)), (-1.0, slice(9, 11), slice(7, 9))]:
        a = x.at[:, 6].set(c)
        b = a.at[:, free].set(a[:, free] @ jnp.array([[0.0, 1.0], [-1.0, 0.0]]))  # turn by pi/2
        signal = lambda z: problem.clean_signal(problem.flow_to_physical(z, f), f)
        assert mismatch(signal(a), signal(b)) < 1e-3  # the clip leaves sin iota 1e-6 short
        d = a.at[:, fixed].set(a[:, fixed] @ jnp.array([[0.0, 1.0], [-1.0, 0.0]]))
        assert mismatch(signal(a), signal(d)) > 0.5


def test_prior_is_uniform_in_the_new_coordinates():
    problem = LisaGB(**XS, polarisation="helicity")
    f = problem.sample_f(jr.key(6))
    x = jnp.concatenate([problem.sample_flow(k, f) for k in jr.split(jr.key(7), 2000)])
    assert abs(float(x[:, 6].mean())) < 0.03 and abs(float(x[:, 6].var()) - 1 / 3) < 0.02
    assert jnp.allclose(jnp.hypot(x[:, 7], x[:, 8]), 1.0) and jnp.allclose(jnp.hypot(x[:, 9], x[:, 10]), 1.0)
    for k in (7, 8, 9, 10):
        assert abs(float(x[:, k].mean())) < 0.04


def test_default_layout_is_unchanged():
    old, new = LisaGB(**XS), LisaGB(**XS, polarisation="helicity")
    f = old.sample_f(jr.key(8))
    p = old.sample_physical(jr.key(9), f)
    assert jnp.allclose(old.flow_to_physical(old.physical_to_flow(p, f), f), p, rtol=1e-5, atol=1e-6)
    assert old.flow_names[6:] == ["or x", "or y", "or z", "cos phi0", "sin phi0"]
    assert new.flow_names[6:] == ["sin iota", "cos phi+", "sin phi+", "cos phi-", "sin phi-"]


def test_fisher_metric_and_training_sample_are_finite():
    problem = LisaGB(**XS, polarisation="helicity")
    sample = train_sample(problem, jr.key(10), 3.0, fisher_cap=100.0)
    assert sample.xt.shape == (4, 11) and bool(jnp.all(jnp.isfinite(sample.metric)))
    # also at the poles of sin iota, where the clip guards arcsin's slope
    f = problem.sample_f(jr.key(11))
    x = problem.sample_flow(jr.key(12), f).at[:, 6].set(jnp.array([1.0, -1.0, 0.0, 0.5]))
    assert bool(jnp.all(jnp.isfinite(fisher_metric(problem, x, jnp.asarray(0.5), f, 100.0))))


def test_pages_fold_every_copy_onto_the_truths():
    from canna.lisa.eval import fold_orientation

    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 2 * np.pi, (4, 8))
    p = np.repeat(truth[None], 4, axis=0)
    p[:, :, 5] += np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])[:, None]
    p[:, :, 7] += np.array([0.0, 0.0, np.pi, np.pi])[:, None]
    np.testing.assert_allclose(fold_orientation(p, truth), np.repeat(truth[None], 4, axis=0), atol=1e-12)


def test_geodesics_stay_on_the_helicity_manifold():
    problem = LisaGB(**XS, polarisation="helicity")
    f = problem.sample_f(jr.key(13))
    x0 = problem.sample_flow(jr.key(14), f)
    x1 = problem.sample_flow(jr.key(15), f)
    v = problem.log_map(x0, x1)
    assert jnp.allclose(problem.exp_map(x0, v), x1, atol=1e-5)
    for s in (0.25, 0.5, 0.75):
        xs = problem.exp_map(x0, s * v)
        assert bool(jnp.all(jnp.abs(xs[:, 6]) <= 1.0))
        assert jnp.allclose(jnp.hypot(xs[:, 7], xs[:, 8]), 1.0, atol=1e-6)
        assert jnp.allclose(jnp.hypot(xs[:, 9], xs[:, 10]), 1.0, atol=1e-6)
    # the two phases are independent and uniform under the prior: no correlation between them
    x = jnp.concatenate([problem.sample_flow(k, f) for k in jr.split(jr.key(16), 1000)])
    pp, pm = jnp.arctan2(x[:, 8], x[:, 7]), jnp.arctan2(x[:, 10], x[:, 9])
    assert abs(complex(jnp.mean(jnp.exp(1j * (pp - pm))))) < 0.06
