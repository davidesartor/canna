"""The orientation as Cartesian helicity amplitudes z± (F49), their radii warped by asinh."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa.chart import flow_to_physical, physical_to_flow, sample_flow
from canna.lisa.network import Z_REACH
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import clean_signal, sample_physical
from canna.lisa.train import train_sample

NARROW = dict(
    n_sources=4, t_obs=63115200.0, response_points=32, f0_range=(1.0e-4, 12.0e-3)
)


def mismatch(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))


def rescaled_helicity(x, scale):
    """x with each z± moved from its radius r to scale(r)."""
    radius = jnp.repeat(jnp.hypot(x[:, 4::2], x[:, 5::2]), 2, axis=-1)
    return x.at[:, 4:].multiply(scale(radius) / radius)


def test_round_trip_keeps_the_signal():
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(0))
    p = sample_physical(problem, jr.key(1), f)
    x = physical_to_flow(problem, p, f)
    assert x.shape == (4, 8)
    back = flow_to_physical(problem, x, f)
    # psi and phi0 come back as another of the four equivalent pairs; the rest is unchanged
    assert jnp.allclose(
        back[:, [0, 1, 2, 3, 4, 6]], p[:, [0, 1, 2, 3, 4, 6]], rtol=1e-5, atol=1e-6
    )
    assert mismatch(clean_signal(problem, p, f), clean_signal(problem, back, f)) < 1e-5


def test_the_symmetric_images_are_one_flow_point():
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(2))
    p = sample_physical(problem, jr.key(3), f)
    x = physical_to_flow(problem, p, f)
    for dpsi, dphi in [(np.pi, 0.0), (np.pi / 2, np.pi), (3 * np.pi / 2, np.pi)]:
        q = p.at[:, 5].add(dpsi).at[:, 7].add(dphi)
        assert jnp.allclose(physical_to_flow(problem, q, f), x, atol=1e-5)


def test_the_signal_is_linear_in_the_unwarped_helicity_amplitudes():
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(4))
    a = sample_flow(problem, jr.key(5), f)
    b = sample_flow(problem, jr.key(6), f).at[:, :4].set(a[:, :4])
    za, zb = rescaled_helicity(a, jnp.sinh), rescaled_helicity(b, jnp.sinh)
    both = rescaled_helicity(za.at[:, 4:].add(zb[:, 4:]), jnp.arcsinh)
    signal = lambda x: clean_signal(problem, flow_to_physical(problem, x, f), f)
    assert mismatch(signal(both), signal(a) + signal(b)) < 1e-5


def test_face_on_is_one_helicity_alone():
    """z+ = 0 is sin iota = +1, where the data see phi0 - 2 psi alone; z- = 0 is sin iota = -1."""
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(7))
    x = sample_flow(problem, jr.key(8), f)
    for sign, off in [(1.0, slice(4, 6)), (-1.0, slice(6, 8))]:
        p = flow_to_physical(problem, x.at[:, off].set(0.0), f)
        assert bool(jnp.all(jnp.isfinite(p)))
        np.testing.assert_allclose(jnp.sin(p[:, 6]), sign, atol=1e-5)


def test_face_on_signal_sees_one_phase_combination():
    """At sin iota = +1 the data see phi0 - 2 psi alone, at -1 phi0 + 2 psi alone."""
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(12))
    p = sample_physical(problem, jr.key(13), f)
    signal = lambda q: clean_signal(problem, q, f)
    turn = np.pi / 2
    for sign in (1.0, -1.0):
        face_on = p.at[:, 6].set(sign * np.pi / 2)
        # turn phi0 + sign 2 psi (the vanished helicity's phase), then phi0 - sign 2 psi
        unseen = face_on.at[:, 5].add(sign * turn / 4).at[:, 7].add(turn / 2)
        seen = face_on.at[:, 5].add(-sign * turn / 4).at[:, 7].add(turn / 2)
        assert mismatch(signal(face_on), signal(unseen)) < 1e-5
        assert mismatch(signal(face_on), signal(seen)) > 0.5


def test_prior_sin_iota_is_uniform():
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(14))
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(15), 1000), f)
    sin_iota = jnp.sin(p[..., 6]).ravel()
    n = sin_iota.size
    # four standard errors of U[-1, 1]'s sample mean (Var x = 1/3) and variance (Var x² = 4/45)
    assert abs(float(sin_iota.mean())) < 4 * np.sqrt(1 / 3 / n)
    assert abs(float(sin_iota.var()) - 1 / 3) < 4 * np.sqrt(4 / 45 / n)


def test_prior_helicity_is_phase_uniform_and_within_reach():
    problem = LisaGB(**NARROW)
    f = sample_f(problem, jr.key(9))
    draw = eqx.filter_jit(jax.vmap(partial(sample_flow, problem), in_axes=(0, None)))
    x = draw(jr.split(jr.key(10), 2000), f).reshape(-1, 8)
    z = x[:, 4::2] + 1j * x[:, 5::2]
    assert bool(jnp.all(jnp.abs(z) <= Z_REACH))
    assert float(jnp.max(jnp.abs(jnp.mean(z / jnp.abs(z), axis=0)))) < 0.04
    # the two phases are independent under the prior
    pp, pm = jnp.angle(z[:, 0]), jnp.angle(z[:, 1])
    assert abs(complex(jnp.mean(jnp.exp(1j * (pp - pm))))) < 0.06


def test_training_sample_is_finite_and_walks_a_straight_line():
    problem = LisaGB(**NARROW)
    sample = eqx.filter_jit(partial(train_sample, problem, time_power=3.0))(jr.key(11))
    assert sample.xt.shape == (1, 4, 8) and all(
        bool(jnp.all(jnp.isfinite(a))) for a in sample
    )
    remaining = ((1 - sample.t) ** 3.0)[:, None, None]
    assert jnp.allclose(sample.xt + remaining * sample.dx, sample.x_target, atol=1e-6)


def test_pages_fold_every_copy_onto_the_truths():
    from canna.lisa.plots import fold_orientation

    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 2 * np.pi, (4, 8))
    p = np.repeat(truth[None], 4, axis=0)
    p[:, :, 5] += np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])[:, None]
    p[:, :, 7] += np.array([0.0, 0.0, np.pi, np.pi])[:, None]
    np.testing.assert_allclose(
        fold_orientation(p, truth), np.repeat(truth[None], 4, axis=0), atol=1e-12
    )
