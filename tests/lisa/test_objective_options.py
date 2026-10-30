"""The flow options and objective: mid-time phase chart, x1 under the NLL."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa.chart import flow_to_physical, physical_to_flow, sample_flow
from canna.lisa.problem import LisaGB
from canna.lisa.sample import sample_posterior
from canna.lisa.simulator import clean_signal, sample_physical
from canna.lisa.train import losses, train_sample
from ._helpers import window


def test_mid_time_chart_round_trips():
    problem = LisaGB(n_sources=2, phase_reference="middle")
    f = window(problem)
    p = sample_physical(problem, jr.key(1), f)
    p_back = flow_to_physical(problem, physical_to_flow(problem, p, f), f)
    # (psi, phi0) and (psi + pi/2, phi0 + pi) are the same signal: compare the two phases
    phases = lambda q: jnp.stack([q[:, 7] + 2 * q[:, 5], q[:, 7] - 2 * q[:, 5]])
    np.testing.assert_allclose(p_back[:, :5], p[:, :5], rtol=1e-6)
    np.testing.assert_allclose(p_back[:, 6], p[:, 6], atol=1e-5)
    np.testing.assert_allclose(
        jnp.exp(1j * phases(p_back)), jnp.exp(1j * phases(p)), atol=1e-6
    )


def test_mid_time_chart_moves_only_the_phases():
    start = LisaGB(n_sources=2, phase_reference="start")
    middle = LisaGB(n_sources=2, phase_reference="middle")
    f = window(start)
    p = sample_physical(start, jr.key(2), f)
    a, b = physical_to_flow(start, p, f), physical_to_flow(middle, p, f)
    radii = lambda x: jnp.hypot(x[:, 4::2], x[:, 5::2])
    np.testing.assert_allclose(a[:, :4], b[:, :4])
    np.testing.assert_allclose(radii(a), radii(b))
    assert not np.allclose(a[:, 4:], b[:, 4:])


def test_mid_time_phase_holds_the_signal_when_f0_moves():
    problem = LisaGB(n_sources=1, phase_reference="middle")
    f = window(problem)
    p = sample_physical(problem, jr.key(3), f)
    # a third of a bin up in f0, holding phi0 or holding the charted mid-time phase
    shifted = p.at[:, 0].add(0.3 / problem.t_obs)
    x = (
        physical_to_flow(problem, shifted, f)
        .at[:, 4:]
        .set(physical_to_flow(problem, p, f)[:, 4:])
    )
    held = flow_to_physical(problem, x, f)
    np.testing.assert_allclose(held[:, 0], shifted[:, 0], rtol=1e-12)
    h = clean_signal(problem, p, f)[:, :2]

    def overlap(q):
        g = clean_signal(problem, q, f)[:, :2]
        return jnp.vdot(h, g).real / jnp.sqrt(jnp.vdot(h, h).real * jnp.vdot(g, g).real)

    # sinc(0.3 pi) ~ 0.86 with the phase held mid-observation, against ~0.5 from the start
    assert overlap(held) > 0.8 and overlap(held) > overlap(shifted) + 0.2


class Oracle(eqx.Module):
    """A stand-in flow whose mean x1 is its conditioning, with a fixed log std."""

    s: float

    def encode(self, y, f):
        return None

    def __call__(self, x, t, y, f, cache=None):
        return jnp.concatenate(
            [jnp.broadcast_to(y, x.shape), jnp.full_like(x, self.s)], axis=-1
        )


def test_sampler_lands_on_the_mean_x1(small_flow):
    problem, _, u0, _, f = small_flow
    x1 = sample_flow(problem, jr.key(9), f)
    out = sample_posterior(Oracle(np.log(1e-2)), u0, x1, f, 8)
    np.testing.assert_allclose(out, jnp.broadcast_to(x1, out.shape), atol=1e-6)


def test_loss_is_the_nll_of_x1(small_flow):
    problem, _, _, _, _ = small_flow
    batch = eqx.filter_jit(jax.vmap(lambda k: train_sample(problem, k)))(
        jr.split(jr.key(4), 8)
    )
    s = np.log(1e-2)
    flow_loss, x_error = losses(Oracle(s), batch._replace(y=batch.x_target))
    np.testing.assert_allclose(flow_loss, 2 * s, atol=1e-6)
    assert x_error < 1e-10
