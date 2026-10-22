"""The opt-in flow options: mid-time phase chart, self-conditioning, x-prediction, loss weights, late times."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa.chart import flow_to_physical, physical_to_flow, sample_flow
from canna.lisa.network import LOSS_GROUPS, LOSS_TIME_BINS, LisaFlow
from canna.lisa.problem import LisaGB
from canna.lisa.sample import sample_posterior
from canna.lisa.simulator import clean_signal, sample_physical
from canna.lisa.train import TrainSample, losses, train_sample
from ._helpers import window

S, X, BINS, C = 4, 8, 64, 6
F = 3.0e-3


def tiny_flow(seed=0, **options):
    problem = SimpleNamespace(
        n_sources=S,
        window_bins=BINS,
        t_obs=63115200.0,
        response_points=16,
        f0_range=(1.0e-4, 1.2e-2),
        phase_reference="middle",
    )
    return LisaFlow(problem, 16, 2, 1, **options, key=jr.key(seed))


def perturbed(module, seed=7, scale=0.3):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jr.split(jr.key(seed), len(leaves))
    noised = [p + scale * jr.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


def inputs(seed=1):
    x = jr.normal(jr.key(seed), (S, X))
    return x, jnp.asarray(0.3), jr.normal(jr.key(seed + 1), (BINS, C)), jnp.asarray(F)


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


def test_options_off_leave_the_flow_and_its_objective_as_before():
    plain = tiny_flow(self_condition=False)
    assert plain.hat_embed is None and plain.loss_log_var is None
    x, t, y, f = inputs()
    np.testing.assert_allclose(plain(x, t, y, f), plain(x, t, y, f, x))
    batch = TrainSample(
        x[None, None], x[None, None], t[None, None], y[None], x[None], f[None]
    )
    parts = losses(plain, batch)
    assert parts.shape == (3,) and parts[2] == parts[0]


def test_self_conditioning_reads_x_hat():
    flow = perturbed(tiny_flow(self_condition=True))
    x, t, y, f = inputs()
    assert not np.allclose(flow(x, t, y, f), flow(x, t, y, f, x))


def test_learned_loss_weights_start_as_the_group_mean_and_train():
    flow = perturbed(
        tiny_flow(loss_weights=True, prediction="velocity", self_condition=False)
    )
    assert flow.loss_log_var.shape == (LOSS_TIME_BINS, len(LOSS_GROUPS))
    flow = eqx.tree_at(
        lambda m: m.loss_log_var, flow, jnp.zeros_like(flow.loss_log_var)
    )
    x, t, y, f = inputs()
    batch = TrainSample(
        x[None, None], 0 * x[None, None], t[None, None], y[None], x[None], f[None]
    )
    dx = flow(x, t, y, f)
    groups = jnp.stack([jnp.mean(jnp.square(dx[:, g])) for g in LOSS_GROUPS])
    np.testing.assert_allclose(losses(flow, batch)[2], groups.mean(), rtol=1e-5)
    grads = eqx.filter_grad(lambda m: losses(m, batch)[2])(flow)
    assert np.count_nonzero(np.asarray(grads.loss_log_var)) == len(LOSS_GROUPS)


def test_x_prediction_loss_weights_start_at_the_fresh_error_per_bin():
    flow = tiny_flow(loss_weights=True, prediction="x")
    log_var = np.asarray(flow.loss_log_var)
    # one λ per bin, shared by the groups, falling from 1 - s = 1e-6 to 1 by 2 ln 10 per decade
    np.testing.assert_allclose(log_var, log_var[:, :1].repeat(len(LOSS_GROUPS), 1))
    np.testing.assert_allclose(
        np.diff(log_var[:, 0]), -2 * np.log(10) * 6.0 / LOSS_TIME_BINS, rtol=1e-6
    )
    np.testing.assert_allclose(
        log_var[-1, 0], 2 * np.log(10) * 6.0 / (2 * LOSS_TIME_BINS), rtol=1e-6
    )


class ConstantTarget(eqx.Module):
    """A stand-in x-predicting flow whose x1 is fixed."""

    x1: jax.Array
    prediction: str = eqx.field(static=True, default="x")
    self_condition: bool = eqx.field(static=True, default=False)

    def encode(self, y, f):
        return None

    def __call__(self, x, t, y, f, x_hat=None, cache=None):
        return self.x1.astype(x.dtype)


def test_x_prediction_sampler_lands_on_its_x1(small_flow):
    problem, _, u0, y, f = small_flow
    x1 = sample_flow(problem, jr.key(9), f)
    out = sample_posterior(ConstantTarget(x1), u0, y, f, 8, 3.0)
    np.testing.assert_allclose(out, jnp.broadcast_to(x1, out.shape), atol=1e-6)


class Echo(eqx.Module):
    """A stand-in x-predicting flow that returns its conditioning as x1."""

    prediction: str = eqx.field(static=True, default="x")
    self_condition: bool = eqx.field(static=True, default=False)
    loss_log_var: None = None

    def encode(self, y, f):
        return None

    def __call__(self, x, t, y, f, x_hat=None, cache=None):
        return y


def test_x_prediction_loss_vanishes_at_the_true_x1(small_flow):
    problem, _, _, _, _ = small_flow
    batch = eqx.filter_jit(jax.vmap(lambda k: train_sample(problem, k, time_power=3.0)))(
        jr.split(jr.key(4), 8)
    )
    remaining = ((1 - batch.t) ** 3.0)[..., None, None]
    np.testing.assert_allclose(
        batch.x_target[:, None] - batch.xt, remaining * batch.dx, atol=1e-5
    )
    parts = losses(Echo(), batch._replace(y=batch.x_target), 3.0)
    assert parts[0] < 1e-8 and parts[1] == 0
    others = jnp.roll(batch.x_target, 1, axis=0)
    assert losses(Echo(), batch._replace(y=others), 3.0)[0] > 1e-2


def test_late_time_draws_reach_the_floor_and_are_off_by_default(small_flow):
    problem, _, _, _, _ = small_flow
    key = jr.key(5)
    plain = train_sample(problem, key, time_power=3.0)
    mixed_off = train_sample(problem, key, time_power=3.0, late_time_mix=0.0)
    # eager GPU ops are not bitwise reproducible: a stray ulp is allowed
    for a, b in zip(plain, mixed_off):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    late = eqx.filter_jit(
        jax.vmap(
            lambda k: train_sample(
                problem, k, time_power=3.0, late_time_mix=1.0, late_time_floor=1e-4
            )
        )
    )(jr.split(key, 256))
    remaining = (1 - late.t) ** 3.0
    assert remaining.min() >= 1e-4 * (1 - 1e-6) and remaining.min() < 1e-3
    # log10(1 - s) uniform on [-4, 0]: Kolmogorov-Smirnov at 1%
    decades = np.sort(np.log10(np.asarray(remaining)).ravel())
    n = decades.size
    ecdf = np.arange(1, n + 1) / n
    assert np.max(np.abs(ecdf - (decades + 4) / 4)) < 1.63 / np.sqrt(n)
