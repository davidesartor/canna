"""The Doppler-demodulated source bands (problem.DopplerGeometry, network.doppler_bands, F57)."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

import canna.lisa as lisa
from canna.lisa import LisaFlow, LisaGB
from canna.lisa.constants import SPEED_OF_LIGHT
from canna.lisa.network import doppler_bands
from canna.lisa.train import TrainState

B = dict(n_sources=4, t_obs=63115200.0, f0_range=(1.0e-4, 12.0e-3))


def test_fitted_orbit_gives_the_doppler_phase_of_the_orbit_file():
    problem = LisaGB(**B)
    geometry = problem.doppler_geometry
    t = np.linspace(0.0, problem.t_obs, 997)
    r = np.asarray(problem.orbit.compute_position(t, [1, 2, 3])).mean(axis=1)
    n = np.array([0.3, -0.5, 0.81])
    n /= np.linalg.norm(n)
    exact = 2 * np.pi * 12e-3 * (r @ n) / SPEED_OF_LIGHT
    fitted = np.asarray(geometry.phase(jnp.array(12e-3), jnp.asarray(n), jnp.asarray(t)))
    assert np.abs(fitted - exact).max() < 3e-3  # rad, at the top of the band
    assert 1.0 < np.abs(exact).max() < 38.0  # at most 2 pi f (1 AU / c) = 37.6 rad at 12 mHz


def band_power(problem, x, y, f):
    bands = doppler_bands(y, x, f, problem.window_geometry, problem.doppler_geometry, 16)
    amp = np.sinh(np.asarray(bands[:, : 16 * 3]))
    return (amp**2).sum(axis=-1)


def test_demodulation_collapses_a_source_at_its_true_sky_only():
    problem = LisaGB(**B)
    p = np.array(problem.sample_physical(jr.key(1), jnp.array(3e-3)))
    p[:, 0] = 3e-3 + np.arange(4) * 200 / problem.t_obs  # four sources 200 bins apart
    p[:, 1] = 0.3  # a small chirp
    # skies near the ecliptic, where the Doppler modulation is largest (index ~ 9 rad at 3 mHz)
    p[:, 3], p[:, 4] = [0.3, 1.9, 3.5, 5.1], [0.1, -0.2, 0.3, -0.1]
    f = jnp.array(p[0, 0])
    y = problem.preprocess(problem.clean_signal(jnp.asarray(p), f), f)
    x = problem.physical_to_flow(jnp.asarray(p), f)
    true = band_power(problem, x, y, f)
    mirror = x.at[:, 3:5].multiply(-1.0)  # the sky through the opposite meridian
    wrong = band_power(problem, mirror, y, f)
    plain = (np.abs(np.asarray(y[:, :3] + 1j * y[:, 3:])) ** 2).sum()
    assert np.all(true > 2.0 * wrong)
    assert true.sum() > 0.3 * plain


def test_the_doppler_option_leaves_every_other_init_unchanged():
    problem = LisaGB(**dict(B, f0_range=(1.0e-4, 1.3e-3)), response_points=16)
    x_shape, y_shape = (4, 11), (problem.window_bins, 6)
    common = dict(x_shape=x_shape, y_shape=y_shape, hidden_dim=32, num_heads=2, num_blocks=1,
                  f_range=problem.f0_range, patch=16, rope=True, window=problem.window_geometry,
                  key=jr.key(3))
    plain = LisaFlow(**common)
    with_doppler = LisaFlow(**common, doppler=problem.doppler_geometry)
    assert plain.d_embed is None and with_doppler.d_embed is not None
    assert jnp.array_equal(plain.x_embed.linear.weight, with_doppler.x_embed.linear.weight)
    assert jnp.array_equal(plain.y_embed.linear.weight, with_doppler.y_embed.linear.weight)


def test_doppler_network_trains_a_step():
    with open(Path(lisa.__file__).parent / "configs" / "B-stft-hd.yaml") as f:
        config = yaml.safe_load(f)
    assert config["network"]["doppler"] and config["problem"]["polarisation"] == "helicity"
    config["network"] = dict(config["network"], hidden_dim=32, num_blocks=1, num_heads=2)
    state = TrainState.from_config(argparse.Namespace(**config, dtype="float32"))
    assert state.flow.doppler is not None
    state, losses = state.train_epoch(2, 2, 3.0, 0.1)
    assert np.all(np.isfinite(np.asarray(losses)))
