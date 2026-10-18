"""The STFT data tokens (network.stft_tokens, LisaFlow(stft=True), F50)."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

import canna.lisa as lisa
from canna.lisa.network import stft_tokens
from canna.lisa.train import TrainState

PATCH = 16


def white(key, bins=64):
    return jr.normal(key, (bins, 6))


def invert(tokens, patch=PATCH, channels=3):
    """z back from the features: |z| = sinh(amplitude), z = phase * sqrt(1 + |z|^2)."""
    n = patch * channels
    amp, re, im = tokens[:, :n], tokens[:, n : 2 * n], tokens[:, 2 * n :]
    z = (re + 1j * im) * jnp.sqrt(1 + jnp.sinh(amp) ** 2)
    z = z.reshape(-1, patch, channels)
    bins = jnp.fft.fft(z, axis=1, norm="ortho").reshape(-1, channels)
    return jnp.concat([bins.real, bins.imag], axis=-1)


def test_shape_and_invertibility():
    y = 50.0 * white(jr.key(0))  # loud, so the inversion is well conditioned
    tokens = stft_tokens(y, PATCH)
    assert tokens.shape == (64 // PATCH, PATCH * 9)
    np.testing.assert_allclose(invert(tokens), y, rtol=1e-4, atol=1e-3)


def test_white_noise_stays_white_and_tokens_keep_their_band():
    y = white(jr.key(1), bins=4096)
    tokens = stft_tokens(y, PATCH)
    n = PATCH * 3
    power = jnp.sinh(tokens[:, :n]) ** 2  # |z|^2: 2 per complex sample for unit parts
    assert abs(float(power.mean()) - 2.0) < 0.05
    # zeroing one token's bins changes that token only
    tokens2 = stft_tokens(y.at[16:32].set(0.0), PATCH)
    changed = jnp.any(tokens2 != tokens, axis=-1)
    assert changed.tolist() == [i == 1 for i in range(4096 // PATCH)]


def test_features_are_bounded_and_finite_at_zero():
    tokens = stft_tokens(jnp.zeros((64, 6)), PATCH)
    assert bool(jnp.all(tokens == 0.0))
    loud = stft_tokens(1e3 * white(jr.key(2)), PATCH)
    n = PATCH * 3
    phase = loud[:, n:2 * n] + 1j * loud[:, 2 * n :]
    assert bool(jnp.all(jnp.abs(phase) < 1.0)) and float(jnp.abs(phase).min()) > 0.9


def test_stft_network_trains_a_step():
    with open(Path(lisa.__file__).parent / "configs" / "XS-stft.yaml") as f:
        config = yaml.safe_load(f)
    assert config["network"]["stft"] and config["decay_adamw"] is False
    config["network"] = dict(config["network"], hidden_dim=32, num_blocks=1, num_heads=2)
    state = TrainState.from_config(argparse.Namespace(**config, dtype="float32"))
    assert state.flow.stft
    state, losses = state.train_epoch(4, 2, 3.0, 0.1)
    assert np.all(np.isfinite(np.asarray(losses)))
