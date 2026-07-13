"""Session fixtures: one shared batch of real physics samples for every probe.

Runs JAX numerical code, so launch on a compute node
(`uv run pytest tests/`). Draw count/seed via env: N_PROBE, PROBE_CHUNK, SEED.
"""

import os
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from canna import lisa

from _helpers import snr_of_signal

N_PROBE = int(os.environ.get("N_PROBE", 64))
PROBE_CHUNK = int(os.environ.get("PROBE_CHUNK", 16))
SEED = int(os.environ.get("SEED", 0))


@pytest.fixture(scope="session")
def batch():
    """N real ``get_physics_sample`` draws, generated in chunks, as host numpy arrays.

    Fields: u, params (u-space and physical, both masked to the active problem),
    datastream, signal, y (datastream WDM), y_clean (signal WDM), snr.
    """
    keys = jr.split(jr.key(SEED), N_PROBE)

    @jax.jit
    def gen(chunk_keys):
        u, params, datastream, signal, y, y_clean = jax.vmap(lisa.get_physics_sample)(chunk_keys)
        snr = jax.vmap(snr_of_signal)(signal)
        return u, params, datastream, signal, y, y_clean, snr

    chunks = [gen(keys[i : i + PROBE_CHUNK]) for i in range(0, N_PROBE, PROBE_CHUNK)]
    u, params, datastream, signal, y, y_clean, snr = (
        np.concatenate([np.asarray(c[j]) for c in chunks], axis=0) for j in range(7)
    )
    return SimpleNamespace(
        n=N_PROBE,
        param_names=list(lisa.PARAMETER_NAMES),
        u=u,
        params=params,
        datastream=datastream,
        signal=signal,
        y=y,
        y_clean=y_clean,
        snr=snr,
    )
