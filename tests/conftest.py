import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches():
    """Drop each test file's compiled functions, so a long-lived xdist worker doesn't keep them all."""
    yield
    jax.clear_caches()
    gc.collect()
