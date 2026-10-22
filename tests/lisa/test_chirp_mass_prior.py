"""The chirp-mass prior: DWD component masses (Korol et al. 2022) and their log whitening."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.physics import chirp_mass_moments, chirp_mass_support, sample_chirp_mass

DRAWS = 20000
MASS_RANGE = (0.15, 1.4)
NARROW = (0.3, 0.8)


def draw(mass_range, key):
    """DRAWS chirp masses from the prior over mass_range."""
    return jax.vmap(partial(sample_chirp_mass, mass_range=mass_range))(
        jr.split(key, DRAWS)
    )


def test_the_quadrature_matches_the_sampled_log_moments():
    log_mean, log_std = chirp_mass_moments(MASS_RANGE)
    logs = jnp.log(draw(MASS_RANGE, jr.key(4)))
    assert jnp.allclose(log_std, logs.std(), rtol=0.05)
    assert jnp.allclose(log_mean, logs.mean(), rtol=0.05)


def test_a_narrower_component_mass_range_narrows_the_whitening():
    assert chirp_mass_moments(NARROW)[1] < chirp_mass_moments(MASS_RANGE)[1]


def test_draws_stay_inside_the_support():
    low, high = chirp_mass_support(MASS_RANGE)
    draws = draw(MASS_RANGE, jr.key(0))
    assert jnp.all(draws >= low) and jnp.all(draws <= high)


def test_the_whitening_takes_the_draws_to_roughly_unit_spread():
    # the whitening spans one sigma in log, so the flow coordinate comes out near unit
    # std and centred, the tails reaching a few units out rather than staying in a box
    log_mean, log_std = chirp_mass_moments(MASS_RANGE)
    draws = draw(MASS_RANGE, jr.key(1))
    points = (jnp.log(draws) - log_mean) / log_std
    assert abs(float(points.mean())) < 0.1
    assert 0.8 < float(points.std()) < 1.2


def test_the_distribution_peaks_where_korol_2022_puts_it():
    # a flat secondary on [0.15, m1] under the Kepler et al. 2015 primary mass
    # function leaves a broad chirp mass peaked near 0.45 Msun, with a median
    # around 0.43 and only per-cent-level weight below 0.25
    draws = draw(MASS_RANGE, jr.key(2))
    assert 0.40 < jnp.median(draws) < 0.46
    assert 0.005 < jnp.mean(draws < 0.25) < 0.03
    assert jnp.mean(draws > 0.7) < 0.05


def test_a_narrower_component_mass_range_narrows_the_chirp_mass():
    wide = draw(MASS_RANGE, jr.key(3))
    narrow = draw(NARROW, jr.key(3))
    assert narrow.std() < wide.std()
    assert jnp.all(narrow >= 0.3 * 2**-0.2)
