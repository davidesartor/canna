"""ChirpMass: DWD component masses (Korol et al. 2022), Euclidean geometry, LogAffine chart."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.geometries import Euclidean
from canna.charts import LogAffine
from canna.problems.lisa import ChirpMass

DRAWS = 20000


def test_geometry_is_euclidean():
    assert isinstance(ChirpMass().geometry, Euclidean)


def test_chart_is_logaffine():
    assert isinstance(ChirpMass().chart, LogAffine)


def test_the_charts_quadrature_matches_the_sampled_log_moments():
    prior = ChirpMass()
    chart = prior.chart
    logs = jnp.log(jax.vmap(prior)(jr.split(jr.key(4), DRAWS)))
    assert jnp.allclose(1 / chart.scale, logs.std(), rtol=0.05)
    assert jnp.allclose(-chart.shift / chart.scale, logs.mean(), rtol=0.05)


def test_a_narrower_component_mass_range_narrows_the_chart():
    assert ChirpMass(m_min=0.3, m_max=0.8).chart.scale > ChirpMass().chart.scale


def test_draws_stay_inside_the_support():
    prior = ChirpMass()
    low, high = prior.support
    draws = jax.vmap(prior)(jr.split(jr.key(0), DRAWS))
    assert jnp.all(draws >= low) and jnp.all(draws <= high)


def test_the_chart_whitens_the_draws_to_roughly_unit_spread():
    # the chart spans one sigma in log, so the flow coordinate comes out near unit std
    # and centred, the tails reaching a few units out rather than staying in a box
    prior = ChirpMass()
    draws = jax.vmap(prior)(jr.split(jr.key(1), DRAWS))
    points = jax.vmap(prior.chart.forward)(draws)
    assert abs(float(points.mean())) < 0.1
    assert 0.8 < float(points.std()) < 1.2


def test_the_distribution_peaks_where_korol_2022_puts_it():
    # a flat secondary on [0.15, m1] under the Kepler et al. 2015 primary mass
    # function leaves a broad chirp mass peaked near 0.45 Msun, with a median
    # around 0.43 and only per-cent-level weight below 0.25
    draws = jax.vmap(ChirpMass())(jr.split(jr.key(2), DRAWS))
    assert 0.40 < jnp.median(draws) < 0.46
    assert 0.005 < jnp.mean(draws < 0.25) < 0.03
    assert jnp.mean(draws > 0.7) < 0.05


def test_a_narrower_component_mass_range_narrows_the_chirp_mass():
    wide = jax.vmap(ChirpMass())(jr.split(jr.key(3), DRAWS))
    narrow = jax.vmap(ChirpMass(m_min=0.3, m_max=0.8))(jr.split(jr.key(3), DRAWS))
    assert narrow.std() < wide.std()
    assert jnp.all(narrow >= 0.3 * 2**-0.2)
