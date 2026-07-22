import jax.numpy as jnp
import jax.random as jr

from canna.priors import (
    Normal,
    Product as ProductPrior,
    Set as SetPrior,
    PeriodicUniform,
    Isotropic,
)

# --- ProductPrior ---


def test_product_prior_geometry_is_product_of_blocks():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    assert prod.geometry.local_geometries == (p1.geometry, p2.geometry)


def test_product_prior_chart_is_product_of_blocks():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    assert prod.chart.local_charts == (p1.chart, p2.chart)


def test_product_prior_named_kwargs_order():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, block=p2)
    assert prod.local_priors == (p1, p2)


def test_product_prior_sample_shape_matches_summed_physical_dim():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    sample = prod(jr.key(0))
    assert sample.shape == (prod.chart.physical_dim,)
    assert sample.shape == (5,)


def test_product_prior_sample_forward_shape_matches_geometry_dim():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    sample = prod(jr.key(0))
    embedded = prod.chart.forward(sample)
    assert embedded.shape == (prod.geometry.dim,)


def test_product_prior_deterministic_given_key():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    a = prod(jr.key(0))
    b = prod(jr.key(0))
    assert jnp.array_equal(a, b)


def test_product_prior_distinct_keys_give_distinct_samples():
    p1 = Normal(mean=jnp.zeros(2), cov=jnp.eye(2))
    p2 = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    prod = ProductPrior(p1, p2)
    a = prod(jr.key(0))
    b = prod(jr.key(1))
    assert not jnp.array_equal(a, b)


def test_product_prior_single_block_matches_raw_prior_dim():
    p1 = Normal(mean=jnp.zeros(4), cov=jnp.eye(4))
    prod = ProductPrior(p1)
    sample = prod(jr.key(0))
    assert sample.shape == (4,)


# --- SetPrior ---


def test_set_prior_sample_shape():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=5)
    sample = sp(jr.key(0))
    assert sample.shape == (5, 3)


def test_set_prior_degenerate_singleton():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=1)
    sample = sp(jr.key(0))
    assert sample.shape == (1, 3)


def test_set_prior_degenerate_empty():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=0)
    sample = sp(jr.key(0))
    assert sample.shape == (0, 3)


def test_set_prior_rows_are_not_identical_draws():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=4)
    sample = sp(jr.key(0))
    assert not jnp.allclose(sample[0], sample[1])
    assert not jnp.allclose(sample[1], sample[2])


def test_set_prior_deterministic_given_key():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=4)
    a = sp(jr.key(0))
    b = sp(jr.key(0))
    assert jnp.array_equal(a, b)


def test_set_prior_geometry_wraps_local_geometry():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=4)
    assert sp.geometry.local_geometry == local.geometry


def test_set_prior_chart_reuses_local_chart_unwrapped():
    local = Normal(mean=jnp.zeros(3), cov=jnp.eye(3))
    sp = SetPrior(local, size=4)
    assert sp.chart == local.chart


def test_set_prior_chart_applies_over_set_axis():
    local = Normal(mean=jnp.zeros(1), cov=jnp.eye(1))
    sp = SetPrior(local, size=6)
    sample = sp(jr.key(0))
    embedded = sp.chart.forward(sample)
    assert embedded.shape[0] == 6


# --- curved local priors ---


def test_product_prior_chart_flow_dim_matches_geometry_dim_with_curved_blocks():
    p1 = PeriodicUniform(period=jnp.array([2 * jnp.pi]))
    p2 = Isotropic(dim=2, radius=jnp.array(1.0))
    prod = ProductPrior(p1, p2)
    assert prod.chart.flow_dim == prod.geometry.dim


def test_product_prior_sample_embeds_correctly_with_curved_blocks():
    p1 = PeriodicUniform(period=jnp.array([2 * jnp.pi]))
    p2 = Isotropic(dim=2, radius=jnp.array(1.0))
    prod = ProductPrior(p1, p2)
    sample = prod(jr.key(0))
    embedded = prod.chart.forward(sample)
    assert embedded.shape == (prod.geometry.dim,)


def test_set_prior_over_curved_local_chart_flow_dim_matches_geometry_dim():
    local = Isotropic(dim=2, radius=jnp.array(1.0))
    sp = SetPrior(local, size=4)
    assert sp.chart.flow_dim == sp.geometry.dim


def test_set_prior_over_curved_local_sample_and_embedding_shapes():
    local = Isotropic(dim=2, radius=jnp.array(1.0))
    sp = SetPrior(local, size=5)
    sample = sp(jr.key(0))
    assert sample.shape == (5, 2)
    embedded = sp.chart.forward(sample)
    assert embedded.shape == (5, 3)
