import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.geometries import Euclidean, Bounded, Reflected
from canna.geometries import Toroidal as ToroidalGeometry
from canna.geometries import Spherical as SphericalGeometry
from canna.geometries import Product as ProductGeometry
from canna.geometries import Set as SetGeometry
from canna.charts import Affine, LogAffine, Periodic
from canna.charts import Spherical as SphericalChart
from canna.charts import Product as ProductChart
from canna.priors import Normal, LogNormal, Uniform, LogUniform, Cosine, Sine
from canna.problems.lisa import ChirpMass
from canna.priors import PeriodicUniform, Isotropic
from canna.priors import Product as ProductPrior
from canna.priors import Set as SetPrior

KEY = jr.key(0)


@pytest.mark.parametrize(
    "prior",
    [
        Normal(mean=jnp.zeros(3)),
        LogNormal(mean=jnp.zeros(2)),
        Uniform(low=jnp.zeros(2), high=jnp.ones(2)),
        LogUniform(low=jnp.ones(2), high=2 * jnp.ones(2)),
        ChirpMass(),
        Cosine(),
        Sine(),
        PeriodicUniform(period=jnp.array([1.0, 2.0])),
        Isotropic(),
    ],
)
def test_leaf_prior_chart_flow_dim_matches_geometry_dim(prior):
    assert prior.chart.flow_dim == prior.geometry.dim


@pytest.mark.parametrize(
    "prior,expected_physical_dim",
    [
        (Normal(mean=jnp.zeros(3)), 3),
        (LogNormal(mean=jnp.zeros(2)), 2),
        (Uniform(low=jnp.zeros(2), high=jnp.ones(2)), 2),
        (LogUniform(low=jnp.ones(4), high=2 * jnp.ones(4)), 4),
        (ChirpMass(), 1),
        (Cosine(dim=1), 1),
        (Sine(dim=1), 1),
        (PeriodicUniform(period=jnp.array([1.0, 2.0, 3.0])), 3),
        (Isotropic(dim=3), 3),
    ],
)
def test_leaf_prior_sample_shape_matches_chart_physical_dim(
    prior, expected_physical_dim
):
    assert prior.chart.physical_dim == expected_physical_dim
    sample = prior(KEY)
    assert sample.shape[-1] == prior.chart.physical_dim


def test_isotropic_chart_flow_dim_is_physical_dim_plus_one():
    prior = Isotropic(dim=3)
    assert prior.chart.flow_dim == prior.chart.physical_dim + 1


def test_periodicuniform_chart_flow_dim_is_twice_physical_dim():
    prior = PeriodicUniform(period=jnp.array([1.0, 2.0, 3.0]))
    assert prior.chart.flow_dim == 2 * prior.chart.physical_dim


def test_productprior_chart_flow_dim_matches_geometry_dim():
    prior = ProductPrior(
        Normal(mean=jnp.zeros(2)),
        PeriodicUniform(period=jnp.array([1.0, 2.0])),
        Isotropic(dim=2),
    )
    assert prior.chart.flow_dim == prior.geometry.dim


def test_productprior_geometry_dim_sums_local_geometry_dims():
    normal = Normal(mean=jnp.zeros(2))
    periodic = PeriodicUniform(period=jnp.array([1.0, 2.0]))
    isotropic = Isotropic(dim=2)
    prior = ProductPrior(normal, periodic, isotropic)
    assert (
        prior.geometry.dim
        == normal.geometry.dim + periodic.geometry.dim + isotropic.geometry.dim
    )


def test_productprior_chart_flow_dim_sums_local_chart_flow_dims():
    normal = Normal(mean=jnp.zeros(2))
    periodic = PeriodicUniform(period=jnp.array([1.0, 2.0]))
    isotropic = Isotropic(dim=2)
    prior = ProductPrior(normal, periodic, isotropic)
    assert (
        prior.chart.flow_dim
        == normal.chart.flow_dim + periodic.chart.flow_dim + isotropic.chart.flow_dim
    )


def test_productprior_chart_physical_dim_sums_local_physical_dims():
    normal = Normal(mean=jnp.zeros(2))
    isotropic = Isotropic(dim=3)
    prior = ProductPrior(normal, isotropic)
    assert (
        prior.chart.physical_dim
        == normal.chart.physical_dim + isotropic.chart.physical_dim
    )


def test_productprior_sample_shape_matches_chart_physical_dim():
    normal = Normal(mean=jnp.zeros(2))
    isotropic = Isotropic(dim=3)
    prior = ProductPrior(normal, isotropic)
    sample = prior(KEY)
    assert sample.shape[-1] == prior.chart.physical_dim


def test_productprior_single_leaf_dims_pass_through():
    normal = Normal(mean=jnp.zeros(4))
    prior = ProductPrior(normal)
    assert prior.chart.physical_dim == normal.chart.physical_dim
    assert prior.chart.flow_dim == normal.chart.flow_dim
    assert prior.geometry.dim == normal.geometry.dim


def test_setprior_chart_reuses_local_chart_dims():
    local = Normal(mean=jnp.zeros(3))
    prior = SetPrior(local_prior=local, size=5)
    assert prior.chart.physical_dim == local.chart.physical_dim
    assert prior.chart.flow_dim == local.chart.flow_dim


def test_setprior_geometry_dim_matches_local_geometry_dim():
    local = Normal(mean=jnp.zeros(3))
    prior = SetPrior(local_prior=local, size=5)
    assert prior.geometry.dim == local.geometry.dim


def test_setprior_chart_flow_dim_matches_geometry_dim():
    local = Normal(mean=jnp.zeros(3))
    prior = SetPrior(local_prior=local, size=5)
    assert prior.chart.flow_dim == prior.geometry.dim


def test_setprior_sample_shape_is_size_by_physical_dim():
    local = Normal(mean=jnp.zeros(3))
    prior = SetPrior(local_prior=local, size=5)
    sample = prior(KEY)
    assert sample.shape == (5, prior.chart.physical_dim)


def test_setprior_singleton_size_sample_shape():
    local = Normal(mean=jnp.zeros(2))
    prior = SetPrior(local_prior=local, size=1)
    sample = prior(KEY)
    assert sample.shape == (1, prior.chart.physical_dim)


def test_setprior_dims_independent_of_set_size():
    local = Normal(mean=jnp.zeros(2))
    small = SetPrior(local_prior=local, size=1)
    big = SetPrior(local_prior=local, size=20)
    assert small.chart.physical_dim == big.chart.physical_dim
    assert small.chart.flow_dim == big.chart.flow_dim
    assert small.geometry.dim == big.geometry.dim
