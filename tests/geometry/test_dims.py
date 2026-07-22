import jax.numpy as jnp
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
from canna.priors import PeriodicUniform, Isotropic
from canna.priors import Product as ProductPrior
from canna.priors import Set as SetPrior


def test_euclidean_dim_default():
    assert Euclidean().dim == 1


def test_euclidean_dim_explicit():
    assert Euclidean(dim=5).dim == 5


def test_bounded_dim_default():
    assert Bounded().dim == 1


def test_bounded_dim_explicit():
    assert Bounded(dim=3).dim == 3


def test_reflected_dim_default():
    assert Reflected().dim == 1


def test_reflected_dim_explicit():
    assert Reflected(dim=4).dim == 4


def test_toroidal_dim_default():
    assert ToroidalGeometry().dim == 2


def test_toroidal_dim_explicit():
    assert ToroidalGeometry(dim=6).dim == 6


def test_spherical_geom_dim_default():
    assert SphericalGeometry().dim == 2


def test_spherical_geom_dim_explicit():
    assert SphericalGeometry(dim=4).dim == 4


@pytest.mark.parametrize(
    "geom",
    [
        Euclidean(dim=3),
        Bounded(dim=2),
        Reflected(dim=2),
        ToroidalGeometry(dim=4),
        SphericalGeometry(dim=4),
    ],
)
def test_leaf_geometry_dim_is_python_int(geom):
    assert isinstance(geom.dim, int)


def test_leaf_geometry_log_map_output_last_axis_matches_dim():
    geom = Euclidean(dim=3)
    x0 = jnp.zeros((3,))
    x1 = jnp.ones((3,))
    out = geom.log_map(x0, x1)
    assert out.shape[-1] == geom.dim


def test_leaf_geometry_log_map_output_matches_dim_with_leading_batch():
    geom = Euclidean(dim=3)
    x0 = jnp.zeros((7, 3))
    x1 = jnp.ones((7, 3))
    out = geom.log_map(x0, x1)
    assert out.shape[-1] == geom.dim


def test_product_geometry_dim_sums_leaves():
    prod = ProductGeometry(
        Euclidean(dim=2), ToroidalGeometry(dim=4), SphericalGeometry(dim=3)
    )
    assert prod.dim == 2 + 4 + 3


def test_product_geometry_dim_single_leaf():
    prod = ProductGeometry(Euclidean(dim=5))
    assert prod.dim == 5


def test_product_geometry_dim_named_kwargs_same_as_positional():
    named = ProductGeometry(a=Euclidean(dim=2), b=ToroidalGeometry(dim=4))
    positional = ProductGeometry(Euclidean(dim=2), ToroidalGeometry(dim=4))
    assert named.dim == positional.dim == 6


def test_product_geometry_dim_is_python_int():
    prod = ProductGeometry(Euclidean(dim=2), ToroidalGeometry(dim=4))
    assert isinstance(prod.dim, int)


def test_set_geometry_dim_equals_local_geometry_dim():
    local = Euclidean(dim=3)
    setgeom = SetGeometry(local_geometry=local)
    assert setgeom.dim == local.dim


def test_set_geometry_dim_independent_of_brute_force_limit():
    local = Euclidean(dim=3)
    a = SetGeometry(local_geometry=local, brute_force_limit=1)
    b = SetGeometry(local_geometry=local, brute_force_limit=10)
    assert a.dim == b.dim == 3


def test_set_geometry_dim_is_python_int():
    setgeom = SetGeometry(local_geometry=Euclidean(dim=3))
    assert isinstance(setgeom.dim, int)


def test_set_geometry_log_map_last_axis_matches_dim_with_set_axis():
    local = Euclidean(dim=3)
    setgeom = SetGeometry(local_geometry=local)
    x0 = jnp.zeros((5, 3))
    x1 = jnp.ones((5, 3))
    out = setgeom.log_map(x0, x1)
    assert out.shape[-1] == setgeom.dim
