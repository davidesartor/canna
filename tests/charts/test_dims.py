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


def test_affine_default_dims():
    chart = Affine()
    assert chart.physical_dim == 1
    assert chart.flow_dim == 1


def test_affine_dims_from_vector_shift():
    chart = Affine(shift=jnp.zeros(3))
    assert chart.physical_dim == 3
    assert chart.flow_dim == 3


def test_affine_dims_from_vector_scale():
    chart = Affine(scale=jnp.ones(4))
    assert chart.physical_dim == 4
    assert chart.flow_dim == 4


def test_affine_dims_from_matrix_scale():
    chart = Affine(scale=jnp.eye(5))
    assert chart.physical_dim == 5
    assert chart.flow_dim == 5


def test_affine_physical_dim_equals_flow_dim():
    for chart in [Affine(), Affine(shift=jnp.zeros(2)), Affine(scale=jnp.eye(6))]:
        assert chart.physical_dim == chart.flow_dim


def test_affine_forward_output_last_axis_matches_flow_dim():
    chart = Affine(shift=jnp.zeros(3))
    out = chart.forward(jnp.ones(3))
    assert out.shape[-1] == chart.flow_dim


def test_affine_forward_output_matches_flow_dim_with_leading_batch():
    chart = Affine(shift=jnp.zeros(3))
    out = chart.forward(jnp.ones((9, 3)))
    assert out.shape[-1] == chart.flow_dim


def test_affine_backward_output_last_axis_matches_physical_dim():
    chart = Affine(scale=jnp.eye(4))
    out = chart.backward(jnp.ones(4))
    assert out.shape[-1] == chart.physical_dim


def test_logaffine_default_dims():
    chart = LogAffine()
    assert chart.physical_dim == 1
    assert chart.flow_dim == 1


def test_logaffine_dims_from_matrix_scale():
    chart = LogAffine(scale=jnp.eye(3))
    assert chart.physical_dim == 3
    assert chart.flow_dim == 3


def test_logaffine_physical_dim_equals_flow_dim():
    chart = LogAffine(shift=jnp.zeros(5))
    assert chart.physical_dim == chart.flow_dim == 5


def test_periodic_default_dims():
    chart = Periodic()
    assert chart.physical_dim == 1
    assert chart.flow_dim == 2


def test_periodic_dims_multi_angle():
    chart = Periodic(period=jnp.array([1.0, 2.0, 3.0]))
    assert chart.physical_dim == 3
    assert chart.flow_dim == 6


def test_periodic_flow_dim_is_twice_physical_dim():
    for chart in [Periodic(), Periodic(period=jnp.array([1.0, 2.0]))]:
        assert chart.flow_dim == 2 * chart.physical_dim


def test_periodic_forward_output_last_axis_matches_flow_dim():
    chart = Periodic(period=jnp.array([1.0, 2.0, 3.0]))
    out = chart.forward(jnp.ones(3))
    assert out.shape[-1] == chart.flow_dim


def test_periodic_backward_output_last_axis_matches_physical_dim():
    chart = Periodic(period=jnp.array([1.0, 2.0, 3.0]))
    out = chart.backward(jnp.ones(6))
    assert out.shape[-1] == chart.physical_dim


def test_spherical_chart_default_dims():
    chart = SphericalChart()
    assert chart.physical_dim == 2
    assert chart.flow_dim == 3


def test_spherical_chart_physical_dim_one():
    chart = SphericalChart(physical_dim=1)
    assert chart.flow_dim == 2


def test_spherical_chart_flow_dim_is_physical_plus_one():
    for chart in [
        SphericalChart(),
        SphericalChart(physical_dim=1),
        SphericalChart(physical_dim=5),
    ]:
        assert chart.flow_dim == chart.physical_dim + 1


def test_spherical_chart_forward_output_last_axis_matches_flow_dim():
    chart = SphericalChart(physical_dim=4)
    out = chart.forward(jnp.ones(4))
    assert out.shape[-1] == chart.flow_dim


def test_spherical_chart_backward_output_last_axis_matches_physical_dim():
    chart = SphericalChart(physical_dim=4)
    out = chart.backward(jnp.ones(5))
    assert out.shape[-1] == chart.physical_dim


def test_product_chart_dims_sum_leaves():
    prod = ProductChart(
        Affine(shift=jnp.zeros(2)),
        Periodic(period=jnp.ones(2)),
        SphericalChart(physical_dim=3),
    )
    assert prod.physical_dim == 2 + 2 + 3
    assert prod.flow_dim == 2 + 4 + 4


def test_product_chart_dims_single_leaf():
    prod = ProductChart(Affine(shift=jnp.zeros(3)))
    assert prod.physical_dim == 3
    assert prod.flow_dim == 3


def test_product_chart_dims_named_kwargs_same_as_positional():
    named = ProductChart(a=Affine(shift=jnp.zeros(2)), b=Periodic(period=jnp.ones(2)))
    positional = ProductChart(Affine(shift=jnp.zeros(2)), Periodic(period=jnp.ones(2)))
    assert named.physical_dim == positional.physical_dim
    assert named.flow_dim == positional.flow_dim


def test_product_chart_forward_output_last_axis_matches_flow_dim():
    prod = ProductChart(Affine(shift=jnp.zeros(2)), SphericalChart(physical_dim=3))
    out = prod.forward(jnp.ones(2 + 3))
    assert out.shape[-1] == prod.flow_dim


@pytest.mark.parametrize(
    "chart",
    [
        Affine(shift=jnp.zeros(2)),
        LogAffine(shift=jnp.zeros(2)),
        Periodic(period=jnp.ones(2)),
        SphericalChart(physical_dim=2),
    ],
)
def test_chart_dim_fields_are_python_int(chart):
    assert isinstance(chart.physical_dim, int)
    assert isinstance(chart.flow_dim, int)


def test_product_chart_dim_fields_are_python_int():
    prod = ProductChart(Affine(shift=jnp.zeros(2)), Periodic(period=jnp.ones(2)))
    assert isinstance(prod.physical_dim, int)
    assert isinstance(prod.flow_dim, int)
