import equinox as eqx
import jax.numpy as jnp

from canna.charts import Affine, Chart, Product as ProductChart, Periodic
from canna.charts import Spherical as SphericalChart


class _ScaleChart(Chart):
    """Toy chart: pure scaling, square (physical_dim == flow_dim), true inverse."""

    physical_dim: int = eqx.field(static=True, default=1)
    flow_dim: int = eqx.field(static=True, default=1)
    factor: float = eqx.field(static=True, default=1.0)

    def forward(self, p):
        return p * self.factor

    def backward(self, x):
        return x / self.factor


class _EmbedChart(Chart):
    """Toy chart: physical_dim != flow_dim, e.g. embedding a scalar as a duplicated pair."""

    physical_dim: int = eqx.field(static=True, default=1)
    flow_dim: int = eqx.field(static=True, default=2)

    def forward(self, p):
        return jnp.concatenate([p, p], axis=-1)

    def backward(self, x):
        return x[..., :1]


def test_product_chart_physical_dim_is_sum_of_blocks():
    c = ProductChart(_ScaleChart(physical_dim=2, flow_dim=2, factor=1.0), _EmbedChart())
    assert c.physical_dim == 2 + 1


def test_product_chart_flow_dim_is_sum_of_blocks():
    c = ProductChart(_ScaleChart(physical_dim=2, flow_dim=2, factor=1.0), _EmbedChart())
    assert c.flow_dim == 2 + 2


def test_product_chart_flow_dim_can_differ_from_physical_dim():
    c = ProductChart(_EmbedChart(), _EmbedChart())
    assert c.physical_dim == 2
    assert c.flow_dim == 4


def test_product_chart_forward_does_not_mix_blocks():
    a = _ScaleChart(physical_dim=1, flow_dim=1, factor=2.0)
    b = _ScaleChart(physical_dim=1, flow_dim=1, factor=5.0)
    c = ProductChart(a, b)
    p = jnp.array([1.0, 3.0])
    out = c.forward(p)
    assert jnp.allclose(out, jnp.array([2.0, 15.0]))


def test_product_chart_forward_output_shape_with_dim_changing_blocks():
    c = ProductChart(_EmbedChart(), _ScaleChart(physical_dim=1, flow_dim=1, factor=1.0))
    p = jnp.array([7.0, 9.0])
    out = c.forward(p)
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.array([7.0, 7.0, 9.0]))


def test_product_chart_roundtrip_square_blocks():
    a = _ScaleChart(physical_dim=1, flow_dim=1, factor=2.0)
    b = _ScaleChart(physical_dim=1, flow_dim=1, factor=0.5)
    c = ProductChart(a, b)
    p = jnp.array([3.0, -4.0])
    assert jnp.allclose(c.backward(c.forward(p)), p)


def test_product_chart_roundtrip_affine_blocks():
    a = Affine(shift=jnp.array(1.0), scale=jnp.array(2.0))
    b = Affine(shift=jnp.array(-1.0), scale=jnp.array(3.0))
    c = ProductChart(a, b)
    p = jnp.array([2.0, 5.0])
    assert jnp.allclose(c.backward(c.forward(p)), p)


def test_product_chart_batch_leading_axes():
    a = _ScaleChart(physical_dim=1, flow_dim=1, factor=2.0)
    b = _EmbedChart()
    c = ProductChart(a, b)
    p = jnp.zeros((4, 3, 2))
    out = c.forward(p)
    assert out.shape == (4, 3, 3)


def test_product_chart_single_block_matches_raw_chart():
    a = _ScaleChart(physical_dim=1, flow_dim=1, factor=3.0)
    c = ProductChart(a)
    p = jnp.array([2.0])
    assert jnp.allclose(c.forward(p), a.forward(p))


def test_product_chart_named_kwargs_order():
    a = _ScaleChart(physical_dim=1, flow_dim=1, factor=2.0)
    b = _ScaleChart(physical_dim=1, flow_dim=1, factor=3.0)
    c = ProductChart(a, second=b)
    assert c.local_charts == (a, b)


# --- real dim-changing charts (Periodic, Spherical) ---


def test_product_chart_roundtrip_periodic_and_spherical_blocks():
    a = Periodic(period=jnp.array([2 * jnp.pi]))
    b = SphericalChart(physical_dim=2, radius=jnp.array(1.0))
    c = ProductChart(a, b)
    assert c.physical_dim == 1 + 2
    assert c.flow_dim == 2 + 3
    p = jnp.array([1.0, 0.5, 0.3])
    recovered = c.backward(c.forward(p))
    assert jnp.allclose(recovered, p, atol=1e-5)
