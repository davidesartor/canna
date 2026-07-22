"""LogAffine: x = exp(scale @ p + shift), log-space invertibility."""

import jax.numpy as jnp
import pytest

from canna.charts import LogAffine


def test_default_forward_is_log():
    chart = LogAffine()
    p = jnp.array([1.0, 2.7182818, 0.5])
    assert jnp.allclose(chart.forward(p), jnp.log(p), atol=1e-5)


def test_default_backward_is_exp():
    chart = LogAffine()
    x = jnp.array([0.0, 1.0, -0.5])
    assert jnp.allclose(chart.backward(x), jnp.exp(x), atol=1e-5)


def test_roundtrip_vector_scale_forward_backward():
    chart = LogAffine(
        shift=jnp.array([0.5, -0.3, 1.0]), scale=jnp.array([2.0, 0.5, 3.0])
    )
    p = jnp.array([0.3, 4.2, 7.0])
    assert jnp.allclose(chart.backward(chart.forward(p)), p, atol=1e-4)


def test_roundtrip_vector_scale_backward_forward():
    chart = LogAffine(
        shift=jnp.array([0.5, -0.3, 1.0]), scale=jnp.array([2.0, 0.5, 3.0])
    )
    x = jnp.array([0.3, -1.2, 0.7])
    assert jnp.allclose(chart.forward(chart.backward(x)), x, atol=1e-4)


def test_roundtrip_matrix_scale_forward_backward():
    scale = jnp.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])
    chart = LogAffine(shift=jnp.array([0.2, -0.1, 0.3]), scale=scale)
    p = jnp.array([0.3, 4.2, 7.0])
    assert jnp.allclose(chart.backward(chart.forward(p)), p, atol=1e-3)


def test_roundtrip_matrix_scale_backward_forward():
    scale = jnp.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])
    chart = LogAffine(shift=jnp.array([0.2, -0.1, 0.3]), scale=scale)
    x = jnp.array([0.3, -1.2, 0.7])
    assert jnp.allclose(chart.forward(chart.backward(x)), x, atol=1e-3)


def test_forward_matches_docstring_formula_vector_scale():
    shift = jnp.array([0.5, -0.3, 1.0])
    scale = jnp.array([2.0, 0.5, 3.0])
    chart = LogAffine(shift=shift, scale=scale)
    p = jnp.array([0.3, 4.2, 7.0])
    expected = scale * jnp.log(p) + shift
    assert jnp.allclose(chart.forward(p), expected, atol=1e-5)


def test_forward_matches_docstring_formula_matrix_scale():
    shift = jnp.array([0.2, -0.1, 0.3])
    scale = jnp.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])
    chart = LogAffine(shift=shift, scale=scale)
    p = jnp.array([0.3, 4.2, 7.0])
    expected = scale @ jnp.log(p) + shift
    assert jnp.allclose(chart.forward(p), expected, atol=1e-5)


def test_backward_matches_real_formula_vector_scale():
    shift = jnp.array([0.5, -0.3, 1.0])
    scale = jnp.array([2.0, 0.5, 3.0])
    chart = LogAffine(shift=shift, scale=scale)
    x = jnp.array([0.3, -1.2, 0.7])
    expected = jnp.exp((x - shift) / scale)
    assert jnp.allclose(chart.backward(x), expected, atol=1e-5)


def test_backward_matches_real_formula_matrix_scale():
    shift = jnp.array([0.2, -0.1, 0.3])
    scale = jnp.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])
    chart = LogAffine(shift=shift, scale=scale)
    x = jnp.array([0.3, -1.2, 0.7])
    expected = jnp.exp(jnp.linalg.inv(scale) @ (x - shift))
    assert jnp.allclose(chart.backward(x), expected, atol=1e-5)


def test_backward_output_strictly_positive():
    chart = LogAffine(shift=jnp.array([0.0, 5.0]), scale=jnp.array([1.0, -3.0]))
    x = jnp.array([-100.0, 100.0])
    out = chart.backward(x)
    assert jnp.all(out > 0.0)


def test_vector_scale_equals_diag_matrix_scale():
    shift = jnp.array([0.5, -0.3, 1.0])
    diag = jnp.array([2.0, 0.5, 3.0])
    chart_vec = LogAffine(shift=shift, scale=diag)
    chart_mat = LogAffine(shift=shift, scale=jnp.diag(diag))
    p = jnp.array([0.3, 4.2, 7.0])
    assert jnp.allclose(chart_vec.forward(p), chart_mat.forward(p), atol=1e-5)
    x = jnp.array([1.1, 0.2, -0.3])
    assert jnp.allclose(chart_vec.backward(x), chart_mat.backward(x), atol=1e-5)


def test_forward_negative_input_is_nan():
    chart = LogAffine()
    p = jnp.array([-1.0, 2.0])
    out = chart.forward(p)
    assert jnp.isnan(out[0])


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_forward_batch_shape_preserved_vector_scale(batch_shape):
    D = 3
    chart = LogAffine(
        shift=jnp.array([0.1, 0.0, -0.1]), scale=jnp.array([2.0, 1.0, 0.5])
    )
    p = jnp.ones(batch_shape + (D,)) * 2.0
    out = chart.forward(p)
    assert out.shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_roundtrip_batch_matrix_scale(batch_shape):
    D = 3
    scale = jnp.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])
    shift = jnp.array([0.2, -0.1, 0.3])
    chart = LogAffine(shift=shift, scale=scale)
    p = jnp.broadcast_to(jnp.array([0.3, 4.2, 7.0]), batch_shape + (D,))
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-3)


def test_forward_matrix_scale_batch_shape_mismatched_from_D():
    D = 5
    batch_shape = (7,)
    scale = jnp.eye(D) * 2.0 + 0.1
    chart = LogAffine(shift=jnp.zeros(D), scale=scale)
    p = jnp.ones(batch_shape + (D,)) * 2.0
    out = chart.forward(p)
    assert out.shape == batch_shape + (D,)


def test_backward_matrix_scale_batch_shape_mismatched_from_D():
    D = 5
    batch_shape = (7,)
    scale = jnp.eye(D) * 2.0 + 0.1
    chart = LogAffine(shift=jnp.zeros(D), scale=scale)
    x = jnp.zeros(batch_shape + (D,))
    out = chart.backward(x)
    assert out.shape == batch_shape + (D,)
