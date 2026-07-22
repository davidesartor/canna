from abc import abstractmethod
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx


class Chart[Physical: Array, Point: Array](eqx.Module):
    """Invertible map between physical units and manifold coordinates."""

    @abstractmethod
    def forward(self, p: Physical) -> Point: ...

    @abstractmethod
    def backward(self, x: Point) -> Physical: ...


class Affine(Chart):
    """Affine chart: p = scale @ x + shift."""

    shift: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=0.0)
    scale: Float[Array, "D"] | Float[Array, "D D"] = eqx.field(
        converter=jnp.array, default=1.0
    )

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... D"]:
        if self.scale.ndim < 2:
            return self.scale * p + self.shift
        return jnp.einsum("ij,...j->...i", self.scale, p) + self.shift

    def backward(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        if self.scale.ndim < 2:
            return (x - self.shift) / self.scale
        return jnp.linalg.solve(self.scale, (x - self.shift)[..., None])[..., 0]


class LogAffine(Chart):
    """Affine chart in log-space: p = exp(scale @ x + shift)."""

    shift: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=0.0)
    scale: Float[Array, "D"] | Float[Array, "D D"] = eqx.field(
        converter=jnp.array, default=1.0
    )

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... D"]:
        if self.scale.ndim < 2:
            return self.scale * jnp.log(p) + self.shift
        return jnp.einsum("ij,...j->...i", self.scale, jnp.log(p)) + self.shift

    def backward(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        if self.scale.ndim < 2:
            return jnp.exp((x - self.shift) / self.scale)
        return jnp.exp(
            jnp.linalg.solve(self.scale, (x - self.shift)[..., None])[..., 0]
        )
