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



class Squash(Chart):
    """Squash chart: maps the box [low, high] onto R via arctanh."""

    low: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=-1.0)
    high: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=1.0)

    @property
    def physical_dim(self) -> int:
        return jnp.broadcast_shapes(self.low.shape, self.high.shape)[-1]

    @property
    def flow_dim(self) -> int:
        return jnp.broadcast_shapes(self.low.shape, self.high.shape)[-1]

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... D"]:
        return jnp.arctanh(2 * (p - self.low) / (self.high - self.low) - 1)

    def backward(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        return self.low + (self.high - self.low) * (jnp.tanh(x) + 1) / 2
