import itertools
from abc import abstractmethod
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange


class Chart[Physical: Array, Point: Array](eqx.Module):
    """Invertible map between physical units and manifold coordinates."""

    physical_dim: eqx.AbstractVar[int]
    flow_dim: eqx.AbstractVar[int]

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

    @property
    def physical_dim(self) -> int:
        return jnp.broadcast_shapes(self.shift.shape, self.scale.shape[-1:])[-1]

    @property
    def flow_dim(self) -> int:
        return jnp.broadcast_shapes(self.shift.shape, self.scale.shape[-1:])[-1]

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

    @property
    def physical_dim(self) -> int:
        return jnp.broadcast_shapes(self.shift.shape, self.scale.shape[-1:])[-1]

    @property
    def flow_dim(self) -> int:
        return jnp.broadcast_shapes(self.shift.shape, self.scale.shape[-1:])[-1]

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


class Periodic(Chart):
    """Maps each angle to a (cos, sin) pair on the unit circle."""

    period: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=2 * jnp.pi)

    @property
    def physical_dim(self) -> int:
        return self.period.shape[-1]

    @property
    def flow_dim(self) -> int:
        return 2 * self.period.shape[-1]

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... D*2"]:
        angle = p / self.period * 2 * jnp.pi
        pairs = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
        return rearrange(pairs, "... d two -> ... (d two)")

    def backward(self, x: Float[Array, "... D*2"]) -> Float[Array, "... D"]:
        pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        angle = jnp.arctan2(pairs[..., 1], pairs[..., 0])
        return jnp.mod(angle, 2 * jnp.pi) / (2 * jnp.pi) * self.period


class Spherical(Chart):
    """Maps hyperspherical angles to Cartesian coordinates on a sphere in R^{D+1}."""

    physical_dim: int = eqx.field(static=True, default=2)
    radius: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=1.0)

    @property
    def flow_dim(self) -> int:
        return self.physical_dim + 1

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... D+1"]:
        D = p.shape[-1]
        point = jnp.zeros((*p.shape[:-1], D + 1), dtype=p.dtype)
        running = self.radius
        for i in range(D, 1, -1):
            latitude = p[..., i - 1]
            point = point.at[..., i].set(running * jnp.sin(latitude))
            running = running * jnp.cos(latitude)
        azimuth = p[..., 0]
        point = point.at[..., 1].set(running * jnp.sin(azimuth))
        point = point.at[..., 0].set(running * jnp.cos(azimuth))
        return point

    def backward(self, x: Float[Array, "... D+1"]) -> Float[Array, "... D"]:
        D = x.shape[-1] - 1
        p = jnp.zeros((*x.shape[:-1], D), dtype=x.dtype)
        for i in range(D, 1, -1):
            base_norm = jnp.linalg.norm(x[..., :i], axis=-1)
            p = p.at[..., i - 1].set(jnp.arctan2(x[..., i], base_norm))
        azimuth = jnp.mod(jnp.arctan2(x[..., 1], x[..., 0]), 2 * jnp.pi)
        return p.at[..., 0].set(azimuth)


class Product(Chart):
    """Product chart: each block of coordinates goes through its own chart."""

    local_charts: tuple[Chart, ...]

    def __init__(self, *charts: Chart, **named: Chart):
        self.local_charts = (*charts, *named.values())

    @property
    def physical_dim(self) -> int:
        return sum(c.physical_dim for c in self.local_charts)

    @property
    def flow_dim(self) -> int:
        return sum(c.flow_dim for c in self.local_charts)

    def forward(self, p: Float[Array, "... D"]) -> Float[Array, "... X"]:
        dims = tuple(c.physical_dim for c in self.local_charts)
        edges = list(itertools.accumulate(dims))[:-1]
        charted = zip(self.local_charts, jnp.split(p, edges, axis=-1))
        return jnp.concat([c.forward(b) for c, b in charted], axis=-1)

    def backward(self, x: Float[Array, "... X"]) -> Float[Array, "... D"]:
        dims = tuple(c.flow_dim for c in self.local_charts)
        edges = list(itertools.accumulate(dims))[:-1]
        charted = zip(self.local_charts, jnp.split(x, edges, axis=-1))
        return jnp.concat([c.backward(b) for c, b in charted], axis=-1)
