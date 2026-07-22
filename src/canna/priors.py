from abc import abstractmethod
from typing import Callable, Optional
from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from . import geometries, charts


class Prior[Physical: Array](eqx.Module):
    """A distribution over one parameter block, carrying its own geometry and chart."""

    geometry: eqx.AbstractVar[geometries.Geometry]
    chart: eqx.AbstractVar[charts.Chart]

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> Physical: ...


class Normal(Prior):
    """Gaussian prior over R^d.
    Embeds to R^d with Euclidean geometry.
    Uses an Affine chart to whiten the distribution.
    """

    mean: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=0.0)
    cov: Float[Array, "D"] | Float[Array, "D D"] = eqx.field(
        converter=jnp.asarray, default=1.0
    )

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        if self.cov.ndim < 2:
            z = jr.normal(key, self.mean.shape)
            return self.mean + jnp.sqrt(self.cov) * z
        return jr.multivariate_normal(key, self.mean, self.cov)

    @property
    def geometry(self) -> geometries.Euclidean:
        return geometries.Euclidean(self.mean.shape[-1])

    @property
    def chart(self) -> charts.Affine:
        if self.cov.ndim < 2:
            scale = 1 / jnp.sqrt(self.cov)
            shift = -scale * self.mean
        else:
            scale = jnp.linalg.inv(jnp.linalg.cholesky(self.cov))
            shift = -scale @ self.mean
        return charts.Affine(shift=shift, scale=scale)


class LogNormal(Prior):
    """Log-normal prior over R^d.
    Embeds to R^d with Euclidean geometry.
    Uses a LogAffine chart to whiten the distribution.
    """

    mean: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=0.0)
    cov: Float[Array, "D"] | Float[Array, "D D"] = eqx.field(
        converter=jnp.asarray, default=1.0
    )

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        if self.cov.ndim < 2:
            z = jr.normal(key, self.mean.shape)
            return jnp.exp(self.mean + jnp.sqrt(self.cov) * z)
        return jnp.exp(jr.multivariate_normal(key, self.mean, self.cov))

    @property
    def geometry(self) -> geometries.Euclidean:
        return geometries.Euclidean(self.mean.shape[-1])

    @property
    def chart(self) -> charts.LogAffine:
        if self.cov.ndim < 2:
            scale = 1 / jnp.sqrt(self.cov)
            shift = -scale * self.mean
        else:
            scale = jnp.linalg.inv(jnp.linalg.cholesky(self.cov))
            shift = -scale @ self.mean
        return charts.LogAffine(shift=shift, scale=scale)


class Uniform(Prior):
    """Uniform prior over a box [low, high].
    Embeds to the box [-1, 1]^d with Bounded geometry.
    Uses an Affine chart keeping the distribution uniform.
    """

    low: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=0.0)
    high: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=1.0)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return jr.uniform(key, self.low.shape, minval=self.low, maxval=self.high)

    @property
    def geometry(self) -> geometries.Bounded:
        return geometries.Bounded(self.low.shape[-1])

    @property
    def chart(self) -> charts.Affine:
        scale = 2 / (self.high - self.low)
        shift = -scale * (self.low + self.high) / 2
        return charts.Affine(shift=shift, scale=scale)


class LogUniform(Prior):
    """Log-uniform prior over a box [low, high].
    Embeds to the box [-1, 1]^d with Bounded geometry.
    Uses a LogAffine chart keeping the distribution log-uniform.
    """

    low: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d)
    high: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        return jnp.exp(jr.uniform(key, self.low.shape, minval=log_low, maxval=log_high))

    @property
    def geometry(self) -> geometries.Bounded:
        return geometries.Bounded(self.low.shape[-1])

    @property
    def chart(self) -> charts.LogAffine:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        scale = 2 / (log_high - log_low)
        shift = -scale * (log_low + log_high) / 2
        return charts.LogAffine(shift=shift, scale=scale)


class Cosine(Prior):
    """Prior with density proportional to cos over [-pi/2, pi/2] (declination).
    Embeds to the box [-1, 1]^d with Bounded geometry.
    Uses an Affine chart to rescale the distribution.
    """

    dim: int = eqx.field(static=True, default=1)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        sine = jr.uniform(key, (self.dim,), minval=-1.0, maxval=1.0)
        return jnp.arcsin(sine)

    @property
    def geometry(self) -> geometries.Bounded:
        return geometries.Bounded(self.dim)

    @property
    def chart(self) -> charts.Affine:
        return charts.Affine(
            shift=jnp.zeros(self.dim),
            scale=2 / jnp.pi * jnp.ones(self.dim),
        )


class Sine(Prior):
    """Prior with density proportional to sin over [0, pi] (inclination).
    Embeds to the box [-1, 1]^d with Reflected geometry.
    Uses an Affine chart to to rescale the distribution.
    """

    dim: int = eqx.field(static=True, default=1)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        cosine = jr.uniform(key, (self.dim,), minval=-1.0, maxval=1.0)
        return jnp.arccos(cosine)

    @property
    def geometry(self) -> geometries.Reflected:
        return geometries.Reflected(self.dim)

    @property
    def chart(self) -> charts.Affine:
        return charts.Affine(
            shift=-jnp.ones(self.dim),
            scale=2 / jnp.pi * jnp.ones(self.dim),
        )


class PeriodicUniform(Prior):
    """Uniform prior over angles [0, period).
    Embeds to a torus of (cos, sin) pairs with Toroidal geometry.
    Uses a Periodic chart keeping the distribution uniform.
    """

    period: Float[Array, "D"] = eqx.field(converter=jnp.atleast_1d, default=2 * jnp.pi)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return jr.uniform(key, self.period.shape, maxval=self.period)

    @property
    def geometry(self) -> geometries.Toroidal:
        return geometries.Toroidal(2 * self.period.shape[-1])

    @property
    def chart(self) -> charts.Periodic:
        return charts.Periodic(self.period)


class Isotropic(Prior):
    """Uniform prior over the surface of a sphere in R^{dim+1}.
    Embeds to that sphere with Spherical geometry.
    Uses a Spherical chart mapping physical angles to Cartesian coordinates.
    """

    dim: int = eqx.field(static=True, default=1)
    radius: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=1.0)

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        v = jr.normal(key, (self.dim + 1,))
        embedded = self.radius * v / jnp.linalg.norm(v)
        return self.chart.backward(embedded)

    @property
    def geometry(self) -> geometries.Spherical:
        return geometries.Spherical(self.dim + 1)

    @property
    def chart(self) -> charts.Spherical:
        return charts.Spherical(self.dim, self.radius)


class Product(Prior):
    """Independent priors over consecutive parameter blocks, drawn jointly.
    Embeds to the product of their geometries.
    Uses a Product chart applying each block's chart to its own slice.
    """

    local_priors: tuple[Prior, ...]

    def __init__(self, *priors: Prior, **named: Prior):
        self.local_priors = (*priors, *named.values())

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        keys = jr.split(key, len(self.local_priors))
        return jnp.concat([p(k) for p, k in zip(self.local_priors, keys)], axis=-1)

    @property
    def geometry(self) -> geometries.Product:
        return geometries.Product(*(p.geometry for p in self.local_priors))

    @property
    def chart(self) -> charts.Product:
        return charts.Product(*(p.chart for p in self.local_priors))


class Set(Prior):
    """Interchangeable draws from one prior, stacked along a set axis.
    Embeds to the Set geometry over that prior's geometry.
    Reuses the local chart, which acts on the trailing axis of every element.
    """

    local_prior: Prior
    size: int = eqx.field(static=True, default=1)
    rank: Optional[Callable[[Float[Array, "... X"]], Float[Array, "..."]]] = eqx.field(
        static=True, default=None
    )

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "S D"]:
        return eqx.filter_vmap(self.local_prior)(jr.split(key, self.size))

    @property
    def geometry(self) -> geometries.Set:
        return geometries.Set(self.local_prior.geometry, self.rank)

    @property
    def chart(self) -> charts.Chart:
        return self.local_prior.chart
