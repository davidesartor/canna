from abc import abstractmethod
from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from . import geometries, charts


class Prior[Physical: Array](eqx.Module):
    """A distribution over one parameter block, carrying its own geometry and chart."""

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> Physical: ...

    @property
    @abstractmethod
    def geometry(self) -> geometries.Geometry: ...

    @property
    @abstractmethod
    def chart(self) -> charts.Chart: ...


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
        return geometries.Euclidean()

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
        return geometries.Euclidean()

    @property
    def chart(self) -> charts.LogAffine:
        if self.cov.ndim < 2:
            scale = 1 / jnp.sqrt(self.cov)
            shift = -scale * self.mean
        else:
            scale = jnp.linalg.inv(jnp.linalg.cholesky(self.cov))
            shift = -scale @ self.mean
        return charts.LogAffine(shift=shift, scale=scale)