from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .. import charts, geometries, priors
from .base import Problem


class NoisyPoint(Problem):
    """Gaussian point in R^D with a random full covariance, observed under additive noise."""

    noise_std: float = eqx.field(static=True, default=0.1)
    prior: priors.Normal = eqx.field(init=False)

    def __init__(self, seed: int = 0, dim: int = 2, noise_std: float = 0.1):
        self.noise_std = noise_std
        A = jr.normal(jr.key(seed), (dim, dim))
        cov = A @ A.T + 1e-3 * jnp.eye(dim)
        self.prior = priors.Normal(mean=jnp.zeros(dim), cov=cov)

    @property
    def chart(self) -> charts.Affine:
        return self.prior.chart

    @property
    def geometry(self) -> geometries.Euclidean:
        return self.prior.geometry

    def sample_physical(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return self.prior(key)

    def sample_point(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return self.chart.forward(self.sample_physical(key))

    def sample_observation(
        self, key: Key[Array, ""], p: Float[Array, "... D"], clean: bool = False
    ) -> Float[Array, "... D"]:
        if clean:
            return p
        return p + self.noise_std * jr.normal(key, p.shape)

    def preprocess(self, o: Float[Array, "... D"]) -> Float[Array, "... D"]:
        return self.chart.forward(o)

    def log_likelihood(
        self, p: Float[Array, "... D"], o: Float[Array, "... D"]
    ) -> Float[Array, "..."]:
        return -0.5 * jnp.sum((o - p) ** 2, axis=-1) / self.noise_std**2
