from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class NoisyPoint(eqx.Module):
    """Gaussian point in R^D with a random full covariance, observed under additive noise."""

    seed: int = eqx.field(static=True, default=0)
    dim: int = eqx.field(static=True, default=2)
    noise_std: float = eqx.field(static=True, default=0.1)

    cov: Float[Array, "D D"] = eqx.field(init=False)
    whitening: Float[Array, "D D"] = eqx.field(init=False)

    def __post_init__(self):
        A = jr.normal(jr.key(self.seed), (self.dim, self.dim))
        self.cov = A @ A.T + 1e-3 * jnp.eye(self.dim)
        self.whitening = jnp.linalg.inv(jnp.linalg.cholesky(self.cov))

    def physical_to_flow(self, p: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.whitening @ p

    def flow_to_physical(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        return jnp.linalg.solve(self.whitening, x)

    def exp_map(self, x: Float[Array, "D"], v: Float[Array, "D"]) -> Float[Array, "D"]:
        return x + v

    def log_map(
        self, x0: Float[Array, "D"], x1: Float[Array, "D"]
    ) -> Float[Array, "D"]:
        return x1 - x0

    def geodesic(
        self, t: Float[Array, ""], x0: Float[Array, "D"], x1: Float[Array, "D"]
    ) -> Float[Array, "D"]:
        return self.exp_map(x0, t * self.log_map(x0, x1))

    def sample_physical(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return jr.multivariate_normal(key, jnp.zeros(self.dim), self.cov)

    def sample_point(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return self.physical_to_flow(self.sample_physical(key))

    def sample_observation(
        self, key: Key[Array, ""], p: Float[Array, "D"]
    ) -> Float[Array, "D"]:
        return p + self.noise_std * jr.normal(key, p.shape)

    def preprocess(self, o: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.physical_to_flow(o)

    def log_likelihood(
        self, p: Float[Array, "D"], o: Float[Array, "D"]
    ) -> Float[Array, ""]:
        return -0.5 * jnp.sum((o - p) ** 2) / self.noise_std**2
