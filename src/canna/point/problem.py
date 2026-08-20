from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class NoisyPoint(eqx.Module):
    """Gaussian point in R^D with a random full covariance, observed under additive noise."""

    seed: int = eqx.field(static=True, default=0)
    dim: int = eqx.field(static=True, default=2)

    prior_chol: Float[Array, "D D"] = eqx.field(init=False)
    noise_chol: Float[Array, "D D"] = eqx.field(init=False)

    def __post_init__(self):
        key_prior, key_noise = jr.split(jr.key(self.seed))
        A = jr.normal(key_prior, (self.dim, self.dim))
        self.prior_chol = jnp.linalg.cholesky(A @ A.T + 1e-3 * jnp.eye(self.dim))

        # unit diagonal, so the noise stays narrow next to the prior
        B = jr.normal(key_noise, (self.dim, self.dim))
        C = B @ B.T + 1e-3 * jnp.eye(self.dim)
        self.noise_chol = jnp.linalg.cholesky(
            C / jnp.sqrt(jnp.outer(jnp.diag(C), jnp.diag(C)))
        )

    @property
    def cov(self) -> Float[Array, "D D"]:
        return self.prior_chol @ self.prior_chol.T

    @property
    def noise_cov(self) -> Float[Array, "D D"]:
        return self.noise_chol @ self.noise_chol.T

    def physical_to_flow(self, p: Float[Array, "D"]) -> Float[Array, "D"]:
        return jnp.linalg.solve(self.prior_chol, p)

    def flow_to_physical(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.prior_chol @ x

    def log_map(
        self, x0: Float[Array, "D"], x1: Float[Array, "D"]
    ) -> Float[Array, "D"]:
        return x1 - x0

    def exp_map(self, x: Float[Array, "D"], v: Float[Array, "D"]) -> Float[Array, "D"]:
        return x + v

    def sample_physical(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return self.prior_chol @ jr.normal(key, (self.dim,))

    def sample_flow(self, key: Key[Array, ""]) -> Float[Array, "D"]:
        return jr.normal(key, (self.dim,))

    def sample_observation(
        self, key: Key[Array, ""], p: Float[Array, "D"]
    ) -> Float[Array, "D"]:
        return p + self.noise_chol @ jr.normal(key, (self.dim,))

    def preprocess(self, o: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.physical_to_flow(o)

    def log_likelihood(
        self, p: Float[Array, "D"], o: Float[Array, "D"]
    ) -> Float[Array, ""]:
        residual = jnp.linalg.solve(self.noise_chol, o - p)
        return -0.5 * residual @ residual
