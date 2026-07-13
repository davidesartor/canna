from jaxtyping import Array, Float
import jax.numpy as jnp


def uniform(u: Float[Array, "..."], range: tuple[float, float]) -> Float[Array, "..."]:
    """Inverse CDF of a uniform distribution."""
    lo, hi = range
    return lo + u * (hi - lo)


def log_uniform(
    u: Float[Array, "..."], range: tuple[float, float]
) -> Float[Array, "..."]:
    """Inverse CDF of a log-uniform distribution."""
    lo, hi = range
    return jnp.exp(jnp.log(lo) + u * (jnp.log(hi) - jnp.log(lo)))


def cosine_pdf(
    u: Float[Array, "..."], range: tuple[float, float]
) -> Float[Array, "..."]:
    """Inverse CDF of a cosine distribution (pdf proportional to cos over the range)."""
    lo, hi = range
    return lo + jnp.arccos(2.0 * u - 1.0) * (hi - lo) / jnp.pi
