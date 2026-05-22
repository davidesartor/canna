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
    """Inverse CDF of cosine distribution. The pdf is proportional to cos(theta) stretched over the given range."""
    lo, hi = range
    return lo + jnp.arccos(2.0 * u - 1.0) * (hi - lo) / jnp.pi
