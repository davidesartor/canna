from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .constants import SUN_MASS_TIME


def fdot_from_chirp_mass(
    chirp_mass: Float[Array, "..."], f0: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Radiation-reaction chirp of a circular binary, for chirp mass in solar masses."""
    ratio = 96.0 / 5.0 * jnp.pi ** (8 / 3) * f0 ** (11 / 3)
    return ratio * (SUN_MASS_TIME * chirp_mass) ** (5 / 3)


def chirp_mass_from_fdot(
    fdot: Float[Array, "..."], f0: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Inverse of `fdot_from_chirp_mass`, in solar masses."""
    ratio = 5.0 / 96.0 * jnp.pi ** (-8 / 3) * f0 ** (-11 / 3)
    return (ratio * fdot) ** (3 / 5) / SUN_MASS_TIME


class ChirpMass(eqx.Module):
    """Chirp mass of a double white dwarf, drawn through its two component masses.

    This is the observationally driven Galactic DWD model of Korol et al. 2022
    (MNRAS 511, 5936; arXiv:2109.10972). The heavier component follows the
    single-white-dwarf mass function of Kepler et al. 2015 (MNRAS 446, 4078), a
    Gaussian mixture here truncated to [m_min, m_max]; the lighter one is uniform on
    [m_min, m1], the flat mass ratio close binaries are observed to follow (Moe &
    Di Stefano 2017). The induced chirp mass peaks near 0.45 Msun, with a median
    around 0.43 and a tail out to ~0.9. Korol et al. also swap the secondary for a
    uniform draw on [0.2, 1.2] when m1 falls in the ELM regime below 0.25 Msun; the
    default mixture never gets there, so that branch is left out.
    """

    m_min: float = eqx.field(static=True, default=0.15)
    m_max: float = eqx.field(static=True, default=1.4)
    weights: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.81, 0.14, 0.05)
    )
    means: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.65, 0.57, 0.81)
    )
    stds: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.044, 0.097, 0.187)
    )

    # the whitening constants of the induced log chirp mass, computed by quadrature
    log_mean: Float[Array, "1"] = eqx.field(init=False)
    log_std: Float[Array, "1"] = eqx.field(init=False)

    def __post_init__(self):
        # the induced log chirp mass moments have no closed form, so take them on a
        # grid over the truncated primary mixture and the flat mass ratio
        m1 = jnp.linspace(self.m_min, self.m_max, 256)[:, None]
        ratio = jnp.linspace(0.0, 1.0, 256)
        m2 = self.m_min + ratio * (m1 - self.m_min)
        log_mc = 0.6 * jnp.log(m1 * m2) - 0.2 * jnp.log(m1 + m2)

        # equinox runs the field converters after __post_init__, so convert by hand here
        means, stds, weights = (
            jnp.asarray(f) for f in (self.means, self.stds, self.weights)
        )
        density = jax.scipy.stats.norm.pdf(m1[..., None], means, stds)
        weight = (density * weights).sum(-1)
        weight = weight / weight.sum() / ratio.size

        self.log_mean = jnp.atleast_1d((weight * log_mc).sum())
        self.log_std = jnp.atleast_1d(
            jnp.sqrt((weight * (log_mc - self.log_mean) ** 2).sum())
        )

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "1"]:
        key_component, key_primary, key_secondary = jr.split(key, 3)

        # truncation leaves each component a different share of the mixture
        low = (self.m_min - self.means) / self.stds
        high = (self.m_max - self.means) / self.stds
        kept = jax.scipy.stats.norm.cdf(high) - jax.scipy.stats.norm.cdf(low)
        k = jr.choice(key_component, self.weights.size, p=self.weights * kept)

        z = jr.truncated_normal(key_primary, low[k], high[k])
        m1 = self.means[k] + self.stds[k] * z
        m2 = jr.uniform(key_secondary, minval=self.m_min, maxval=m1)
        return jnp.atleast_1d((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)

    @property
    def support(self) -> tuple[float, float]:
        # both extremes are equal-mass pairs, where the chirp mass is m / 2^(1/5)
        return self.m_min * 2**-0.2, self.m_max * 2**-0.2
