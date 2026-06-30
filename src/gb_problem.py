"""Shared definition of the single-source Galactic-Binary inference problem.

This module is the single source of truth used by ``train_Giorgio.py``,
``test_Giorgio.py`` and ``GB_inference.ipynb`` so that the network is trained,
tested and compared against ``jexplore`` on *exactly* the same problem:

* one Galactic Binary,
* four inferred parameters ``[f0, fdot, A, psi]`` (sky position and orientation
  fixed to :data:`FIXED_SKY`),
* the datastream and noise produced by :mod:`src.lisa` (``clean_signal`` /
  ``sample_noise`` / ``noise_psd``),
* the conditioning ``y`` is the WDM time-frequency image of the *datastream*
  (NOT the noisy ground-truth parameters used by the original sanity-check draft).

The four parameters are sampled in the unit cube ``u in [0, 1]^4`` and mapped to
physical values by :func:`u_to_physical`; the prior ranges match the jexplore
prior bounds in ``GB_inference.ipynb``.
"""

from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from wdm_transform.transforms import from_freq_to_wdm

from . import lisa, inverse_cdfs, networks

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
DT = lisa.SAMPLING_STEP_s          # Nyquist step (~167 s)
NCROP = 32                         # frequency-grid crop (multiple of 32 for WDM)
N_SLOW = 256                       # points for the slow TDI response (clean_signal `n`)
NT_WDM = 32                        # WDM time bins

PARAM_NAMES = ("f0", "fdot", "A", "psi")
X_DIM = 4

# Inferred-parameter prior ranges (== GB_inference.ipynb jexplore prior bounds).
# fdot / A caps are extended above lisa.prior_inverse_cdf to bracket a heavy-chirp,
# louder injection.
RANGES = {
    "f0": (1e-4, 3e-3),
    "fdot": (1e-22, 1e-15),
    "A": (1e-25, 1.7e-22),
    "psi": (0.0, float(jnp.pi)),
}

# Sky position / orientation held fixed (ra, dec, iota, phi0).
FIXED_SKY = dict(ra=1.0, dec=-0.5, iota=1.0, phi0=0.0)


def u_to_physical(u: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    """Map unit-cube samples ``u in [0,1]^4`` to physical ``[f0, fdot, A, psi]``."""
    f0 = inverse_cdfs.log_uniform(u[..., 0], RANGES["f0"])
    fdot = inverse_cdfs.log_uniform(u[..., 1], RANGES["fdot"])
    A = inverse_cdfs.log_uniform(u[..., 2], RANGES["A"])
    psi = inverse_cdfs.uniform(u[..., 3], RANGES["psi"])
    return jnp.stack([f0, fdot, A, psi], axis=-1)


def physical_to_u(phys: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    """Inverse of :func:`u_to_physical` (handy to place a known injection in the cube)."""
    def log_u(x, rng):
        lo, hi = rng
        return (jnp.log(x) - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
    def lin_u(x, rng):
        lo, hi = rng
        return (x - lo) / (hi - lo)
    return jnp.stack([
        log_u(phys[..., 0], RANGES["f0"]),
        log_u(phys[..., 1], RANGES["fdot"]),
        log_u(phys[..., 2], RANGES["A"]),
        lin_u(phys[..., 3], RANGES["psi"]),
    ], axis=-1)


def full_params(phys4: Float[Array, "4"]) -> Float[Array, "1 8"]:
    """Build the (1, 8) ``[f0,fdot,A,ra,dec,psi,iota,phi0]`` array lisa expects."""
    f0, fdot, A, psi = phys4
    s = FIXED_SKY
    return jnp.array([[f0, fdot, A, s["ra"], s["dec"], psi, s["iota"], s["phi0"]]])


def datastream_to_y(
    datastream: Float[Array, "F 3"], dt: float = DT
) -> Float[Array, "T C"]:
    """WDM time-frequency conditioning of a frequency-domain datastream.

    Same recipe as the original ``lisa.get_train_batch``: WDM transform of the
    A/E/T channels, channels folded into the feature axis, then sign-preserving
    log compression.
    """
    nf = datastream.shape[0] // NT_WDM
    y = from_freq_to_wdm(
        datastream.T, nt=NT_WDM, nf=nf, a=1.0 / 3.0, d=1.0, dt=dt, backend="jax"
    )
    y = rearrange(y, "c t f -> t (f c)")
    return jnp.where(y == 0, 0.0, jnp.sign(y) * jnp.log(jnp.abs(y)))


def get_train_batch(
    key: Key, batch_size: int, t_obs: float = lisa.MONTH_s
) -> tuple[
    Float[Array, "B 1 4"], Float[Array, "B 1 4"], Float[Array, "B"], Float[Array, "B T C"]
]:
    """Conditional flow-matching training batch.

    Returns ``(xt, dx, t, y)`` where, per sample, the base ``x0`` and target
    ``x1`` are drawn ``Uniform[0,1]^4``, ``xt = (1-t) x0 + t x1`` is the point on
    the straight probability path and ``dx = x1 - x0`` is the target velocity.
    ``y`` is the WDM image of the datastream generated from ``x1``.
    """
    def one(rng):
        k_x1, k_x0, k_t, k_noise = jr.split(rng, 4)
        x1 = jr.uniform(k_x1, (X_DIM,))
        x0 = jr.uniform(k_x0, (X_DIM,))
        t = jr.uniform(k_t)

        params = full_params(u_to_physical(x1))
        signal = lisa.clean_signal(params, t_obs=t_obs, dt=DT, n=N_SLOW, ncrop=NCROP)
        noise = lisa.sample_noise(k_noise, t_obs=t_obs, dt=DT, ncrop=NCROP)
        y = datastream_to_y(signal + noise)

        xt = (x0 + t * (x1 - x0))[None, :]   # (1, 4): a single GB "token"
        dx = (x1 - x0)[None, :]
        return xt, dx, t, y

    return jax.vmap(one)(jr.split(key, batch_size))


def build_flow(y_dim: int, *, hidden_dim: int, num_blocks: int, num_heads: int, key: Key):
    """Construct the MMDiT flow for the 4-parameter, single-source problem."""
    return networks.MMDiT(
        x_dim=X_DIM,
        y_dim=y_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        key=key,
    )


def sample_flow(
    flow, x0: Float[Array, "1 4"], y: Float[Array, "T C"], ode_steps: int = 16
) -> Float[Array, "1 4"]:
    """Integrate the learned velocity field from ``x0`` (t=0) to t=1 with RK4.

    Unlike ``MMDiT.push``, this does **not** wrap the state modulo 1 after each
    step — that ``x % 1`` hack corrupts the non-periodic parameters (f0, fdot, A,
    and psi, which are not periodic in the unit cube). We integrate freely and
    only clip the final sample back into the unit cube for safety.
    """
    def body(i, x):
        dt = 1.0 / ode_steps
        t = i * dt
        k1 = flow(x, y, t)
        k2 = flow(x + k1 * dt / 2, y, t + dt / 2)
        k3 = flow(x + k2 * dt / 2, y, t + dt / 2)
        k4 = flow(x + k3 * dt, y, t + dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    x = jax.lax.fori_loop(0, ode_steps, body, x0)
    return jnp.clip(x, 0.0, 1.0)
