"""The flow's coordinates: 8 per source, all flat."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from .physics import (
    FDOT_STD,
    equatorial_to_galactic,
    galactic_scales,
    galactic_to_equatorial,
    log10_fdot_mean,
)
from .problem import LisaGB, amplitude_unit, f0_window
from .simulator import sample_physical

# per source: [f, fdot, galactic l, galactic b, z+ (2), z- (2)]
SKY = slice(2, 4)
GALACTIC_SCALE = galactic_scales()


def physical_to_flow(
    problem: LisaGB, p: Float[Array, "... S 8"], f: Float[Array, ""]
) -> Float[Array, "... S 8"]:
    """z± = A (1 ∓ sin iota)² / 4 exp(i (phi0 ± 2 psi)) in amplitude_unit, its radius warped by asinh.

    log10 fdot is standardised by the prior's conditional mean and std at the source's own f0.
    """
    f0, fdot, amp, sky_lon, sky_lat, psi, iota, phi0 = jnp.split(p, 8, axis=-1)
    log_f = jnp.log(jnp.stack(f0_window(problem, f)))
    amp_snr_1 = amplitude_unit(problem, f)
    l, b = equatorial_to_galactic(sky_lon, sky_lat)
    if problem.phase_reference == "middle":
        phi0 = phi0 - jnp.pi * f0 * problem.t_obs - jnp.pi * fdot * problem.t_obs**2 / 4
    phase_p, phase_m = phi0 + 2 * psi, phi0 - 2 * psi
    radius_p = jnp.arcsinh(amp / amp_snr_1 * jnp.square(1 - jnp.sin(iota)) / 4)
    radius_m = jnp.arcsinh(amp / amp_snr_1 * jnp.square(1 + jnp.sin(iota)) / 4)
    return jnp.concat(
        [
            (2 * jnp.log(f0) - log_f.sum()) / (log_f[1] - log_f[0]),
            (jnp.log10(fdot) - log10_fdot_mean(f0)) / FDOT_STD,
            l / GALACTIC_SCALE[0],
            b / GALACTIC_SCALE[1],
            radius_p * jnp.cos(phase_p),
            radius_p * jnp.sin(phase_p),
            radius_m * jnp.cos(phase_m),
            radius_m * jnp.sin(phase_m),
        ],
        axis=-1,
    )


def flow_to_physical(
    problem: LisaGB, x: Float[Array, "... S 8"], f: Float[Array, ""]
) -> Float[Array, "... S 8"]:
    log_f = jnp.log(jnp.stack(f0_window(problem, f)))
    amp_snr_1 = amplitude_unit(problem, f)
    x_f, x_fdot, l, b, re_p, im_p, re_m, im_m = jnp.split(x, 8, axis=-1)

    # |z±| = A (1 ∓ sin iota)² / 4, so sqrt|z+| + sqrt|z-| = sqrt A
    root_p = jnp.sqrt(jnp.sinh(jnp.hypot(re_p, im_p)))
    root_m = jnp.sqrt(jnp.sinh(jnp.hypot(re_m, im_m)))
    root_a = root_p + root_m
    sin_iota = (root_m - root_p) / jnp.maximum(root_a, jnp.finfo(x.dtype).tiny)
    # off the poles by a hair: arcsin's slope is infinite there
    iota = jnp.arcsin(jnp.clip(sin_iota, -1 + 1e-6, 1 - 1e-6))
    phase_p, phase_m = jnp.arctan2(im_p, re_p), jnp.arctan2(im_m, re_m)
    # one of the four (psi, phi0) the two phases stand for, all the same signal
    psi, phi0 = (phase_p - phase_m) / 4, (phase_p + phase_m) / 2
    f0 = jnp.exp((x_f * (log_f[1] - log_f[0]) + log_f.sum()) / 2)
    fdot = 10.0 ** (x_fdot * FDOT_STD + log10_fdot_mean(f0))
    sky_lon, sky_lat = galactic_to_equatorial(
        l * GALACTIC_SCALE[0], b * GALACTIC_SCALE[1]
    )
    if problem.phase_reference == "middle":
        phi0 = phi0 + jnp.pi * f0 * problem.t_obs + jnp.pi * fdot * problem.t_obs**2 / 4
    return jnp.concat(
        [
            f0,
            fdot,
            amp_snr_1 * jnp.square(root_a),
            sky_lon,
            sky_lat,
            jnp.mod(psi, 2 * jnp.pi),
            iota,
            jnp.mod(phi0, 2 * jnp.pi),
        ],
        axis=-1,
    )


def sample_flow(
    problem: LisaGB, key: Key[Array, ""], f: Float[Array, ""]
) -> Float[Array, "S 8"]:
    """A prior draw in flow coordinates: the flow's base distribution, its sources in no particular order."""
    return physical_to_flow(problem, sample_physical(problem, key, f), f)

