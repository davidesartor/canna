"""Forward WDM transform of a compact frequency band, in JAX.

A stripped-down replacement for `wdm_transform.transforms.from_freq_to_wdm_band`,
specialised to the one case this project needs: a contiguous block of WDM channels
computed from the matching contiguous span of one-sided Fourier bins.

The upstream kernel loops over channels in Python, so tracing unrolls it into a
graph that grows linearly with the channel count. At the sizes used here that is
~2M HLO instructions and hours of compilation. Everything below is batched: the
per-channel gather becomes a reshape, and the per-channel length-`ntimes` inverse
DFT becomes one small matmul, so the graph size is independent of `nfreq_bands`.
"""

import math

from jaxtyping import Array, Complex, Float
import jax.numpy as jnp

WINDOW_ROLLOFF = 1.0 / 3.0  # `a`: half-width of the phi-window's flat passband


def phi_window(ntimes: int, a: float = WINDOW_ROLLOFF) -> Float[Array, " t"]:
    """Cosine-tapered phi-window on the length-`ntimes` FFT frequency grid."""
    b = 1.0 - 2.0 * a
    half = ntimes // 2
    # fftfreq order: [0, 1, ..., half-1, -half, ..., -1], normalised to [-1, 1)
    l = (jnp.arange(ntimes) + half) % ntimes - half
    f = jnp.abs(2.0 * l / ntimes)
    taper = jnp.cos((jnp.pi / 2.0) * (f - a) / b)
    return jnp.where(f > a + b, 0.0, jnp.where(f > a, taper, 1.0)) * math.sqrt(
        2.0 / ntimes
    )


def from_freq_to_wdm_band(
    spectrum: Complex[Array, "... w"],
    *,
    ntimes: int,
    nfreq_bands: int,
    mmin: int = 0,
    a: float = WINDOW_ROLLOFF,
) -> Float[Array, "... t f"]:
    """WDM coefficients for channels `[mmin, mmin + nfreq_bands)`.

    Args:
        spectrum: One-sided Fourier samples spanning exactly the bins
            `[mmin * ntimes//2, (mmin + nfreq_bands) * ntimes//2)`, so the last
            axis has length `nfreq_bands * ntimes//2`. Leading axes are batch
            axes and are carried through untouched.
        ntimes: Number of WDM time rows; must be even.
        nfreq_bands: Number of WDM frequency channels to return.
        mmin: Index of the first returned channel in the full WDM grid. Only its
            value relative to 0 matters: channel 0 is the DC edge channel, which
            obeys a different formula from the interior ones.
        a: Window roll-off parameter.

    Returns:
        Real coefficients of shape `(..., ntimes, nfreq_bands)`.

    Note:
        The Nyquist edge channel (`m == nfreqs_wdm` in the full grid) is not
        implemented: it needs the two-sided spectrum, and no band this project
        transforms reaches the sampling Nyquist frequency. Requesting a span that
        includes it returns interior-channel values for it, which are wrong.

        Bins below `mmin * ntimes//2` are treated as zero. When `mmin > 0` the
        first returned channel legitimately draws on the block just below the
        span, so it is only correct if the signal really does vanish there.
    """
    if ntimes % 2 != 0:
        raise ValueError(f"ntimes must be even, got {ntimes}.")
    half = ntimes // 2
    if spectrum.shape[-1] != nfreq_bands * half:
        raise ValueError(
            f"spectrum last axis is {spectrum.shape[-1]}, expected "
            f"nfreq_bands * ntimes//2 = {nfreq_bands * half}."
        )

    cdtype = spectrum.dtype
    rdtype = jnp.zeros((), cdtype).real.dtype
    window = phi_window(ntimes, a).astype(rdtype)
    narr = jnp.arange(ntimes)

    # channel m draws on Fourier bins [(m-1)*half, (m+1)*half), i.e. two adjacent
    # half-blocks, so the whole per-channel gather is a reshape plus two slices
    blocks = spectrum.reshape(*spectrum.shape[:-1], nfreq_bands, half)
    if mmin > 0:
        # the block below the span is not supplied; it contributes zero
        blocks = jnp.concatenate([jnp.zeros_like(blocks[..., :1, :]), blocks], axis=-2)
    mid = jnp.concatenate([blocks[..., 1:, :], blocks[..., :-1, :]], axis=-1)
    mid = mid * window

    # ifft(x) * ntimes is the unnormalised inverse DFT; at these tiny `ntimes` a
    # matmul against the constant-folded kernel beats a batched FFT
    idft = jnp.exp(2j * jnp.pi * narr[:, None] * narr[None, :] / ntimes).astype(cdtype)
    interior = jnp.einsum("...ml,nl->...mn", mid, idft)

    # coefficient is sqrt(2) * (-1)^(n*m) * Re[conj(C_nm) * x], with
    # conj(C_nm) = 1 for n+m even and -1j for n+m odd, so the whole factor
    # depends only on the parities of n and m: a 2x2 table, broadcast out
    sqrt2 = math.sqrt(2.0)
    n_odd = (narr % 2).astype(bool)
    factor_even_m = jnp.where(n_odd, -1j, 1.0 + 0j) * sqrt2
    factor_odd_m = jnp.where(n_odd, -1.0 + 0j, -1j) * sqrt2
    m_odd = ((jnp.arange(nfreq_bands) + mmin) % 2).astype(bool)[:, None]
    factor = jnp.where(m_odd, factor_odd_m[None, :], factor_even_m[None, :])

    if mmin > 0:
        return jnp.real(interior * factor.astype(cdtype)).swapaxes(-1, -2)

    # channel 0 is the DC edge channel: only the first half-block, no C_nm phase,
    # a doubled frequency stride, and the l=0 bin entering at half weight
    dc_window = window[:half].at[0].divide(2.0)
    dc_idft = jnp.exp(
        4j * jnp.pi * narr[:, None] * jnp.arange(half)[None, :] / ntimes
    ).astype(cdtype)
    dc = sqrt2 * jnp.real(
        jnp.einsum("...l,nl->...n", blocks[..., 0, :] * dc_window, dc_idft)
    )

    # `interior` already starts at channel 1, so it pairs with factor[1:] directly
    coeffs = jnp.real(interior * factor[1:].astype(cdtype))
    return jnp.concatenate([dc[..., None, :], coeffs], axis=-2).swapaxes(-1, -2)
