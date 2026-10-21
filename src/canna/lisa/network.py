import math

from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from ..networks import MMDiT, FeedForward, Modulation, PiecewiseLinear, SinusoidalEmbed
from ..networks.utils import DType
from .problem import DopplerGeometry, WindowGeometry

# PLE bins. The data are whitened to unit noise per real and imaginary part, and reach
# ~700 sigma for the loudest sources on XS; their edges are uniform in arcsinh out to
# DATA_REACH, so a bin is 0.24 sigma wide near the noise and ~70 sigma wide at 300 sigma.
# Quantile edges would leave everything from ~80 sigma up in one bin. f0 gets four times
# the bins of the other coordinates: it is the one the posterior resolves most finely
# against its prior range (F26).
DATA_BINS, DATA_REACH = 64, 1000.0
# with `stft`, each sample is fed as asinh|z| and the gated phase z / sqrt(1 + |z|^2). The
# amplitude's bins are uniform in asinh from 0 to asinh(DATA_REACH), 0.06 wide (4x finer than
# the re/im bins above); the phase's are uniform on [-1, 1]
STFT_AMPLITUDE_BINS, STFT_PHASE_BINS = 128, 64
STFT_GROUP = 6  # channels per window: re A, E, T, im A, E, T
COORD_BINS, F0_BINS, WINDOW_BINS = 64, 256, 64
# the flow coordinates' ranges: log f0 and log A are mapped onto [-1, 1], log Mc is
# standardised, and the sky, orientation and phase are unit vectors
COORD_RANGES = ((-1.0, 1.0), (-4.0, 4.0)) + ((-1.0, 1.0),) * 9


def uniform_edges(low: float, high: float, bins: int) -> tuple[float, ...]:
    return tuple(low + (high - low) * i / bins for i in range(bins + 1))


def arcsinh_edges(reach: float, bins: int) -> tuple[float, ...]:
    """Edges symmetric about 0 and uniform in arcsinh, out to +-reach."""
    return tuple(math.sinh(u) for u in uniform_edges(-math.asinh(reach), math.asinh(reach), bins))


def stft_tokens(y: Float[Array, "F C"], patch: int) -> Float[Array, "N P"]:
    """The whitened window as STFT tokens: each token's `patch` rfft bins (its band, as before)
    turned into the band's `patch` time samples by an orthonormal inverse DFT, T_obs / patch
    apart (45.7 days for 16 bins over 2 years).

    This is a critically sampled STFT: invertible, each token keeps exactly its band, and the
    whitened noise stays white with unit variance per part. Within a token, a source's f0
    offset is the slope of the phase across the samples, its initial phase the intercept,
    and the sky the annual Doppler and antenna-pattern modulation along them.

    Each complex sample z of A, E and T is fed as asinh|z|, log-like for loud samples and
    linear near zero, and the gated phase z / sqrt(1 + |z|^2): the unit phasor for loud
    samples, fading to 0 where the phase is noise. Both are smooth at z = 0, and together they
    determine z. Features are ordered [amplitudes, real parts, imaginary parts], each over
    (sample, channel).

    y may stack several such windows as groups of 6 channels (re A, E, T, im A, E, T), as the
    template residual does (train.with_residual: the data, then the data minus the sources at
    the flow's current point); each group gets its own features, one group after the other.
    """
    groups = []
    for g in jnp.split(y, y.shape[-1] // STFT_GROUP, axis=-1):
        z = g[:, :3] + 1j * g[:, 3:]
        z = rearrange(z, "(n p) c -> n p c", p=patch)
        z = jnp.fft.ifft(z, axis=1, norm="ortho")
        phase = z * jax.lax.rsqrt(1.0 + jnp.square(jnp.abs(z)))
        flat = lambda a: rearrange(a, "n p c -> n (p c)")
        groups += [flat(jnp.arcsinh(jnp.abs(z))), flat(phase.real), flat(phase.imag)]
    return jnp.concat(groups, axis=-1).astype(y.dtype)


def doppler_bands(
    y: Float[Array, "F C"],
    x: Float[Array, "S X"],
    f: Float[Array, ""],
    window: WindowGeometry,
    doppler: DopplerGeometry,
    band: int,
) -> Float[Array, "S B"]:
    """Each source's band of the data, Doppler-demodulated at its current sky and f0 (F57).

    Cornish & Larson's demodulation (gr-qc/0206017): the whitened window (its first 6 channels,
    re and im of A, E, T) goes to time, is multiplied by exp(-i phi_D(t)) for the source's
    direction and frequency (DopplerGeometry), and comes back to frequency. At the right sky the
    source's power, spread by the annual Doppler sidebands, collapses into a few bins; away from
    it, the leftover spread and phase wobble measure the sky error. phi_D depends on f0 only as a
    factor, so a frequency off by bins still demodulates correctly. The `band` bins around each
    source's f0 are returned with the STFT tokens' features (asinh|z|, then the gated phase), so
    [amplitudes, real parts, imaginary parts], each over (bin, channel).
    """
    f32 = jnp.float32
    z = (y[:, :3] + 1j * y[:, 3:6]).astype(jnp.complex64)
    n_bins = z.shape[0]
    times = jnp.arange(n_bins, dtype=f32) * (doppler.t_obs / n_bins)
    series = jnp.fft.ifft(z, axis=0)
    start = window.start(f)
    bins = window.source_bins(x[:, 0].astype(f32), f).astype(f32)
    f0 = (start + bins) / window.t_obs
    direction = x[:, 3:6].astype(f32)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    phase = doppler.phase(f0.astype(f32), direction, times)
    demodulated = jnp.fft.fft(series[None] * jnp.exp(-1j * phase)[..., None], axis=1)
    low = jnp.clip(jnp.round(bins).astype(jnp.int32) - band // 2, 0, n_bins - band)
    cut = jax.vmap(lambda d, lo: jax.lax.dynamic_slice(d, (lo, jnp.zeros_like(lo)), (band, 3)))(demodulated, low)
    gated = cut * jax.lax.rsqrt(1.0 + jnp.square(jnp.abs(cut)))
    flat = lambda a: a.reshape(a.shape[0], -1)
    return jnp.concat(
        [flat(jnp.arcsinh(jnp.abs(cut))), flat(gated.real), flat(gated.imag)], axis=-1
    ).astype(f32)


class LisaFlow(eqx.Module):
    """Velocity field over a set of binaries, conditioned on a whitened frequency-domain window.

    The data are the window's rfft bins, whitened, with the real and imaginary parts of A, E
    and T stacked as six channels (twelve with `templates`: the residual of the data against
    the sources at the flow's current point, train.with_residual). `patch` neighbouring bins make one token,
    either as they are or, with `stft`, as the band's time samples (`stft_tokens`). Every scalar
    the network takes in -- each data value, each source coordinate, the window frequency --
    goes through a piecewise linear encoding (`PiecewiseLinear`) and one linear map.

    Precision: the embeddings, the conditioning and the velocity head run in float32 and
    only the MMDiT backbone in `dtype`. The arcsinh(WDM) input of the earlier networks was
    cast to bf16 before its first layer, which rounded the loudest pixels by 0.5-3.6 noise
    sigmas rms; here the cast comes after the encoding.

    The sources come loudest first (LisaGB.loudest_first), so each source slot gets a learned
    embedding of its own, as each y token gets one for its position.

    Two heads read the source stream: the velocity, and the reconstructed parameters x1 (the
    auxiliary target: at t = 0 it is the posterior mean of each slot's parameters).
    """

    dtype: DType = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    # rotary position embedding in frequency (optional): the window's geometry, to place each
    # source and each data token in bins, and the rotation periods, in bins
    window: WindowGeometry | None = eqx.field(static=True)
    rope_periods: tuple[float, ...] | None = eqx.field(static=True)
    # the conditioning carries the template residual at the flow's current point beside the
    # data (train.with_residual): 12 channels instead of 6. The network takes whatever y it is
    # given; this flag tells the sampler to build the same conditioning at every step
    templates: bool = eqx.field(static=True)
    # the data tokens as short-time Fourier transforms (`stft_tokens`) instead of rfft bins
    stft: bool = eqx.field(static=True)
    # each source also sees its band of the data Doppler-demodulated at its current sky
    # (`doppler_bands`), embedded and added to its token; None leaves it out
    doppler: DopplerGeometry | None = eqx.field(static=True)
    doppler_band: int = eqx.field(static=True)
    d_embed: PiecewiseLinear | None
    x_embed: PiecewiseLinear
    y_embed: PiecewiseLinear
    y_pos: Float[Array, "N D"]
    x_slot: Float[Array, "S D"] | None
    t_embed: SinusoidalEmbed
    t_mlp: FeedForward
    f_embed: PiecewiseLinear
    f_mlp: FeedForward
    backbone: MMDiT
    x_modulation: Modulation
    x_unembed: FeedForward

    def __init__(
        self,
        x_shape: tuple[int, int],
        y_shape: tuple[int, int],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        f_range: tuple[float, float],
        patch: int = 2,
        expand: int = 2,
        slots: bool = True,
        rope: bool = False,
        window: WindowGeometry | None = None,
        rope_min_period: float = 2.0,
        templates: bool = False,
        stft: bool = False,
        doppler: DopplerGeometry | None = None,
        doppler_band: int = 16,
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        (n_sources, x_dim), (*_, n_bins, y_dim) = x_shape, y_shape
        assert x_dim == len(COORD_RANGES), f"expected {len(COORD_RANGES)} flow coordinates"
        assert n_bins % patch == 0, f"window of {n_bins} bins is not divisible by {patch}"
        self.dtype, self.patch, self.templates, self.stft = dtype, patch, templates, stft
        assert not stft or y_dim % STFT_GROUP == 0, f"STFT tokens take groups of {STFT_GROUP} channels"
        # RoPE: one angle per token and per pair of a head's dimensions, periods log-spaced from
        # rope_min_period bins to twice the window, so attention resolves f0 - f_token from a
        # fraction of a bin up to the whole window
        assert not rope or window is not None, "RoPE needs the window geometry"
        assert doppler is None or window is not None, "the Doppler bands need the window geometry"
        self.window = window if (rope or doppler is not None) else None
        self.doppler, self.doppler_band = doppler, doppler_band
        pairs = hidden_dim // num_heads // 2
        self.rope_periods = (
            tuple(
                float(rope_min_period * (2.0 * window.window_bins / rope_min_period) ** (k / max(pairs - 1, 1)))
                for k in range(pairs)
            )
            if rope
            else None
        )

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(kwargs, dtype=param_dtype)
        keys = iter(jr.split(key, 11))

        coord_edges = tuple(
            uniform_edges(lo, hi, F0_BINS if i == 0 else COORD_BINS)
            for i, (lo, hi) in enumerate(COORD_RANGES)
        )
        self.x_embed = PiecewiseLinear(coord_edges, hidden_dim, key=next(keys), **kwargs)
        if stft:
            # per token and group (the data, and the template residual with `templates`): patch
            # time samples x 3 complex channels x (amplitude, then the 2 phase parts)
            n = patch * (STFT_GROUP // 2)
            data_edges = (
                (uniform_edges(0.0, math.asinh(DATA_REACH), STFT_AMPLITUDE_BINS),) * n
                + (uniform_edges(-1.0, 1.0, STFT_PHASE_BINS),) * (2 * n)
            ) * (y_dim // STFT_GROUP)
        else:
            data_edges = (arcsinh_edges(DATA_REACH, DATA_BINS),) * (patch * y_dim)
        self.y_embed = PiecewiseLinear(data_edges, hidden_dim, key=next(keys), **kwargs)
        # per source: doppler_band bins x 3 channels x (amplitude, 2 phase parts)
        key_doppler = jr.fold_in(key, 11)  # its own key: the split above keeps every other init as before
        n = doppler_band * 3
        self.d_embed = (
            PiecewiseLinear(
                (uniform_edges(0.0, math.asinh(DATA_REACH), STFT_AMPLITUDE_BINS),) * n
                + (uniform_edges(-1.0, 1.0, STFT_PHASE_BINS),) * (2 * n),
                hidden_dim, key=key_doppler, **kwargs,
            )
            if doppler is not None
            else None
        )
        self.y_pos = 0.02 * jr.normal(
            next(keys), (n_bins // patch, hidden_dim), param_dtype
        )
        # a learned embedding per source slot, for ordered sources; without it the source
        # stream is permutation equivariant, as an unordered set needs
        key_slot = next(keys)
        self.x_slot = (
            0.02 * jr.normal(key_slot, (n_sources, hidden_dim), param_dtype) if slots else None
        )

        self.t_embed = SinusoidalEmbed(hidden_dim, key=next(keys), **kwargs)
        self.t_mlp = FeedForward(hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs)
        log_f = (uniform_edges(math.log(f_range[0]), math.log(f_range[1]), WINDOW_BINS),)
        self.f_embed = PiecewiseLinear(log_f, hidden_dim, key=next(keys), **kwargs)
        self.f_mlp = FeedForward(hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs)

        # no biases inside the blocks: every parameter there is a matrix Muon can take
        self.backbone = MMDiT(
            hidden_dim, num_heads, num_blocks, expand, key=next(keys), use_bias=False, **kwargs
        )

        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, key=next(keys), **kwargs
        )

    def __call__(
        self,
        x: Float[Array, "S X"],
        t: Float[Array, ""],
        y: Float[Array, "F C"],
        f: Float[Array, ""],
    ) -> tuple[Float[Array, "S X"], Float[Array, "S X"]]:
        assert self.x_slot is None or x.shape[0] == self.x_slot.shape[0], (
            f"built for {self.x_slot.shape[0]} source slots, got {x.shape[0]} sources"
        )
        f32 = jnp.float32
        x_in = x  # the input precision, for the sources' positions
        c = self.t_mlp(self.t_embed(t.astype(f32)))
        c = c + self.f_mlp(self.f_embed(jnp.log(f.astype(f32))[None]))
        x = jax.vmap(self.x_embed)(x.astype(f32))
        if self.doppler is not None:
            bands = doppler_bands(y.astype(f32), x_in, f, self.window, self.doppler, self.doppler_band)
            x = x + jax.vmap(self.d_embed)(bands)
        if self.x_slot is not None:
            x = x + self.x_slot
        if self.stft:
            y = stft_tokens(y.astype(f32), self.patch)
        else:
            y = rearrange(y.astype(f32), "(n p) c -> n (p c)", p=self.patch)
        y = jax.vmap(self.y_embed)(y) + self.y_pos

        # mixed precision: the stored params keep param_dtype, the backbone runs in dtype
        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        backbone = eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)
        rope = None
        if self.rope_periods is not None:
            # positions in bins from the window start, computed before any cast: each source at
            # its current f0, each data token at its centre bin
            x_bins = self.window.source_bins(x_in[:, 0], f).astype(f32)
            y_bins = self.window.token_bins(y.shape[0], self.patch)
            bins = jnp.concat([x_bins, y_bins])
            angles = 2 * jnp.pi * bins[:, None] / jnp.asarray(self.rope_periods, f32)[None, :]
            rope = (jnp.cos(angles), jnp.sin(angles))
        x, _ = backbone(x.astype(self.dtype), y.astype(self.dtype), c.astype(self.dtype), rope)

        x, _ = self.x_modulation(x.astype(f32), c)
        dx, x1 = jnp.split(jax.vmap(self.x_unembed)(x), 2, axis=-1)
        return dx, x1
