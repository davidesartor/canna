"""RoPE on the shared frequency axis: the positions, the periods and the attention kernel they give, against what f0 needs."""

import argparse
import json
import math
from pathlib import Path
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from canna.lisa import config
from canna.lisa.chart import flow_to_physical, sample_flow
from canna.lisa.injections import scaled_inverse
from canna.lisa.network import FREQ_FINEST_PERIOD, STFT_GROUP, LisaFlow
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    FDOT_STD,
    SPEED_OF_LIGHT,
    YEAR,
    log10_fdot_mean,
)
from canna.lisa.problem import (
    WINDOW_STRIDE,
    LisaGB,
    window_power,
    window_start,
)
from canna.lisa.simulator import (
    sample_physical,
    sample_window,
    source_segments,
    source_snrs,
)
from canna.networks import mmdit

REPORT = Path(__file__).parents[1]
BANDS, SIZES = ("narrow", "full"), ("XS", "S", "B", "L", "XL")
f32, bf16 = jnp.float32, jnp.bfloat16
# the half width's fdot, this many FDOT_STDs above the prior's mean: a report convention, src applies no fdot cut
FDOT_SIGMAS = 5.0


def rope_angles(
    bins: Float[Array, " T"], periods: tuple[float, ...]
) -> tuple[Float[Array, "T P"], Float[Array, "T P"]]:
    """The network's RoPE (cos, sin): turns in float64, cast to float32."""
    angles = 2 * jnp.pi * jnp.mod(bins[:, None] / jnp.asarray(periods), 1.0).astype(f32)
    return jnp.cos(angles), jnp.sin(angles)


def rope_bins(flow: LisaFlow) -> tuple[float, ...]:
    """The network's RoPE periods (Hz) in bins."""
    return tuple(p * flow.t_obs for p in flow.rope_periods)


def captured_positions(problem: LisaGB, flow: LisaFlow, key) -> dict:
    """The (cos, sin) LisaFlow hands its backbone for data and source tokens, against rope_angles of the exact frequencies."""
    seen = {}

    def encode(self, y, c, rope, r):
        seen["data"] = rope
        return jnp.zeros(()), r

    def decode(self, x, c, cache, rope, s):
        seen["source"] = rope
        return jnp.zeros_like(x)

    key_f, key_x = jr.split(key)
    f = sample_window(problem, key_f)
    x = sample_flow(problem, key_x, f)
    no_frame = (jnp.zeros_like(x), jnp.zeros_like(x))
    with (
        mock.patch.object(mmdit.MMDiT, "encode", encode),
        mock.patch.object(mmdit.MMDiT, "decode", decode),
    ):
        _, cache = flow.encode(jnp.zeros((problem.window_bins, STFT_GROUP)), f)
        flow(x, jnp.asarray(0.5), no_frame, f, cache)

    # absolute position in Hz: window start plus the token's bins, over t_obs
    start = window_start(problem, f)
    data_bins = (
        flow.patch * np.arange(problem.window_bins // flow.patch) + (flow.patch - 1) / 2
    )
    source_bins = np.asarray(flow.source_bins(x[:, 0], f))
    exact = np.asarray(
        flow_to_physical(problem, x, f)[:, 0] * problem.t_obs - start
    )
    periods = jnp.asarray(flow.rope_periods)
    expected = lambda bins: np.cos(
        2 * np.pi * np.asarray(jnp.mod((start + jnp.asarray(bins))[:, None] / problem.t_obs / periods[None], 1.0), np.float32)
    )
    return dict(
        f=float(f),
        data_bins=data_bins.tolist(),
        source_bins=exact.round(3).tolist(),
        source_bins_code=source_bins.tolist(),
        source_bins_exact=exact.tolist(),
        data_angle_err=float(np.abs(np.asarray(seen["data"][0]) - expected(data_bins)).max()),
        source_angle_err=float(np.abs(np.asarray(seen["source"][0]) - expected(exact)).max()),
    )


def source_bins_accuracy(problem: LisaGB, flow: LisaFlow) -> dict:
    """source_bins against flow_to_physical's f0 T - start, over windows and x_f, in float64; and float32 x_f's step."""
    lo, hi = problem.window_index_range
    fs = (
        np.unique(np.linspace(lo, hi, 64).round()) * WINDOW_STRIDE
        + problem.window_bins / 2
    ) / problem.t_obs
    x_f = jnp.linspace(-1.0, 1.0, 2001)

    def one(f):
        x = jnp.zeros((x_f.size, 8)).at[:, 0].set(x_f)
        exact = flow_to_physical(problem, x, f)[:, 0] * problem.t_obs - window_start(
            problem, f
        )
        code = flow.source_bins(x_f, f)
        x32 = x_f.astype(f32)
        step = flow.source_bins(
            jnp.nextafter(x32, jnp.inf).astype(jnp.float64), f
        ) - flow.source_bins(x32.astype(jnp.float64), f)
        slope = (
            flow.source_bins(x_f + 1e-6, f) - flow.source_bins(x_f - 1e-6, f)
        ) / 2e-6
        return (
            jnp.abs(code - exact).max(),
            jnp.abs(step).max(),
            slope.max(),
            code.min(),
            code.max(),
        )

    err, step, slope, low, high = (
        np.asarray(a) for a in jax.lax.map(jax.jit(one), jnp.asarray(fs))
    )
    return dict(
        windows=len(fs),
        max_err=float(err.max()),
        f32_step=float(step.max()),
        max_slope=float(slope.max()),
        drawable_low=float(low.min()),
        drawable_high=float(high.max()),
    )


def path_error(problem: LisaGB, flow: LisaFlow, n: int, key) -> dict:
    """|source_bins(x_t) - source_bins(x1)| along x_t = x0 + s (x1 - x0), x0 from the base (the prior)."""
    one_minus_s = np.logspace(-6, 0, 25)

    def one(key):
        key_f, key_0, key_1 = jr.split(key, 3)
        f = sample_window(problem, key_f)
        x0, x1 = (
            sample_flow(problem, key_0, f)[:, 0],
            sample_flow(problem, key_1, f)[:, 0],
        )
        xt = x1[None] + jnp.asarray(one_minus_s)[:, None] * (x0 - x1)[None]
        return jnp.abs(flow.source_bins(xt, f) - flow.source_bins(x1, f)[None])

    err = np.asarray(jax.lax.map(jax.jit(one), jr.split(key, n), batch_size=64))
    err = np.moveaxis(err, 1, 0).reshape(one_minus_s.size, -1)
    return dict(
        one_minus_s=one_minus_s.tolist(),
        **{f"q{q}": np.percentile(err, q, axis=1).tolist() for q in (50, 90, 99)},
    )


def aligned_kernel(flow: LisaFlow, deltas: np.ndarray) -> np.ndarray:
    """q = k = ones through the code's rotate: the score is the mean over pairs of cos(2 pi delta / period)."""
    d, periods = 2 * len(flow.rope_periods), rope_bins(flow)
    ones = jnp.ones((deltas.size, 1, d), f32)
    q = mmdit.rotate(ones, *rope_angles(jnp.asarray(deltas), periods))
    k = mmdit.rotate(ones[:1], *rope_angles(jnp.zeros(1), periods))
    return np.asarray(jnp.einsum("thd,hd->t", q, k[0]) / d, np.float64)


def kernel_record(flow: LisaFlow, n_bins: int) -> dict:
    """The aligned kernel's shape, its sidelobes over the window and the margin of a source's own token."""
    dense = np.arange(0, n_bins + 0.125, 0.125)
    kern = aligned_kernel(flow, dense)
    below = lambda level: (
        float(dense[np.argmax(kern < level)]) if (kern < level).any() else None
    )
    first_zero = below(0.0)
    beyond = dense >= (first_zero or n_bins)
    side = int(np.argmax(np.where(beyond, kern, -np.inf)))

    # a source u bins off its own token's centre, against the best other token in the window
    patch, centres = (
        flow.patch,
        flow.patch * np.arange(n_bins // flow.patch) + (flow.patch - 1) / 2,
    )
    own = n_bins // flow.patch // 2
    u = np.arange(-patch / 2, patch / 2 + 1e-9, 1 / 16)
    offsets = np.abs(centres[own] + u[:, None] - centres[None])
    scores = aligned_kernel(flow, offsets.ravel()).reshape(offsets.shape)
    other = np.delete(scores, own, axis=1).max(1)

    fine = np.arange(0, 64 + 1e-9, 1 / 16)
    coarse = np.unique(
        np.concatenate(
            [
                np.geomspace(0.01, n_bins, 600),
                np.arange(0, n_bins + 1, max(n_bins // 1024, 1)),
            ]
        )
    )
    return dict(
        fine=dict(delta=fine.tolist(), k=aligned_kernel(flow, fine).round(5).tolist()),
        coarse=dict(
            delta=coarse.round(4).tolist(),
            k=aligned_kernel(flow, coarse).round(5).tolist(),
        ),
        hwhm=below(0.5),
        first_zero=first_zero,
        min_k=float(kern.min()),
        min_at=float(dense[np.argmin(kern)]),
        sidelobe=float(kern[side]),
        sidelobe_at=float(dense[side]),
        at={
            str(d): float(aligned_kernel(flow, np.array([d]))[0])
            for d in (1, 2, 4, 8, 16, 32, 64, 128)
        },
        margin=dict(
            u=u.tolist(),
            own=scores[:, own].round(5).tolist(),
            other=other.round(5).tolist(),
            min=float((scores[:, own] - other).min()),
        ),
    )


def bf16_noise(flow: LisaFlow, n_bins: int, n: int, key) -> dict:
    """Logit error of bf16-rounded rotated q and k (backbone and attention modes), and of float32 angles, as position noise."""
    d, periods = 2 * len(flow.rope_periods), rope_bins(flow)
    key_q, key_k, key_x = jr.split(key, 3)
    rms = lambda z: z * jax.lax.rsqrt(jnp.mean(z**2, -1, keepdims=True))
    q = rms(jr.normal(key_q, (n, 1, d))).astype(bf16)
    k = rms(jr.normal(key_k, (n, 1, d))).astype(bf16)
    xq, xk = (jr.uniform(kk, (n,), minval=0.0, maxval=n_bins) for kk in jr.split(key_x))

    def logit(q, k, xq, xk, dtype):
        qr = mmdit.rotate(q.astype(f32), *rope_angles(xq, periods)).astype(dtype)
        kr = mmdit.rotate(k.astype(f32), *rope_angles(xk, periods)).astype(dtype)
        return np.asarray(jnp.einsum("nhd,nhd->n", qr, kr, preferred_element_type=f32) * d**-0.5, np.float64)

    def logit64(q, k, xq, xk):
        def rot(z, x):
            a = 2 * jnp.pi * x[:, None] / jnp.asarray(periods)
            z1, z2 = jnp.split(z[:, 0].astype(jnp.float64), 2, -1)
            return jnp.concatenate([z1 * jnp.cos(a) - z2 * jnp.sin(a), z1 * jnp.sin(a) + z2 * jnp.cos(a)], -1)

        return np.asarray(jnp.sum(rot(q, xq) * rot(k, xk), -1) * d**-0.5)

    h = 1e-3
    slope = (logit64(q, k, xq + h, xk) - logit64(q, k, xq - h, xk)) / (2 * h)
    exact = logit64(q, k, xq, xk)
    err_bf16 = logit(q, k, xq, xk, bf16) - exact
    err_f32 = logit(q, k, xq, xk, f32) - exact
    noise = lambda err: float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(slope**2)))

    # one (q, k) pair: the logit against a sub-bin shift of the query
    delta = np.linspace(-0.01, 0.01, 401)
    qs, ks = jnp.repeat(q[:1], delta.size, 0), jnp.repeat(k[:1], delta.size, 0)
    xs_q, xs_k = jnp.asarray(n_bins / 2 + 0.3 + delta), jnp.full(delta.size, n_bins / 4 + 0.7)
    return dict(
        logit_rms=float(np.sqrt(np.mean(exact**2))),
        slope_rms=float(np.sqrt(np.mean(slope**2))),
        err_bf16=float(np.sqrt(np.mean(err_bf16**2))),
        err_f32=float(np.sqrt(np.mean(err_f32**2))),
        noise_bf16=noise(err_bf16),
        noise_f32=noise(err_f32),
        stair=dict(
            delta=delta.tolist(),
            f32=logit(qs, ks, xs_q, xs_k, f32).round(7).tolist(),
            bf16=logit(qs, ks, xs_q, xs_k, bf16).round(7).tolist(),
            exact=logit64(qs, ks, xs_q, xs_k).round(7).tolist(),
        ),
    )


def fisher(problem: LisaGB, n_windows: int, key) -> list[dict]:
    """Each prior source's marginal f0 std in bins, from the Fisher matrix of its own response (weak priors on fdot and angles)."""
    t_obs = problem.t_obs
    prior = jnp.diag(
        jnp.concatenate(
            [
                jnp.array([0.0, 1 / (FDOT_STD * math.log(10)) ** 2, 0.0]),
                jnp.full(5, 1 / jnp.pi**2),
            ]
        )
    )

    def to_physical(theta):
        return jnp.concatenate([theta[:1] / t_obs, jnp.exp(theta[1:3]), theta[3:]])

    def one_source(f, p):
        theta = jnp.concatenate([p[:1] * t_obs, jnp.log(p[1:3]), p[3:]])
        idx, _ = source_segments(problem, p[None], f)
        power = window_power(problem, f)[idx[0]]

        def seg(theta):
            s = source_segments(problem, to_physical(theta)[None], f)[1][0]
            return jnp.stack([s.real, s.imag])

        jac = jax.jacfwd(seg)(theta)
        info = 2 * jnp.einsum("rnct,rncu,nc->tu", jac, jac, 1 / power)
        cov = scaled_inverse(info + prior)
        return jnp.sqrt(cov[0, 0]), 1 / jnp.sqrt(info[0, 0]), jnp.sqrt(jnp.diag(cov)[1])

    def one_window(key):
        key_f, key_p = jr.split(key)
        f = sample_window(problem, key_f)
        p = sample_physical(problem, key_p, f)
        sigma, conditional, sigma_fdot = jax.vmap(one_source, in_axes=(None, 0))(f, p)
        return f, p, source_snrs(problem, p, f), sigma, conditional, sigma_fdot

    f, p, snr, sigma, conditional, sigma_fdot = (
        np.asarray(a)
        for a in jax.lax.map(
            jax.jit(one_window), jr.split(key, n_windows), batch_size=8
        )
    )
    return [
        dict(
            f0=float(p[w, s, 0]),
            snr=float(snr[w, s]),
            sigma=float(sigma[w, s]),
            conditional=float(conditional[w, s]),
            sigma_log_fdot=float(sigma_fdot[w, s]),
        )
        for w in range(n_windows)
        for s in range(problem.n_sources)
    ]


def band_record(band: str, n_path: int, n_fisher: int, n_noise: int) -> dict:
    problem = LisaGB(**config.load(f"{band}-B").problem)
    flows = {
        size: eqx.filter_eval_shape(
            lambda key, s=size: LisaFlow(
                problem, **config.load(f"{band}-{s}").network, key=key
            ),
            jr.key(0),
        )
        for size in SIZES
    }
    flow = flows["B"]
    for size, other in flows.items():
        for name in ("rope_periods", "freq_periods", "c_periods"):
            assert getattr(other, name) == getattr(
                flow, name
            ), f"{band}-{size} has other {name}"
    xs_cfg = config.load(f"{band}-XS")
    xs = LisaFlow(problem, **xs_cfg.network, key=jr.key(0))
    n_bins, patch = problem.window_bins, flow.patch
    f_lo, f_hi = problem.f0_range
    half_width = (
        lambda f0: f0 * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT * problem.t_obs
        + 10.0 ** (float(log10_fdot_mean(f0)) + FDOT_SIGMAS * FDOT_STD) * problem.t_obs**2
        + ANTENNA_HARMONICS * problem.t_obs / YEAR
    )
    periods = rope_bins(flow)
    lattice = [abs(patch / p - round(patch / p)) for p in periods]
    print(
        f"{band}: {n_bins} bins, rope {periods[0]:.3g}..{periods[-1]:.4g}",
        flush=True,
    )
    record = dict(
        band=band,
        t_obs=problem.t_obs,
        f0_range=[f_lo, f_hi],
        window_bins=n_bins,
        patch=patch,
        guard=flow.guard,
        sizes={
            s: dict(
                head_dim=config.load(f"{band}-{s}").network["hidden_dim"]
                // config.load(f"{band}-{s}").network["num_heads"],
                pairs=len(f.rope_periods),
            )
            for s, f in flows.items()
        },
        # RoPE periods (Hz in the network) in bins, positions absolute in Hz
        rope_periods=list(periods),
        lattice_alias=lattice,
        # f0's and the data tokens' frequency features, Hz periods in bins
        freq_periods=[p * problem.t_obs for p in flow.freq_periods],
        freq_finest=FREQ_FINEST_PERIOD,
        half_width=dict(low=half_width(f_lo), high=half_width(f_hi)),
        positions=captured_positions(problem, xs, jr.key(3)),
        source_bins=source_bins_accuracy(problem, flow),
        path=path_error(problem, flow, n_path, jr.key(4)),
        kernel=kernel_record(flow, n_bins),
        bf16=bf16_noise(flow, n_bins, n_noise, jr.key(5)),
    )
    print(
        f"{band}: kernel hwhm {record['kernel']['hwhm']}, zero {record['kernel']['first_zero']}, min {record['kernel']['min_k']:.3f}; bf16 noise {record['bf16']['noise_bf16']:.2e} bins",
        flush=True,
    )
    record["fisher"] = fisher(problem, n_fisher, jr.key(6))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n_path, n_fisher, n_noise = (64, 2, 256) if args.smoke else (2048, 64, 8192)
    record = dict(
        bands={band: band_record(band, n_path, n_fisher, n_noise) for band in BANDS}
    )
    out = REPORT / "data" / ("rope-smoke.json" if args.smoke else "rope.json")
    out.write_text(json.dumps(record, separators=(",", ":")))
    for band, r in record["bands"].items():
        sig = np.array([s["sigma"] for s in r["fisher"]])
        print(
            f"{band}: sigma_f0 bins 5/50/95% {np.percentile(sig, [5, 50, 95]).round(5).tolist()}, source_bins max err {r['source_bins']['max_err']:.2e}, captured angle err {r['positions']['data_angle_err']:.1e}/{r['positions']['source_angle_err']:.1e}"
        )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
