"""What the data stream sees: STFT patch tokens per band, and where each source's power lands among them."""

import argparse
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float

from canna.lisa import config
from canna.lisa.network import LisaFlow, stft_tokens
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    SPEED_OF_LIGHT,
    YEAR,
    FDOT_STD,
    equatorial_to_galactic,
    log10_fdot_mean,
)
from canna.lisa.problem import LisaGB, f0_window, window_power, window_start
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
    sample_window,
    source_segments,
)

REPORT = Path(__file__).parents[1]
OBLIQUITY = math.radians(23.4392911)
BANDS = ("narrow", "full")
# the analytic curve's fdot, this many FDOT_STDs above the prior's mean: a report convention, src applies no fdot cut
FDOT_SIGMAS = 5.0


def ecliptic_latitude(
    ra: Float[Array, "..."], dec: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Ecliptic latitude of an equatorial (ICRS) direction [rad]."""
    sin_beta = jnp.sin(dec) * math.cos(OBLIQUITY) - jnp.cos(dec) * math.sin(
        OBLIQUITY
    ) * jnp.sin(ra)
    return jnp.arcsin(jnp.clip(sin_beta, -1, 1))


def central(cum: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """First and last bin of the central interval holding cum's [lo, hi] share."""
    return int(np.searchsorted(cum, lo)), int(np.searchsorted(cum, hi))


def population(problem: LisaGB, flow: LisaFlow, n_windows: int, key) -> list[dict]:
    """Per source: its snr² per bin, summarised as the bins and tokens that hold it."""
    patch, n_bins, guard = flow.patch, problem.window_bins, flow.guard

    def one(key):
        key_f, key_p = jr.split(key)
        f = sample_window(problem, key_f)
        p = sample_physical(problem, key_p, f)
        idx, seg = source_segments(problem, p, f)
        rho2 = 2 * jnp.sum(
            jnp.abs(seg) ** 2 / window_power(problem, f)[jnp.clip(idx, 0, n_bins - 1)],
            -1,
        )
        return (
            f,
            p,
            idx,
            rho2,
            window_start(problem, f),
            jnp.stack(f0_window(problem, f)),
        )

    f, p, idx, rho2, start, f0_lim = jax.device_get(
        jax.lax.map(jax.jit(one), jr.split(key, n_windows), batch_size=16)
    )
    beta = np.asarray(ecliptic_latitude(p[..., 3], p[..., 4]))
    gal_l, gal_b = (np.asarray(a) for a in equatorial_to_galactic(p[..., 3], p[..., 4]))
    fdot = np.asarray(p[..., 1])
    rows = []
    for w in range(n_windows):
        for s in range(problem.n_sources):
            r2, bins = rho2[w, s], idx[w, s]
            assert (
                bins.min() >= 0 and bins.max() < n_bins
            ), "a response leaves the window"
            cum = np.cumsum(r2) / r2.sum()
            f0_pos = p[w, s, 0] * problem.t_obs - start[w]
            token_power = (
                np.bincount(bins // patch, weights=r2, minlength=n_bins // patch)
                / r2.sum()
            )
            greedy = np.sort(token_power)[::-1].cumsum()
            row = dict(
                f=float(f[w]),
                f0=float(p[w, s, 0]),
                f0_bin=float(f0_pos),
                snr=float(np.sqrt(r2.sum())),
                fdot=float(fdot[w, s]),
                beta=float(beta[w, s]),
                gal_l=float(gal_l[w, s]),
                gal_b=float(gal_b[w, s]),
                doppler=float(
                    p[w, s, 0]
                    * EARTH_ORBIT_SPEED
                    / SPEED_OF_LIGHT
                    * abs(math.cos(beta[w, s]))
                    * problem.t_obs
                ),
                drift=float(fdot[w, s] * problem.t_obs**2),
                own_share=float(token_power[int(f0_pos // patch)]),
                top_share=float(token_power.max()),
                greedy99=int(np.searchsorted(greedy, 0.99) + 1),
                greedy90=int(np.searchsorted(greedy, 0.90) + 1),
                guard_share=float(
                    r2[(bins < guard) | (bins >= n_bins - guard)].sum() / r2.sum()
                ),
            )
            for level, (lo, hi) in (("99", (0.005, 0.995)), ("90", (0.05, 0.95))):
                a, b = central(cum, lo, hi)
                row[f"lo{level}"], row[f"hi{level}"] = int(bins[a]), int(bins[b])
                row[f"tokens{level}"] = int(bins[b] // patch - bins[a] // patch + 1)
                row[f"width{level}"] = int(bins[b] - bins[a] + 1)
            rows.append(row)
    return rows


def token_channels(tokens: Float[Array, "N Q"], patch: int) -> dict:
    """asinh|z| and the phase angle per channel, as [channel][token][time] ×100 integers."""
    amp, re, im = jnp.split(tokens, 3, axis=-1)
    split = lambda a: rearrange(a, "n (p c) -> c n p", p=patch)
    to_int = lambda a: np.rint(100 * np.asarray(a, np.float64)).astype(int).tolist()
    return dict(amp=to_int(split(amp)), phase=to_int(split(jnp.arctan2(im, re))))


def observations(problem: LisaGB, flow: LisaFlow, n_obs: int, key) -> list[dict]:
    """A few drawn windows: their tokens with and without noise, the per-bin whitened magnitude, the sources."""
    out = []
    for k in jr.split(key, n_obs):
        key_f, key_p, key_o = jr.split(k, 3)
        f = sample_window(problem, key_f)
        p = sample_physical(problem, key_p, f)
        y = preprocess(problem, sample_observation(problem, key_o, p, f), f)
        y_signal = preprocess(problem, clean_signal(problem, p, f), f)
        start = int(window_start(problem, f))
        idx, seg = source_segments(problem, p, f)
        rho2 = 2 * jnp.sum(jnp.abs(seg) ** 2 / window_power(problem, f)[idx], -1)
        sources = []
        for s in range(problem.n_sources):
            cum = np.cumsum(np.asarray(rho2[s])) / float(rho2[s].sum())
            a, b = central(cum, 0.005, 0.995)
            sources.append(
                dict(
                    f0_bin=float(p[s, 0] * problem.t_obs - start),
                    lo99=int(idx[s, a]),
                    hi99=int(idx[s, b]),
                    snr=float(jnp.sqrt(rho2[s].sum())),
                    beta=float(ecliptic_latitude(p[s, 3], p[s, 4])),
                )
            )
        magnitude = lambda v: jnp.arcsinh(jnp.sqrt(jnp.sum(jnp.square(v), -1) / 3))
        out.append(
            dict(
                f=float(f),
                start=start,
                data=token_channels(stft_tokens(y, flow.patch), flow.patch),
                signal=token_channels(stft_tokens(y_signal, flow.patch), flow.patch),
                bin_data=np.rint(100 * np.asarray(magnitude(y), np.float64))
                .astype(int)
                .tolist(),
                bin_signal=np.rint(100 * np.asarray(magnitude(y_signal), np.float64))
                .astype(int)
                .tolist(),
                sources=sources,
            )
        )
    return out


def band_record(band: str, n_windows: int, n_obs: int) -> dict:
    cfg = config.load(f"{band}-B")
    problem = LisaGB(**cfg.problem)
    flow = eqx.filter_eval_shape(
        lambda key: LisaFlow(problem, **cfg.network, key=key), jr.key(0)
    )
    n_tokens = problem.window_bins // flow.patch

    # the analytic half-width of the response across the band: doppler at |cos beta| = 1, fdot FDOT_SIGMAS above its mean, the antenna sidebands
    f0_grid = np.geomspace(*problem.f0_range, 64)
    fdot_hi = 10.0 ** (np.asarray(log10_fdot_mean(f0_grid)) + FDOT_SIGMAS * FDOT_STD)
    curve = dict(
        f0=f0_grid.tolist(),
        doppler=(f0_grid * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT * problem.t_obs).tolist(),
        drift=(fdot_hi * problem.t_obs**2).tolist(),
        antenna=ANTENNA_HARMONICS * problem.t_obs / YEAR,
    )
    print(
        f"{band}: {problem.window_bins} bins, patch {flow.patch}, {n_tokens} tokens, guard {flow.guard} bins",
        flush=True,
    )
    return dict(
        band=band,
        t_obs=problem.t_obs,
        f0_range=list(problem.f0_range),
        response_points=problem.response_points,
        window_bins=problem.window_bins,
        patch=flow.patch,
        tokens=n_tokens,
        guard=flow.guard,
        features=9 * flow.patch,
        n_sources=problem.n_sources,
        curve=curve,
        population=population(problem, flow, n_windows, jr.key(1)),
        observations=observations(problem, flow, n_obs, jr.key(2)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n_windows, n_obs = (16, 2) if args.smoke else (256, 6)
    record = dict(bands={band: band_record(band, n_windows, n_obs) for band in BANDS})
    out = REPORT / "data" / ("tokens-smoke.json" if args.smoke else "tokens.json")
    out.write_text(json.dumps(record, separators=(",", ":")))
    for band, r in record["bands"].items():
        pop = r["population"]
        q = lambda k: np.percentile([s[k] for s in pop], [5, 50, 95]).round(2).tolist()
        print(
            f"{band}: tokens99 {q('tokens99')}, greedy99 {q('greedy99')}, own share {q('own_share')}, guard share max {max(s['guard_share'] for s in pop):.2e}"
        )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
