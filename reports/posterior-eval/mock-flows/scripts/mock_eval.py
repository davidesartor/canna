"""Mock flows: the lisa eval pipeline with the flow's sampler swapped for stand-in posteriors with known defects.

Usage: uv run --frozen python reports/posterior-eval/mock-flows/scripts/mock_eval.py <band> [--mocks a,b] [--n-validation N] [--n-random N]
Writes data/<band>/<mock>/{eval.yaml, draws.npz, scorecard.npz, mock.json}.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from einops import rearrange
from jaxtyping import Array, Complex, Float, Key

from canna.lisa import config as run_config
from canna.lisa import validate
from canna.lisa.chart import flow_to_physical, physical_to_flow, sample_flow
from canna.lisa.injections import (
    N_POSTERIOR,
    N_PRIOR_DRAWS,
    eval_injections,
    fisher_samples,
    match_sources,
    scaled_inverse,
)
from canna.lisa.problem import LisaGB, sample_f, window_power
from canna.lisa.samples import load as load_samples
from canna.lisa.samples import pack_validation, unpack
from canna.lisa.scorecard import N_DRAWS as SCORE_DRAWS
from canna.lisa.scorecard import SNR_BANDS, score
from canna.lisa.simulator import (
    clean_signal,
    sample_observation,
    sample_physical,
    snr,
    source_snrs,
)

REPORT = Path(__file__).parents[1]
MOCKS = ("fisher", "wide", "narrow", "biased", "prior", "phases-prior", "loud-floor")
WIDE, NARROW = 3.0, 0.5
# B's loud-f0 floor (research log F44, F48): the monochromatic Fisher f0 width sqrt(3) / (pi SNR) reaches it at SNR 18.4
FLOOR_BINS = 0.03
FLOOR_SNR = float(np.sqrt(3) / (np.pi * FLOOR_BINS))
DESCRIPTIONS = {
    "fisher": "Linearised Gaussian posterior in flow coordinates: Fisher + Gaussian-prior precision at the truth, centred on the linearised MAP of the actual noisy data. Approximately calibrated.",
    "wide": f"fisher with every width x{WIDE:g} (same centre and correlations).",
    "narrow": f"fisher with every width x{NARROW:g} (same centre and correlations).",
    "biased": "fisher with every source's f0 centre shifted by +1 marginal posterior sigma.",
    "prior": "Draws from the prior for the same window f, ignoring the data (the flow's own standard-normal base draws, an identity flow).",
    "phases-prior": "fisher with the phases of z+ and z- redrawn uniform on the circle, radii kept: the phases come out as the prior (research log F52).",
    "loud-floor": f"fisher with each source's widths stopped shrinking at SNR {FLOOR_SNR:.1f} (f0 floor {FLOOR_BINS} bins, B's loud-f0 floor, F44/F48), capped at the prior width: quiet sources exact, loud ones up to ~{1000 / FLOOR_SNR:.0f}x too wide.",
}


def flow_prior(
    problem: LisaGB, seed: int
) -> tuple[Float[Array, " P"], Float[Array, "P P"]]:
    """Mean and covariance of N_PRIOR_DRAWS prior draws in flow coordinates, sources flattened."""
    key_pick, key_window, _ = jr.split(jr.key(seed), 3)
    windows = jax.vmap(partial(sample_f, problem))(jr.split(key_window, N_PRIOR_DRAWS))
    latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, N_PRIOR_DRAWS), windows
    )
    x = jax.vmap(partial(physical_to_flow, problem))(latents, windows)
    flat = rearrange(x, "n s d -> n (s d)")
    return flat.mean(0), jnp.cov(flat.T)


@eqx.filter_jit
def gaussian_posterior(
    problem: LisaGB,
    x0: Float[Array, "S 8"],
    o: Complex[Array, "F 3"],
    f: Float[Array, ""],
    prior_mean: Float[Array, " P"],
    prior_precision: Float[Array, "P P"],
) -> tuple[Float[Array, " P"], Float[Array, "P P"]]:
    """N(x0 + C (score + P (mu - x0)), C), C = (Fisher + P)^-1 at the truth x0: the linear-Gaussian posterior."""
    signal = lambda v: clean_signal(
        problem, flow_to_physical(problem, v.reshape(x0.shape), f), f
    )
    v0 = x0.reshape(-1)
    jac = jax.jacfwd(signal)(v0)
    weighted = jac / window_power(problem, f)[..., None]
    fisher = 2 * jnp.einsum("fcp,fcq->pq", jnp.conj(jac), weighted).real
    data_score = 2 * jnp.einsum("fc,fcp->p", jnp.conj(o - signal(v0)), weighted).real
    cov = scaled_inverse(fisher + prior_precision)
    cov = (cov + cov.T) / 2
    return v0 + cov @ (data_score + prior_precision @ (prior_mean - v0)), cov


def mock_draw(
    problem: LisaGB,
    mock: str,
    key: Key[Array, ""],
    f: Float[Array, ""],
    centre: Float[Array, " P"],
    root: Float[Array, "P P"],
) -> Float[Array, "S 8"]:
    """One mock draw in flow coordinates from the (already reshaped) Gaussian centre + root z."""
    if mock == "prior":
        return sample_flow(problem, key, f)
    z = jr.normal(jr.fold_in(key, 100), centre.shape, centre.dtype)
    x = (centre + root @ z).reshape(problem.n_sources, 8)
    if mock == "phases-prior":
        # z+ is columns 4, 5 and z- 6, 7: keep each radius, draw its angle uniform
        angle = jr.uniform(
            jr.fold_in(key, 101), (problem.n_sources, 2), x.dtype, 0, 2 * jnp.pi
        )
        radius = jnp.hypot(x[:, 4::2], x[:, 5::2])
        x = (
            x.at[:, 4::2]
            .set(radius * jnp.cos(angle))
            .at[:, 5::2]
            .set(radius * jnp.sin(angle))
        )
    return x


@eqx.filter_jit
def mock_draws(
    problem: LisaGB,
    mock: str,
    keys: Key[Array, " N"],
    f: Float[Array, ""],
    centre: Float[Array, " P"],
    cov: Float[Array, "P P"],
    prior_sd: Float[Array, " P"],
) -> Float[Array, "N S 8"]:
    """The mock's draws for one observation, in physical units like the flow's: draws = flow_to_physical(x)."""
    sd = jnp.sqrt(jnp.diag(cov))
    w, v = jnp.linalg.eigh(cov / jnp.outer(sd, sd))
    root = sd[:, None] * v * jnp.sqrt(jnp.maximum(w, 0.0))

    # each mock is a per-coordinate rescaling K of the root and a shift of the centre
    scale, shift = jnp.ones_like(sd), jnp.zeros_like(sd)
    if mock == "wide":
        scale = scale * WIDE
    if mock == "narrow":
        scale = scale * NARROW
    if mock == "biased":
        shift = jnp.zeros((problem.n_sources, 8)).at[:, 0].set(1.0).reshape(-1) * sd
    if mock == "loud-floor":
        # the source SNR at the centre, a function of the data
        centre_snr = source_snrs(
            problem, flow_to_physical(problem, centre.reshape(-1, 8), f), f
        )
        factor = jnp.repeat(jnp.maximum(centre_snr / FLOOR_SNR, 1.0), 8)
        scale = jnp.maximum(1.0, jnp.minimum(factor, prior_sd / sd))
    x = jax.vmap(mock_draw, in_axes=(None, None, 0, None, None, None))(
        problem, mock, keys, f, centre + shift, scale[:, None] * root
    )
    return jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(x, f)


@eqx.filter_jit
def observe(
    problem: LisaGB,
    key_n: Key[Array, ""],
    latent: Float[Array, "S 8"],
    f: Float[Array, ""],
    prior: tuple,
):
    """The noisy observation and its Gaussian posterior."""
    o = sample_observation(problem, key_n, latent, f)
    return o, *gaussian_posterior(
        problem, physical_to_flow(problem, latent, f), o, f, *prior
    )


@eqx.filter_jit
def validation_injection(problem: LisaGB, key: Key[Array, ""], prior: tuple):
    """validate_injection's prior injection and keys, without the flow."""
    key_f, key_p, key_n, key_u, key_tarp = jr.split(key, 5)
    f = sample_f(problem, key_f)
    truth = sample_physical(problem, key_p, f)
    o, centre, cov = observe(problem, key_n, truth, f, prior)
    return (
        f,
        truth,
        o,
        centre,
        cov,
        jr.split(key_u, validate.N_DRAWS),
        sample_flow(problem, key_tarp, f),
    )


quantities = eqx.filter_jit(validate.quantities)


def write_eval_yaml(out: Path, n_random: int, n_validation: int) -> None:
    settings = dict(
        epoch=-1,
        weights="mock",
        ode_steps=0,
        bf16="none",
        backend=jax.default_backend(),
        n_random=n_random,
        n_validation=n_validation,
    )
    with open(out / "eval.yaml", "w") as file:
        yaml.safe_dump(settings, file, sort_keys=False)


def width_ratios(problem: LisaGB, d: dict) -> dict:
    """Matched mock width over the repo's Fisher-forecast width (1.4826 MAD), per flow coordinate and true-SNR band."""
    to_flow = jax.vmap(jax.vmap(partial(physical_to_flow, problem), in_axes=(0, None)))
    widths = []
    for key in ("matched", "fisher"):
        x = np.asarray(to_flow(jnp.asarray(d[key]), jnp.asarray(d["f"])))
        widths.append(1.4826 * np.median(np.abs(x - np.median(x, 1, keepdims=True)), 1))
    ratio = rearrange(widths[0] / widths[1], "n s d -> (n s) d")
    source_snr = d["source_snr"].reshape(-1)
    out = {}
    for lo, hi in [(0, 40), (40, 100), (100, np.inf), (0, np.inf)]:
        m = (source_snr >= lo) & (source_snr < hi)
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
        out[label] = dict(
            n=int(m.sum()),
            **{c: float(np.median(ratio[m, i])) for i, c in enumerate(validate.COORDS)},
        )
    return out


def scorecard_checks(rows: dict) -> dict:
    """The scorecard summary's columns per true-SNR band, as numbers."""
    out = {}
    for lo, hi in [*SNR_BANDS, (0, np.inf)]:
        m = (rows["snr"] >= lo) & (rows["snr"] < hi)
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
        out[label] = dict(
            n=int(m.sum()),
            width_over_ideal=float(np.median(rows["width"][m] / rows["ideal"][m])),
            offset_over_width=float(np.median(rows["offset"][m] / rows["width"][m])),
            in68=float(rows["in68"][m].mean()),
            in95=float(rows["in95"][m].mean()),
            mean_rank=float(rows["rank"][m].mean()),
        )
    return out


def validation_checks(out: Path) -> dict:
    """γ p-value, mean-rank offset and middle-half share of every rank set, pooled over slots, all and loud (snr_hat >= 100)."""
    scores = load_samples(out / "scorecard.npz")
    k = int(scores["draws"]) + 1

    def cell(r: np.ndarray) -> dict:
        if len(r) < 10:
            return dict(n=int(len(r)))
        p, _ = validate.gamma_test(r, k)
        return dict(
            n=int(len(r)),
            p=float(p),
            bias=float(r.mean() / (k - 1) - 0.5),
            middle=float(np.mean((r >= k / 4) & (r < 3 * k / 4))),
        )

    loud = scores["snr_hat"] >= 100
    joint_loud = scores["joint_snr_hat"] >= 100
    per_slot = {
        name.removeprefix("rank_"): dict(all=cell(v.reshape(-1)), loud=cell(v[loud]))
        for name, v in scores.items()
        if name.startswith("rank_")
    }
    joint = {
        name.removeprefix("joint_rank_"): dict(all=cell(v), loud=cell(v[joint_loud]))
        for name, v in scores.items()
        if name.startswith("joint_rank_")
    }
    return dict(
        per_slot=per_slot,
        joint=joint,
        energy_median=float(np.median(scores["energy"])),
        joint_energy_median=float(np.median(scores["joint_energy"])),
        width_ratio_median=dict(
            all=float(np.median(scores["width_ratio"])),
            loud=float(np.median(scores["width_ratio"][loud])) if loud.any() else None,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("band", choices=("narrow", "full"))
    parser.add_argument("--mocks", default=",".join(MOCKS))
    parser.add_argument("--n-validation", type=int, default=200)
    parser.add_argument("--n-random", type=int, default=None)
    args = parser.parse_args()
    mocks = args.mocks.split(",")
    assert set(mocks) <= set(MOCKS), mocks

    # the problem and seed a run of this band trains and evaluates with; the size does not enter
    config = run_config.load(f"{args.band}-B")
    problem = LisaGB(**config.problem)
    seed, n_validation = config.seed, args.n_validation
    n_random = config.n_random if args.n_random is None else args.n_random
    outs = {m: REPORT / "data" / args.band / m for m in mocks}
    for out in outs.values():
        out.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.band}: mocks {mocks}, n_random {n_random}, n_validation {n_validation}, backend {jax.default_backend()}",
        flush=True,
    )
    start = time.perf_counter()

    prior_mean, prior_cov = flow_prior(problem, seed)
    prior = (prior_mean, scaled_inverse(prior_cov))
    prior_sd = jnp.sqrt(jnp.diag(prior_cov))

    # eval.py: the eval grid's injections and their Fisher forecasts
    latents, windows, snrs, grid, noise_keys, prior_prec_p = eval_injections(
        problem, seed
    )
    n_eval = len(grid)
    # scorecard.py: the same, then n_random plain prior draws on the seed's stream 4
    key_window, key_pick, key_noise = jr.split(jr.fold_in(jr.key(seed), 4), 3)
    random_windows = jax.vmap(partial(sample_f, problem))(jr.split(key_window, n_random))
    random_latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, n_random), random_windows
    )
    score_latents = jnp.concatenate([latents, random_latents])
    score_windows = jnp.concatenate([windows, random_windows])
    score_keys = jnp.concatenate([noise_keys, jr.split(key_noise, n_random)])
    print(
        f"candidates and prior ready, {time.perf_counter() - start:.0f} s", flush=True
    )

    # the eval injections are the scorecard's first n_eval: one loop, 1024 draws there, 256 elsewhere
    draws = {m: [] for m in mocks}
    matched = {m: [] for m in mocks}
    rows = {m: [] for m in mocks}
    fisher, source_snr, injection, is_eval = [], [], [], []
    for n, key_n in enumerate(score_keys):
        latent, f = score_latents[n], score_windows[n]
        _, centre, cov = observe(problem, key_n, latent, f, prior)
        keys = jr.split(key_n, N_POSTERIOR)
        for m in mocks:
            if n < n_eval:
                d = np.asarray(mock_draws(problem, m, keys, f, centre, cov, prior_sd))
                draws[m].append(d)
                matched[m].append(match_sources(problem, d, np.asarray(latent), f))
                d = d[:SCORE_DRAWS]
            else:
                d = np.asarray(
                    mock_draws(problem, m, keys[:SCORE_DRAWS], f, centre, cov, prior_sd)
                )
            rows[m].append(score(problem, d, np.asarray(latent), f))
        if n < n_eval:
            fisher.append(fisher_samples(problem, latent, f, key_n, prior_prec_p))
            source_snr.append(np.asarray(snr(problem, latent[:, None, :], f)))
        injection += [n] * problem.n_sources
        is_eval += [n < n_eval] * problem.n_sources
        if (n + 1) % 20 == 0 or n + 1 == n_eval:
            print(
                f"[{n + 1}/{len(score_keys)}] scorecard injections, {time.perf_counter() - start:.0f} s",
                flush=True,
            )

    checks, cards = {m: {} for m in mocks}, {}
    for m in mocks:
        out = outs[m]
        write_eval_yaml(out, n_random, n_validation)
        d = dict(
            draws=np.stack(draws[m]),
            matched=np.stack(matched[m]),
            fisher=np.stack(fisher),
            truth=np.asarray(latents),
            f=np.asarray(windows),
            snr=snrs,
            source_snr=np.stack(source_snr),
            grid=grid,
            epoch=-1,
        )
        np.savez(out / "draws.npz", **d)
        checks[m]["width_over_fisher"] = width_ratios(problem, d)
        r = {k: np.concatenate([row[k] for row in rows[m]]) for k in rows[m][0]}
        r["injection"], r["is_eval"] = np.array(injection), np.array(is_eval)
        cards[m] = dict(
            epoch=-1,
            ode_steps=0,
            n_random=n_random,
            t_obs=problem.t_obs,
            **{f"card_{k}": v for k, v in r.items()},
        )
        checks[m]["scorecard"] = scorecard_checks(r)
    print(
        f"draws.npz saved, {time.perf_counter() - start:.0f} s",
        flush=True,
    )

    # validate.sample: n_validation plain prior injections, 127 draws each, every test quantity
    samples = {m: [] for m in mocks}
    for j, key in enumerate(jr.split(jr.fold_in(jr.key(seed), 1), n_validation)):
        f, truth, o, centre, cov, keys, tarp_ref = validation_injection(
            problem, key, prior
        )
        for m in mocks:
            d = mock_draws(problem, m, keys, f, centre, cov, prior_sd)
            q = quantities(problem, jnp.concat([truth[None], d]), o, f)
            samples[m].append(
                jax.tree.map(np.asarray, dict(f=f, tarp_ref=tarp_ref, **q))
            )
        if (j + 1) % 25 == 0:
            print(
                f"[{j + 1}/{n_validation}] validation injections, {time.perf_counter() - start:.0f} s",
                flush=True,
            )

    for m in mocks:
        out = outs[m]
        rows_v = {
            k: np.stack([s[k] for s in samples[m]]) for k in sorted(samples[m][0])
        }
        print(f"\n===== {args.band} / {m} =====", flush=True)
        validation = pack_validation(rows_v)
        validation |= validate.summarise(unpack(validation), seed, f"{args.band} / {m}")
        np.savez(out / "scorecard.npz", **cards[m], **validation)
        checks[m]["validation"] = validation_checks(out)
        record = dict(
            mock=m,
            band=args.band,
            description=DESCRIPTIONS[m],
            seed=seed,
            problem=config.problem,
            n_eval=n_eval,
            eval_draws=N_POSTERIOR,
            n_random=n_random,
            score_draws=SCORE_DRAWS,
            n_validation=n_validation,
            validation_draws=validate.N_DRAWS,
            wide=WIDE,
            narrow=NARROW,
            floor_bins=FLOOR_BINS,
            floor_snr=FLOOR_SNR,
            checks=checks[m],
        )
        (out / "mock.json").write_text(json.dumps(record, indent=1, default=float))
    print(f"\ndone, {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
