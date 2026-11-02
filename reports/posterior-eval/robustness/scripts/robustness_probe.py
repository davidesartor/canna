"""The flow on misspecified windows: noise level, sources just outside the prior, a source fewer or more, no source.

Every case reuses the baseline's truth, window and noise key and changes one thing, so each matched
source pairs with its in-distribution self (slots and truth sources both by f0). Flow only: the case's data come from the repo's
simulator with modified parameters, the posterior from sample_posterior alone.
Run from the repo root as robustness.sbatch does:
    LISA_RUN=<run> PYTHONPATH=src python reports/posterior-eval/robustness/scripts/robustness_probe.py \
        [--draws 256] [--injections 128] [--ode-steps <run's>] [--cases baseline,noise x2,...]
The record goes to data/<run>.json, rewritten after every case.
"""

import argparse
import dataclasses
import itertools
import json
import time
import warnings
from datetime import date
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from canna.lisa import config as run_config
from canna.lisa.chart import flow_to_physical, physical_to_flow
from canna.lisa.injections import N_POSTERIOR, eval_injections, sort_by_f0
from canna.lisa.physics import FDOT_STD, log10_fdot_mean
from canna.lisa.problem import LisaGB, f0_window
from canna.lisa.sample import sample_posterior
from canna.lisa.simulator import (
    preprocess,
    sample_observation,
    sample_physical,
    sample_window,
    snr,
    source_snrs,
)
from canna.lisa.train import load_trained

REPORT = Path(__file__).parents[1]
N_CANDIDATES = 1024
COLUMNS = ["f0 [bins]", "fdot", "l", "b", "Re z+", "Im z+", "Re z−", "Im z−"]
CASES = {
    "baseline": "in distribution: the eval's injections as drawn",
    "noise ×0.5": "the noise PSD halved (oms and acceleration noise × √0.5); the flow whitens by its own PSD",
    "noise ×2": "the noise PSD doubled (oms and acceleration noise × √2); the flow whitens by its own PSD",
    "f0 past the edge": "the loudest source's f0 a quarter of the guard band past the window's drawable top",
    "fdot above the prior": "the loudest source's log10 fdot 5 sigma above the prior's mean at its f0",
    "SNR above the prior": "the loudest source at 1.5× the prior's 99.9% source SNR",
    "one source fewer": "the quietest source removed",
    "one source more": "a fifth source, one slot of another prior draw",
    "empty window": "noise only",
}


def misspecify(
    problem: LisaGB, case: str, latent: Float[Array, "S 8"], f, key, snr_top: float
) -> tuple[LisaGB, Float[Array, "T 8"], np.ndarray, np.ndarray, np.ndarray]:
    """The case's noise problem, truth [T, 8] by f0, which sources are present and perturbed, and each one's baseline slot."""
    n = latent.shape[0]
    present, perturbed, base_slot = np.ones(n, bool), np.zeros(n, bool), np.arange(n)
    noisy = problem
    own_snr = np.asarray(source_snrs(problem, latent, f))
    loudest, quietest = int(np.argmax(own_snr)), int(np.argmin(own_snr))
    scale_to = lambda p, s, target: p.at[s, 2].multiply(target / own_snr[s])
    if case.startswith("noise"):
        k = float(case.removeprefix("noise ×"))
        noisy = dataclasses.replace(
            problem,
            oms_noise=problem.oms_noise * k**0.5,
            acceleration_noise=problem.acceleration_noise * k**0.5,
        )
    elif case == "f0 past the edge":
        past = problem.response_points // 8 / problem.t_obs
        latent = latent.at[loudest, 0].set(f0_window(problem, f)[1] + past)
        perturbed[loudest] = True
    elif case == "fdot above the prior":
        latent = latent.at[loudest, 1].set(
            10.0 ** (log10_fdot_mean(latent[loudest, 0]) + 5 * FDOT_STD)
        )
        perturbed[loudest] = True
    elif case == "SNR above the prior":
        latent = scale_to(latent, loudest, 1.5 * snr_top)
        perturbed[loudest] = True
    elif case == "one source fewer":
        latent = latent.at[quietest, 2].set(0.0)
        present[quietest] = False
    elif case == "one source more":
        other = sample_physical(problem, key, f)
        latent = jnp.concat([latent, other[int(jr.randint(jr.fold_in(key, 1), (), 0, n))][None]])
        present, perturbed = np.append(present, True), np.append(perturbed, True)
        base_slot = np.append(base_slot, -1)
    elif case == "empty window":
        latent = latent.at[:, 2].set(0.0)
        present[:] = False
    else:
        assert case == "baseline", case
    # truth sources by f0, as eval's: the perturbed f0 and an appended source move
    order = np.argsort(np.asarray(latent[:, 0]))
    return noisy, latent[order], present[order], perturbed[order], base_slot[order]


def features(problem: LisaGB, p: np.ndarray, f) -> np.ndarray:
    """The matching feature: f0 over the window's width."""
    lo, hi = (float(v) for v in f0_window(problem, f))
    return p[..., :1] / (hi - lo)


def match_any(problem: LisaGB, draws: np.ndarray, truth: np.ndarray, f) -> np.ndarray:
    """Draws [N, D, 8] by f0 onto truths [T, 8] by f0, the order-preserving injective pairing nearest in f0; a truth left without a slot gets NaN."""
    fd, ft = features(problem, draws, f), features(problem, truth, f)
    n, d, t = draws.shape[0], draws.shape[1], truth.shape[0]
    if t <= d:
        # each truth picks a distinct slot
        perms = np.array(list(itertools.combinations(range(d), t)))
        cost = ((fd[:, perms] - ft[None, None]) ** 2).sum((-1, -2))
        best = perms[np.argmin(cost, axis=1)]
        return np.take_along_axis(draws, best[:, :, None], axis=1)
    # each slot picks a distinct truth
    perms = np.array(list(itertools.combinations(range(t), d)))
    cost = ((fd[:, None] - ft[perms][None]) ** 2).sum((-1, -2))
    best = perms[np.argmin(cost, axis=1)]
    aligned = np.full((n, t, draws.shape[2]), np.nan)
    aligned[np.arange(n)[:, None], best] = draws
    return aligned


def columns(problem: LisaGB, p: np.ndarray, f) -> np.ndarray:
    """Physical sources -> COLUMNS: f0 in bins, then the flow coordinates past f0."""
    x = np.asarray(physical_to_flow(problem, jnp.asarray(np.nan_to_num(p)), f))
    out = np.concatenate([p[..., :1] * problem.t_obs, x[..., 1:]], axis=-1)
    return np.where(np.isnan(p[..., :1]), np.nan, out)


def rounded(v) -> list:
    return np.vectorize(
        lambda x: None if np.isnan(x) else float(f"{x:.4g}"), otypes=[object]
    )(np.asarray(v, np.float64)).tolist()


def injection_set(problem: LisaGB, seed: int, n: int):
    """n of the eval's injections spread over the band, with the eval's noise keys, sources by f0.

    Also returns the 99.9% quantile of plain prior draws' per-source SNR, the top of the prior's loudness.
    """
    latents, windows, snrs, noise_keys, _ = eval_injections(problem, seed)
    key_pick, key_window = jr.split(jr.fold_in(jr.key(seed), 7))
    prior_windows = jax.vmap(partial(sample_window, problem))(jr.split(key_window, N_CANDIDATES))
    prior_latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, N_CANDIDATES), prior_windows
    )
    source_snr = np.asarray(
        jax.lax.map(lambda pc: source_snrs(problem, *pc), (prior_latents, prior_windows))
    )
    picked = np.linspace(0, len(windows) - 1, n).round().astype(int)
    return latents[picked], windows[picked], snrs[picked], list(noise_keys[picked]), float(np.quantile(source_snr, 0.999))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--injections", type=int, default=128)
    parser.add_argument("--ode-steps", type=int, default=None)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cases = [c for c in args.cases.split(",") if c]
    assert set(cases) <= set(CASES), f"unknown cases {set(cases) - set(CASES)}"

    config = run_config.load_run()
    problem, flow, epoch = load_trained(config)
    ode_steps = args.ode_steps or config.ode_steps
    out = Path(args.out or REPORT / "data" / f"{config.run}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    latents, windows, window_snrs, keys, snr_top = injection_set(
        problem, config.seed, args.injections
    )

    record = dict(
        run=config.run,
        config=config.name,
        commit=config.commit,
        epoch=int(epoch),
        total_steps=config.total_steps,
        weights=config.weights,
        bf16=config.bf16,
        device=jax.devices()[0].device_kind,
        ode_steps=ode_steps,
        draws=args.draws,
        n_sources=problem.n_sources,
        snr_top=round(snr_top, 3),
        date=str(date.today()),
        columns=COLUMNS,
        cases=[dict(name=c, what=CASES[c]) for c in cases],
        injections=[
            dict(
                index=j,
                eval=True,
                f=float(windows[j]),
                window_snr=round(float(window_snrs[j]), 3),
                f0_window_bins=round(
                    float(
                        np.diff([float(v) for v in f0_window(problem, windows[j])])[0]
                    )
                    * problem.t_obs,
                    3,
                ),
            )
            for j in range(len(keys))
        ],
        sources=[],
        slots=[],
    )
    save = lambda: out.write_text(json.dumps(record) + "\n")
    # the eval's base draws, one slot per flow slot whatever the case's source count
    u0s = [
        jr.normal(jr.fold_in(key_n, 1), (N_POSTERIOR, problem.n_sources, 8))[: args.draws]
        for key_n in keys
    ]

    mismatch = 0.0
    for case in cases:
        start = time.perf_counter()
        for j, key_n in enumerate(keys):
            f = windows[j]
            noisy, truth, present, perturbed, base_slot = misspecify(
                problem, case, latents[j], f, jr.fold_in(key_n, 7), snr_top
            )
            # the case's data, whitened by the PSD the flow was trained on
            y = preprocess(problem, sample_observation(noisy, key_n, truth, f), f)
            post = sample_posterior(problem, flow, u0s[j], y, f, ode_steps)
            draws = np.asarray(
                sort_by_f0(
                    jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
                )
            )
            truth = np.asarray(truth)
            seen_snr = np.asarray(source_snrs(noisy, jnp.asarray(truth), f))

            # in the baseline the nearest-f0 pairing should be the slot index
            if case == "baseline":
                same = match_any(problem, draws, truth, f) == draws
                mismatch = max(mismatch, float(1 - same.all(axis=(1, 2)).mean()))

            # each present true source against the draws paired to it
            if present.any():
                matched = columns(
                    problem, match_any(problem, draws, truth[present], f), f
                )
                true = columns(problem, truth[present], f)
                for k, s in enumerate(np.flatnonzero(present)):
                    a = matched[:, k]
                    found = ~np.isnan(a[:, 0])
                    a = a[found]
                    median = np.median(a, axis=0) if len(a) else np.full(8, np.nan)
                    record["sources"].append(
                        dict(
                            case=case,
                            injection=j,
                            slot=int(s),
                            base_slot=int(base_slot[s]),
                            perturbed=bool(perturbed[s]),
                            snr=round(float(seen_snr[s]), 3),
                            found=round(float(found.mean()), 4),
                            width=rounded(
                                1.4826 * np.median(np.abs(a - median), axis=0)
                            ),
                            offset=rounded(median - true[k]),
                            rank=rounded(np.mean(a < true[k], axis=0)),
                        )
                    )

            # what each of the flow's slots claims, matched or not: its f0 spread and its SNR
            claimed = np.asarray(
                jax.vmap(partial(source_snrs, problem), in_axes=(0, None))(
                    jnp.asarray(draws), f
                )
            )
            f0 = draws[..., 0] * problem.t_obs
            for k in range(draws.shape[1]):
                record["slots"].append(
                    dict(
                        case=case,
                        injection=j,
                        slot=k,
                        f0_width=round(
                            float(
                                1.4826
                                * np.median(np.abs(f0[:, k] - np.median(f0[:, k])))
                            ),
                            4,
                        ),
                        snr=round(float(np.median(claimed[:, k])), 3),
                    )
                )
        save()
        rows = [r for r in record["sources"] if r["case"] == case]
        width = np.median(
            [r["width"][0] for r in rows if r["width"][0] is not None] or [np.nan]
        )
        slot_snr = np.median([r["snr"] for r in record["slots"] if r["case"] == case])
        print(
            f"{case:22s} {time.perf_counter() - start:7.1f} s: {len(rows)} sources,"
            f" median f0 width {width:.4g} bins, median claimed slot SNR {slot_snr:.3g}",
            flush=True,
        )
    record["matcher_mismatch"] = mismatch
    save()
    print(f"baseline draws the f0 matcher pairs unlike the slot index: {mismatch:.3g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
