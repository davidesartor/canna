"""The eval injections' Fisher precision per band, as injections.fisher_samples builds it: its diagonal's spread and f0 width.

Run from the repo root (JAX: a job, not the login node):
    uv run python reports/mcmc/scripts/fisher_diagonal.py reports/mcmc/data/fisher-diagonal.json
"""

import json
import sys
import time
from functools import partial

import jax
import jax.random as jr
import numpy as np

from canna.lisa import config
from canna.lisa.injections import N_FISHER_DRAWS, N_POSTERIOR, eval_injections
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import log_likelihood, sample_observation, source_snrs

BANDS = ("narrow", "full")
PARAMS = ["f0", "Mc", "A", "ra", "dec", "psi", "iota", "phi0"]


def averaged_hessian(problem: LisaGB, latent, f, key_n) -> np.ndarray:
    """fisher_samples' nll Hessian at the truth, averaged over its N_FISHER_DRAWS noise realisations, before the prior."""
    p0 = latent.reshape(-1)
    nll = lambda p, o: -log_likelihood(problem, p.reshape(latent.shape), o, f)
    replicas = jax.vmap(partial(sample_observation, problem), in_axes=(0, None, None))(
        jr.split(jr.split(jr.fold_in(key_n, N_POSTERIOR))[0], N_FISHER_DRAWS), latent, f
    )
    return np.asarray(jax.lax.map(lambda o: jax.hessian(nll)(p0, o), replicas).mean(0))


def forecast_covariance(
    latent: np.ndarray, precision: np.ndarray, prior_precision: np.ndarray
) -> tuple[np.ndarray, int]:
    """The covariance fisher_draws samples in (.., log Mc, ..), and how many eigenvalues its prior floor lifts."""
    jac = np.ones_like(latent)
    jac[..., 1] = latent[..., 1]
    jac = jac.reshape(-1)
    precision = (precision + precision.T) / 2 * np.outer(jac, jac)
    prior_precision = prior_precision * np.outer(jac, jac)
    d = np.sqrt(np.abs(np.diag(precision)))
    w, v = np.linalg.eigh(precision / np.outer(d, d))
    prior_w = np.einsum("ik,ij,jk->k", v, prior_precision / np.outer(d, d), v)
    floored = int(np.sum(w < prior_w))
    w = np.maximum(w, prior_w)
    return (v / w) @ v.T / np.outer(d, d), floored


def decades(diagonal: np.ndarray) -> float:
    return float(np.log10(diagonal.max() / diagonal.min()))


def main(out: str) -> None:
    result = dict(params=PARAMS, fisher_draws=N_FISHER_DRAWS, bands={})
    for band in BANDS:
        cfg = config.load(f"{band}-B")
        problem = LisaGB(**cfg.problem)
        start = time.perf_counter()
        latents, windows, snrs, quantiles, noise_keys, prior_prec = eval_injections(
            problem, cfg.seed
        )
        prior_diag = np.diag(np.asarray(prior_prec))
        injections = []
        for j, key_n in enumerate(noise_keys):
            hessian = averaged_hessian(problem, latents[j], windows[j], key_n)
            hessian_diag = np.diag(hessian)
            precision_diag = hessian_diag + prior_diag
            latent = np.asarray(latents[j])
            covariance, floored = forecast_covariance(
                latent, hessian + np.asarray(prior_prec), np.asarray(prior_prec)
            )
            f0_index = np.arange(latent.shape[0]) * latent.shape[1]
            source_snr = np.asarray(source_snrs(problem, latents[j], windows[j]))
            injections.append(
                dict(
                    quantile=float(quantiles[j]),
                    window_snr=float(snrs[j]),
                    source_snr=source_snr.tolist(),
                    log10_hessian_diag=np.log10(hessian_diag).tolist(),
                    log10_precision_diag=np.log10(precision_diag).tolist(),
                    hessian_decades=decades(hessian_diag),
                    precision_decades=decades(precision_diag),
                    # the forecast's f0 marginal and the monochromatic bound sqrt(3) / (pi snr), in bins
                    f0_width_bins=(
                        np.sqrt(np.diag(covariance)[f0_index]) * problem.t_obs
                    ).tolist(),
                    ideal_width_bins=(np.sqrt(3) / (np.pi * source_snr)).tolist(),
                    floored_eigenvalues=floored,
                )
            )
            print(
                f"{band} q{quantiles[j]:.2f}: window snr {snrs[j]:.0f},"
                f" precision diagonal spans {injections[-1]['precision_decades']:.1f} decades"
                f" (Hessian alone {injections[-1]['hessian_decades']:.1f})",
                flush=True,
            )
        spans = [i["precision_decades"] for i in injections]
        result["bands"][band] = dict(
            seed=cfg.seed,
            log10_prior_diag=np.log10(prior_diag).tolist(),
            injections=injections,
            median_precision_decades=float(np.median(spans)),
            median_hessian_decades=float(
                np.median([i["hessian_decades"] for i in injections])
            ),
            seconds=time.perf_counter() - start,
        )
        print(f"{band}: median {np.median(spans):.1f} decades", flush=True)
    every = [
        i["precision_decades"]
        for b in result["bands"].values()
        for i in b["injections"]
    ]
    result["median_precision_decades"] = float(np.median(every))
    print(f"both bands: median {result['median_precision_decades']:.1f} decades")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
