"""The flow's draws on the ten eval injections, matched to the truth, beside their Fisher forecasts, as draws.npz (packed, see samples)."""

from functools import partial

import jax
import jax.random as jr
import numpy as np

from . import config as run_config
from .chart import flow_to_physical, sample_flow
from .injections import (
    N_POSTERIOR,
    N_QUANTILES,
    eval_injections,
    fisher_samples,
    match_sources,
)
from .sample import sample_posterior
from .samples import pack_draws
from .simulator import preprocess, sample_observation, snr
from .train import load_trained, print_peak_memory


def main() -> None:
    config = run_config.load_run()
    problem, flow, epoch = load_trained(config)
    out_dir = run_config.eval_dir(config, epoch, backend=jax.default_backend())
    latents, windows, snrs, quantiles, noise_keys, prior_prec_p = eval_injections(
        problem, config.seed
    )

    draws, matched, fisher, source_snr = [], [], [], []
    for j, key_n in enumerate(noise_keys):
        latent, f = latents[j], windows[j]

        # inject, sample the flow, and map back to physical units: [draw, source, param]
        y = preprocess(problem, sample_observation(problem, key_n, latent, f), f)
        u0 = jax.vmap(partial(sample_flow, problem), in_axes=(0, None))(
            jr.split(key_n, N_POSTERIOR), f
        )
        post = sample_posterior(flow, u0, y, f, config.ode_steps, config.time_power)
        draws.append(
            np.asarray(
                jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
            )
        )
        matched.append(match_sources(problem, draws[-1], np.asarray(latent), f))
        fisher.append(fisher_samples(problem, latent, f, key_n, prior_prec_p))
        source_snr.append(np.asarray(snr(problem, latent[:, None, :], f)))
        print(f"[{j + 1}/{N_QUANTILES}] q{quantiles[j]:.2f} sampled", flush=True)

    path = out_dir / "draws.npz"
    np.savez(
        path,
        **pack_draws(
            dict(
                draws=np.stack(draws),
                matched=np.stack(matched),
                fisher=np.stack(fisher),
                truth=np.asarray(latents),
                f=np.asarray(windows),
                snr=snrs,
                source_snr=np.stack(source_snr),
                quantiles=quantiles,
                epoch=epoch,
            )
        ),
    )
    print(f"saved the draws to {path}", flush=True)
    print_peak_memory()


if __name__ == "__main__":
    main()
