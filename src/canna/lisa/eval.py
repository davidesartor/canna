"""The flow's draws on the eval grid's injections, matched to the truth, beside their Fisher forecasts, as draws.npz (packed, see samples)."""

from functools import partial

import jax
import jax.random as jr
import numpy as np

from . import config as run_config
from .chart import flow_to_physical
from .injections import (
    N_POSTERIOR,
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
    latents, windows, snrs, grid, noise_keys, prior_prec_p = eval_injections(
        problem, config.seed
    )

    draws, matched, fisher, source_snr = [], [], [], []
    for j, key_n in enumerate(noise_keys):
        latent, f = latents[j], windows[j]

        # inject, sample the flow, and map back to physical units: [draw, source, param]
        y = preprocess(problem, sample_observation(problem, key_n, latent, f), f)
        # the base draws: their own stream beside the noise's key_n
        z0 = jr.normal(jr.fold_in(key_n, 1), (N_POSTERIOR, *latent.shape))
        post = sample_posterior(problem, flow, z0, y, f, config.ode_steps)
        draws.append(
            np.asarray(
                jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
            )
        )
        matched.append(match_sources(problem, draws[-1], np.asarray(latent), f))
        fisher.append(fisher_samples(problem, latent, f, key_n, prior_prec_p))
        source_snr.append(np.asarray(snr(problem, latent[:, None, :], f)))
        print(
            f"[{j + 1}/{len(grid)}] window {grid[j, 0]}, SNR rank {grid[j, 1]} sampled",
            flush=True,
        )

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
                grid=grid,
                epoch=epoch,
            )
        ),
    )
    print(f"saved the draws to {path}", flush=True)
    print_peak_memory()


if __name__ == "__main__":
    main()
