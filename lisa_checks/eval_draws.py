"""The ten eval injections' posterior draws of one local run, with eval.py's keys.

The draw recipe of backfill_corner.py, for any run (e.g. XS-pos, whose eval job on TREX writes
corner pages but no draws): the same injections, noise and base draws, so the result is what
the run's eval saw and can be compared with the MCMC of lisa_checks/mcmc_xs.py. Writes
<output_dir>/lisa-<config>/eval_draws.npz ([injection, draw, source, param], physical units).

Run from the repo root (extra flags go to the run's parse_args, e.g. --weights training):
    .venv/bin/python lisa_checks/eval_draws.py XS-fd [outputs [out.npz [flags]]]
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.random as jr
import numpy as np

from canna.lisa.eval import N_CANDIDATES, N_POSTERIOR, N_QUANTILES, ODE_STEPS, sample_posterior
from canna.lisa.train import load_trained, parse_args

config = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
out_file = sys.argv[3] if len(sys.argv) > 3 else None  # default <run>/eval_draws.npz
sys.argv = ["eval_draws", "--config", config, "--output_dir", output_dir, *sys.argv[4:]]
args = parse_args()
state, epoch, out_dir = load_trained(args)
problem = state.problem
print(f"{out_dir} epoch {epoch}, time_power {args.time_power}, flow {type(state.flow).__name__}", flush=True)

key_pick, key_window, key_noise = jr.split(jr.key(args.seed), 3)
windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
latents, snrs, windows = latents[chosen], snrs[chosen], windows[chosen]
noise_keys = jr.split(key_noise, N_QUANTILES)
truths = np.asarray(latents)

all_draws = []
start = time.perf_counter()
for j, key_n in enumerate(noise_keys):
    latent, f = latents[j], windows[j]
    y = problem.preprocess(problem.sample_observation(key_n, latent, f), f)
    u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(jr.split(key_n, N_POSTERIOR), f)
    post = sample_posterior(problem, state.flow, u0, y, f, ODE_STEPS, args.time_power)
    all_draws.append(np.asarray(jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f)))
    print(f"  injection {j + 1}/{N_QUANTILES}, {time.perf_counter() - start:.0f} s", flush=True)
all_draws = np.stack(all_draws)
out_file = out_file or out_dir / "eval_draws.npz"
np.savez(out_file, draws=all_draws, truth=truths, snr=snrs, quantiles=quantiles, epoch=epoch)
print(f"saved {out_file} {all_draws.shape}")
