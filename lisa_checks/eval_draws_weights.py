"""eval.py's ten injections, sampled with a run's training weights y alone.

For a checkpoint whose optimizer state did not arrive (an interrupted transfer): restores
only `flow`, the trained weights, and draws as outputs/lisa-XS/eval-tools/eval_draws.py does
(the same keys, injections and base draws), for compare_mcmc.py.

    .venv/bin/python lisa_checks/eval_draws_weights.py XS-fd outputs/lisa-XS-sfplus [out.npz]
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import orbax.checkpoint as ocp

config, run = sys.argv[1], Path(sys.argv[2])
out_file = Path(sys.argv[3]) if len(sys.argv) > 3 else run / "eval_draws_training.npz"
sys.argv = ["x", "--config", config]
import canna.lisa.train as T
from canna.lisa.eval import N_CANDIDATES, N_POSTERIOR, N_QUANTILES, ODE_STEPS, sample_posterior

args = T.parse_args()
fresh = T.TrainState.from_config(args)
manager = ocp.CheckpointManager((run / "checkpoints").absolute())
epoch = manager.latest_step()
restored = manager.restore(
    epoch, args=ocp.args.Composite(flow=ocp.args.StandardRestore(eqx.filter(fresh.flow, eqx.is_array)))
)
flow, problem = eqx.combine(restored["flow"], fresh.flow), fresh.problem
print(f"{run} epoch {epoch}: training weights y", flush=True)

key_pick, key_window, key_noise = jr.split(jr.key(args.seed), 3)
windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
latents, snrs, windows = latents[chosen], snrs[chosen], windows[chosen]
draws, start = [], time.perf_counter()
for j, key_n in enumerate(jr.split(key_noise, N_QUANTILES)):
    latent, f = latents[j], windows[j]
    y = problem.preprocess(problem.sample_observation(key_n, latent, f), f)
    u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(jr.split(key_n, N_POSTERIOR), f)
    post = sample_posterior(problem, flow, u0, y, f, ODE_STEPS, args.time_power)
    draws.append(np.asarray(jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f)))
    print(f"  injection {j + 1}/{N_QUANTILES}, {time.perf_counter() - start:.0f} s", flush=True)
np.savez(out_file, draws=np.stack(draws), truth=np.asarray(latents), snr=snrs, quantiles=quantiles, epoch=epoch)
print(f"saved {out_file}")
