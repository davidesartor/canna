"""How far a 1-sigma (MCMC) shift of each parameter moves the data tokens, in bf16 ulps (F49).

LisaFlow embeds each data token in float32 (PLE + position) and casts it to bf16 for the
backbone. For every eval source with a narrowband MCMC, on the eval's own noise realisation,
each physical parameter is shifted by its MCMC standard deviation (psi and phi0 folded onto
the truth's copy, the longitude wrapped around it), and the most affected token's change is
reported in units of its own bf16 spacing: max and rms over the hidden components. Below ~1
rms ulp a feature is under the cast's rounding.

    .venv/bin/python lisa_checks/token_ulps.py B-fd-rope-long <dir holding lisa-B-fd-rope-long>
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

config, out_dir = sys.argv[1], sys.argv[2]
sys.argv = ["token_ulps", "--config", config, "--output_dir", out_dir, *sys.argv[3:]]
import canna.lisa.train as T
from canna.lisa import mcmc as M
from canna.lisa.eval import eval_injections, fold_orientation

args = T.parse_args()
state, epoch, _ = T.load_trained(args)
flow, problem = state.flow, state.problem
latents, windows, _, _, noise_keys, _ = eval_injections(problem, args.seed)
chains = M.load_chains(problem, args.seed, n=10**9)
snr = np.asarray(jax.vmap(problem.source_snrs)(latents, windows))


def tokens(p, f, noise):
    y = problem.preprocess(problem.clean_signal(p, f) + noise, f)
    y = rearrange(y.astype(jnp.float32), "(n p) c -> n (p c)", p=flow.patch)
    return jax.vmap(flow.y_embed)(y) + flow.y_pos


def ulp(e):  # bf16 spacing at each component
    e16 = jnp.abs(e).astype(jnp.bfloat16)
    return (jnp.nextafter(e16, jnp.array(jnp.inf, jnp.bfloat16)) - e16).astype(jnp.float32)


print(f"{config} epoch {epoch}: 1 MCMC sd moves the most affected data token by (max / rms) bf16 ulps")
names = ["f0", "Mc", "A", "lon", "lat", "psi", "iota", "phi0"]
print("source  SNR    " + "".join(f"{n:>14s}" for n in names if n != "Mc"))
for (j, s), chain in sorted(chains.items(), key=lambda kv: -snr[kv[0]]):
    p, f = latents[j], windows[j]
    truth = np.asarray(p[s])
    noise = problem.sample_observation(noise_keys[j], p, f) - problem.clean_signal(p, f)
    e0 = tokens(p, f, noise)
    spacing = ulp(e0)
    folded = fold_orientation(chain, truth)
    folded[:, 3] = truth[3] + (folded[:, 3] - truth[3] + np.pi) % (2 * np.pi) - np.pi
    sd = folded.std(0)
    row = []
    for c, name in enumerate(names):
        if name == "Mc":
            continue
        r = jnp.abs(tokens(p.at[s, c].add(sd[c]), f, noise) - e0) / spacing
        worst = jnp.argmax(jnp.linalg.norm(r * spacing, axis=-1))
        row.append(f"{float(r[worst].max()):7.2f}/{float(jnp.sqrt(jnp.mean(r[worst] ** 2))):<6.2f}")
    print(f"{j}_{s}  {snr[j, s]:7.1f} " + "".join(f"{v:>14s}" for v in row))
