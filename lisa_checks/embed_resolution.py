"""How finely the bf16 cast of a run's source embedding still resolves each coordinate (F33, F38).

LisaFlow computes the source embedding in float32 and casts it to bf16 for the backbone. The
cast error, read along the embedding's gradient in a coordinate, is how far that coordinate
would have to move to produce it: its resolution after the cast (if the next layer reads along
the gradient). Reported for f0 in frequency bins of the run's window, and for A, sky and
orientation in flow-coordinate units, over prior draws.

    .venv/bin/python lisa_checks/embed_resolution.py XS-fd outputs/sfplus [--weights training]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import jax.random as jr

config, out_dir, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["x", "--config", config, "--output_dir", out_dir, *extra]
import canna.lisa.train as T

args = T.parse_args()
state, epoch, _ = T.load_trained(args)
flow, problem = state.flow, state.problem
f = jax.vmap(problem.sample_f)(jr.split(jr.key(5), 1000))
x = jax.vmap(problem.sample_flow)(jr.split(jr.key(6), 1000), f).reshape(-1, 11)[:4000].astype(jnp.float32)
emb = jax.vmap(flow.x_embed)(x)
err = emb.astype(jnp.bfloat16).astype(jnp.float32) - emb
# one unit of x_f in frequency bins: the f0 window spans 2 units (F26)
bins_per_unit = (problem.window_bins - 2 * (problem.response_points // 2)) / 2
print(f"{config} ({out_dir}), epoch {epoch}: window {problem.window_bins} bins, 1 unit of x_f ~ {bins_per_unit:.0f} bins")
for i, name in [(0, "x_f"), (2, "x_a"), (3, "sky_x"), (6, "orient_x")]:
    g = jax.vmap(jax.jacfwd(flow.x_embed))(x)[:, :, i]
    along = jnp.abs(jnp.sum(err * g, -1)) / jnp.sum(g * g, -1)
    worst = jnp.linalg.norm(err, axis=-1) / jnp.linalg.norm(g, axis=-1)
    unit = bins_per_unit if i == 0 else 1.0
    tag = "bins" if i == 0 else "coord units"
    print(f"  {name:9s} resolution along the gradient: median {float(jnp.median(along)) * unit:.2e} {tag},"
          f" worst case (all error against it) {float(jnp.median(worst)) * unit:.2e}")
