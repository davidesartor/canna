"""Does the network see the absolute phase of the data? (F52)

Multiplying the whole whitened window by e^{i alpha} (every channel, every bin) leaves the
noise distribution unchanged and shifts every source's phi0 by -alpha (jaxgb's e^{-i phi0}),
with every other parameter as it was. So the true posterior moves in phi0 only. On fresh
training samples, this prints, per flow coordinate, how much the predicted velocity changes
under the rotation (rms of the change over the rms of the target velocity), next to how much
it changes when the data are replaced by another sample's (a scale for "sees the data at all").
A network that reads phi0 changes its phi0 velocity under the rotation; one that has become
blind to the global phase changes nothing.

    .venv/bin/python lisa_checks/phase_rotation.py B-stft <dir holding lisa-B-stft> [flags]
"""
import sys
import warnings
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

config, out = sys.argv[1], sys.argv[2]
sys.argv = ["phase_rotation", "--config", config, "--output_dir", out, *sys.argv[3:]]
from canna.lisa import train as T

args = T.parse_args()
state, epoch, _ = T.load_trained(args)
problem, flow = state.problem, state.flow
names = problem.flow_names if hasattr(problem, "flow_names") else [str(i) for i in range(11)]
sample = jax.jit(jax.vmap(partial(T.train_sample, problem, time_power=args.time_power)))
# with the template residual (flow.templates) the conditioning is [data, data - h(x_t)]: rotate the
# data, then rebuild the residual from the rotated data
condition = jax.jit(jax.vmap(
    lambda xt, y, f: T.with_residual(problem, xt, y, f) if flow.templates else y))
velocity = eqx.filter_jit(lambda f, xt, t, y, w: jax.vmap(f)(xt, t, y, w)[0])


def rotate(y, alpha):
    c = y.shape[-1] // 2
    re, im = y[..., :c], y[..., c:]
    return jnp.concat([re * jnp.cos(alpha) - im * jnp.sin(alpha), re * jnp.sin(alpha) + im * jnp.cos(alpha)], axis=-1)


rows = {}
for k in range(8):
    b = sample(jr.split(jr.key(300 + k), 64))
    data = b.y
    v0 = velocity(flow, b.xt, b.t, condition(b.xt, data, b.f), b.f)
    out_rows = {"target": b.dx}
    for alpha in (0.3, np.pi / 2, np.pi):
        out_rows[f"rotate {alpha:.2f}"] = velocity(flow, b.xt, b.t, condition(b.xt, rotate(data, alpha), b.f), b.f) - v0
    # same x_t, t, f, other data: how much the output depends on the data at all
    out_rows["other data"] = velocity(flow, b.xt, b.t, condition(b.xt, jnp.roll(data, 1, axis=0), b.f), b.f) - v0
    for name, v in out_rows.items():
        rows.setdefault(name, []).append(np.asarray(v).reshape(-1, 11))
target_rms = np.concatenate(rows.pop("target")).std(0)
print(f"{config} epoch {epoch}: rms change of the predicted velocity / rms of the target velocity")
print(f"{'':16s}" + " ".join(f"{n:>9s}" for n in names))
for name, v in rows.items():
    rms = np.sqrt((np.concatenate(v) ** 2).mean(0))
    print(f"{name:16s}" + " ".join(f"{x:9.3f}" for x in rms / target_rms))
