"""What a trained LisaFlow moves: per flow coordinate, the velocity it predicts against the target.

On 1,024 fresh training samples (one draw per observation, the velocity the sampler
follows), for the run's averaged weights x or, with LISA_WEIGHTS=training, its training weights y:
the predicted velocity's spread over the target's; its R^2 per band of path position s; the
phi+/phi- error of the x1 the flow implies next to x_t's own; the output head's last-layer row
norms; and the velocity's rms change when the whitened data are rotated by e^{i alpha} (which
moves every phi0 by -alpha and nothing else) or swapped for another sample's. The record goes to
a JSON that build_page.py, beside this script, turns into the data-dependence page's velocity-probe card.

Run from the repo root on the run's commit (slurm/run-src.sh), as velocity_probe.sbatch does:
    LISA_RUN=<run> [LISA_OUTPUT_DIR=...] uv run --frozen python reports/architecture/data-dependence/scripts/velocity_probe.py reports/architecture/data-dependence/data/velocity-probe/<tag>.json
LISA_OUTPUT_DIR holds runs/<run>.
"""

import json
import sys
import warnings
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange

from canna.lisa import config
from canna.lisa.train import load_trained, train_sample

COORDINATES = ["f0", "Mc", "l", "b", "Re z+", "Im z+", "Re z-", "Im z-"]
PHASES = {"φ+": 4, "φ-": 6}
S_BANDS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0)]
ALPHAS = (0.3, np.pi / 2, np.pi)
BATCHES, BATCH = 16, 64


def rotate(y, alpha):
    """The whitened window times e^{i alpha}: [re A, E, T, im A, E, T] per bin."""
    z = rearrange(y, "... f (k c) -> ... f k c", k=2)
    re, im = z[..., 0, :], z[..., 1, :]
    rotated = jnp.stack(
        [
            re * jnp.cos(alpha) - im * jnp.sin(alpha),
            re * jnp.sin(alpha) + im * jnp.cos(alpha),
        ],
        axis=-2,
    )
    return rearrange(rotated, "... f k c -> ... f (k c)")


def angle(x, c):
    return np.arctan2(x[..., c + 1], x[..., c])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main(out: str) -> None:
    cfg = config.load_run()
    problem, flow, epoch = load_trained(cfg)
    sample = jax.jit(jax.vmap(partial(train_sample, problem)))

    @eqx.filter_jit
    def velocity(flow, b, y):
        # one draw per observation; the sampler's velocity (m - xt) / (1 - t) and the flow's mean m of x1
        xt, t = b.xt[:, 0], b.t[:, 0]
        m, _ = jnp.split(jax.vmap(flow)(xt, t, y, b.f), 2, axis=-1)
        return (m - xt) / (1 - t)[:, None, None], m

    # same x_t, t, f: the data rotated, or another sample's data
    perturbations = {
        **{f"rotate {a:.2f}": partial(rotate, alpha=a) for a in ALPHAS},
        "other data": lambda y: jnp.roll(y, 1, axis=0),
    }
    parts = {k: [] for k in ("predicted", "implied", "target", "xt", "x1", "t")}
    changes = {name: [] for name in perturbations}
    for k in range(BATCHES):
        b = sample(jr.split(jr.key(200 + k), BATCH))
        predicted, implied = velocity(flow, b, b.y)
        for name, v in zip(
            parts, (predicted, implied, b.dx[:, 0], b.xt[:, 0], b.x_target, b.t[:, 0])
        ):
            parts[name].append(np.asarray(v))
        for name, perturb in perturbations.items():
            change = velocity(flow, b, perturb(b.y))[0] - predicted
            changes[name].append(np.asarray(change).reshape(-1, len(COORDINATES)))
    P, X, D, XT, X1, T = (np.concatenate(v) for v in parts.values())
    S = T
    target_sd = D.reshape(-1, len(COORDINATES)).std(0)

    bands = []
    for lo, hi in S_BANDS:
        m = (S >= lo) & (S < hi)
        p, t = P[m].reshape(-1, len(COORDINATES)), D[m].reshape(-1, len(COORDINATES))
        error = lambda x, c: float(
            np.median(np.abs(wrap(angle(x[m], c) - angle(X1[m], c))))
        )
        bands.append(
            dict(
                lo=lo,
                hi=hi,
                n=int(m.sum()),
                r2=(1 - ((p - t) ** 2).mean(0) / t.var(0)).tolist(),
                head_error={n: error(X, c) for n, c in PHASES.items()},
                xt_error={n: error(XT, c) for n, c in PHASES.items()},
            )
        )
    rows = np.linalg.norm(np.asarray(flow.x_unembed.linear2.weight), axis=-1)
    record = dict(
        run=cfg.run,
        config=cfg.name,
        epoch=epoch,
        weights=cfg.weights,
        device=jax.devices()[0].device_kind,
        samples=len(S),
        coordinates=COORDINATES,
        spread=(P.reshape(-1, len(COORDINATES)).std(0) / target_sd).tolist(),
        bands=bands,
        head_rows=(rows / np.median(rows)).tolist(),
        changes={
            name: (np.sqrt((np.concatenate(v) ** 2).mean(0)) / target_sd).tolist()
            for name, v in changes.items()
        },
    )
    with open(out, "w") as fh:
        json.dump(record, fh, indent=1)

    row = lambda label, values: print(
        f"{label:<20}" + " ".join(f"{v:8.3f}" for v in values)
    )
    print(
        f"{cfg.name} epoch {epoch} ({cfg.weights} weights), {len(S)} samples -> {out}"
    )
    print(f"{'coordinate':<20}" + " ".join(f"{n:>8s}" for n in COORDINATES))
    row("predicted / target", record["spread"])
    for band in bands:
        row(f"R^2, s {band['lo']:.2f}-{band['hi']:.2f}", band["r2"])
    row("head row / median", record["head_rows"])
    print("rms velocity change / target sd")
    for name, values in record["changes"].items():
        row(f"  {name}", values)
    print("median phase error [rad], head x1 / x_t")
    for band in bands:
        print(
            f"  s {band['lo']:.2f}-{band['hi']:.2f}  "
            + "  ".join(
                f"{n} {band['head_error'][n]:.3f} / {band['xt_error'][n]:.3f}"
                for n in PHASES
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
