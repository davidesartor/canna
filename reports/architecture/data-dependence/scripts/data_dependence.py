"""How much a trained LisaFlow's velocity depends on the data, per source slot, coordinate, path time and SNR.

At fixed path times t, on fresh training observations (the velocity the sampler follows):
- shares: the velocity's spread when only the data change (another observation's y, same x_t) and
  when only x_t changes (another base draw for the same observation), each over twice its variance;
- sensitivity: the Jacobian of the velocity in the whitened data, its squared norm per window bin,
  split into the slot's own band (the bins holding 95% of its source's power), the other sources'
  bands and the rest, and against the distance from its own f0 bin.
The record goes to a JSON that build_page.py, beside this script, turns into the page.

Run from the repo root on the run's commit (slurm/run-src.sh), as data_dependence.sbatch does:
    LISA_RUN=<run> uv run --extra cuda --frozen python reports/architecture/data-dependence/scripts/data_dependence.py [tag]
"""

import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import reduce
from jaxtyping import Array, Float, Key

from canna.lisa import config
from canna.lisa.network import LisaFlow
from canna.lisa.problem import LisaGB, window_power, window_start
from canna.lisa.simulator import (
    sample_physical,
    sample_window,
    source_segments,
    source_snrs,
)
from canna.lisa.train import load_trained, train_sample

REPORT = Path(__file__).parents[1]
COORDINATES = ["f0", "Mc", "l", "b", "Re z+", "Im z+", "Re z-", "Im z-"]
T_GRID = (0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
SNR_EDGES = (7.0, 20.0, 60.0, 200.0, 1000.0)
BAND_POWER = 0.95
SHARE_BATCHES, SHARE_BATCH = 8, 64
JACOBIAN_OBSERVATIONS, JACOBIAN_BATCH = 32, 8
EXAMPLE_COLUMNS = 512


def source_bands(
    problem: LisaGB, key: Key[Array, ""]
) -> tuple[Float[Array, " S"], Float[Array, "S F"], Float[Array, " S"]]:
    """train_sample's sources for this key: their snr, whitened power per window bin and f0 bin."""
    key_c, key_p = jr.split(key, 5)[:2]
    f = sample_window(problem, key_c)
    p = sample_physical(problem, key_p, f)
    idxs, segments = source_segments(problem, p, f)
    power = reduce(
        2
        * jnp.abs(segments) ** 2
        / window_power(problem, f)[jnp.clip(idxs, 0, problem.window_bins - 1)],
        "s n c -> s n",
        "sum",
    )
    rows = jnp.arange(problem.n_sources)[:, None]
    dense = (
        jnp.zeros((problem.n_sources, problem.window_bins)).at[rows, idxs].add(power)
    )
    return (
        source_snrs(problem, p, f),
        dense,
        p[:, 0] * problem.t_obs - window_start(problem, f),
    )


def band_masks(power: np.ndarray) -> np.ndarray:
    """[..., S, F] -> each source's smallest set of bins holding BAND_POWER of its power."""
    order = np.argsort(-power, axis=-1)
    sorted_power = np.take_along_axis(power, order, -1)
    before = np.cumsum(sorted_power, -1) - sorted_power
    keep = before < BAND_POWER * sorted_power.sum(-1, keepdims=True)
    mask = np.zeros_like(keep)
    np.put_along_axis(mask, order, keep, -1)
    return mask


def velocity(
    flow: LisaFlow,
    xt: Float[Array, "S X"],
    t: Float[Array, ""],
    y: Float[Array, "F 6"],
    f: Float[Array, ""],
) -> Float[Array, "S X"]:
    """The velocity (m - xt) / (1 - t) the sampler follows, m the flow's mean of x1."""
    m, _ = jnp.split(flow(xt, t, y, f), 2, axis=-1)
    return (m - xt) / (1 - t)


def offset_rings(window_bins: int) -> list[int]:
    """|bin - f0 bin| ring edges: 0, 1, 2, 4, ... up to the window."""
    edges = [0, 1]
    while edges[-1] < window_bins:
        edges.append(2 * edges[-1])
    return edges


def main(tag: str | None) -> None:
    cfg = config.load_run()
    out = REPORT / "data" / f"{tag or cfg.run}.json"
    problem, flow, epoch = load_trained(cfg)
    n_bins, n_slots, n_coords = problem.window_bins, problem.n_sources, len(COORDINATES)
    started = time.perf_counter()

    v = velocity
    draws = jax.jit(
        jax.vmap(
            lambda key, t: train_sample(problem, key, jnp.full((2,), t)),
            in_axes=(0, None),
        )
    )
    bands = jax.jit(jax.vmap(partial(source_bands, problem)))
    @eqx.filter_jit
    def batched(flow, xt, t, y, f):
        return jax.vmap(lambda xt, y, f: v(flow, xt, t, y, f))(xt, y, f)

    @eqx.filter_jit
    def jacobian(flow, xt, t, y, f):
        return jax.vmap(lambda xt, y, f: jax.jacrev(lambda y: v(flow, xt, t, y, f))(y))(xt, y, f)

    # every t sees the same observations: their sources' snr, bands and f0 bins once
    keys = [jr.split(jr.key(300 + k), SHARE_BATCH) for k in range(SHARE_BATCHES)]
    snr, power, f0_bin = (
        np.concatenate(a) for a in zip(*(map(np.asarray, bands(k)) for k in keys))
    )
    own = band_masks(power)
    others = np.stack(
        [
            own[:, [j for j in range(n_slots) if j != s]].any(1) & ~own[:, s]
            for s in range(n_slots)
        ],
        1,
    )
    snr_bin = np.clip(np.digitize(snr, SNR_EDGES) - 1, 0, len(SNR_EDGES) - 2)
    rings = offset_rings(n_bins)
    stride = max(1, n_bins // EXAMPLE_COLUMNS)
    ring = np.clip(
        np.searchsorted(
            rings, np.abs(np.arange(n_bins) - f0_bin[..., None]), side="right"
        )
        - 1,
        0,
        len(rings) - 2,
    )
    ring_counts = np.stack([(ring == r).sum(-1) for r in range(len(rings) - 1)], -1)

    shares, jac = [], []
    for t in T_GRID:
        # the velocity, with x_t swapped for another base draw, with the data swapped for another observation's
        parts = {k: [] for k in ("v", "x", "y")}
        jac_maps = []
        for k, batch_keys in enumerate(keys):
            b = draws(batch_keys, t)
            tt = jnp.asarray(t)
            parts["v"].append(np.asarray(batched(flow, b.xt[:, 0], tt, b.y, b.f)))
            parts["x"].append(np.asarray(batched(flow, b.xt[:, 1], tt, b.y, b.f)))
            parts["y"].append(
                np.asarray(batched(flow, b.xt[:, 0], tt, jnp.roll(b.y, 1, axis=0), b.f))
            )
            for lo in range(
                0,
                min(JACOBIAN_OBSERVATIONS - k * SHARE_BATCH, SHARE_BATCH),
                JACOBIAN_BATCH,
            ):
                sl = slice(lo, lo + JACOBIAN_BATCH)
                J = jacobian(flow, b.xt[sl, 0], tt, b.y[sl], b.f[sl])
                jac_maps.append(
                    np.asarray(
                        reduce(
                            jnp.square(J.astype(jnp.float32)),
                            "b s x f c -> b s x f",
                            "sum",
                        )
                    )
                )
        V, VX, VY = (np.concatenate(parts[k]) for k in ("v", "x", "y"))

        def share_of(mask):
            var = V[mask].var(0)
            return dict(
                data=(((V - VY)[mask] ** 2).mean(0) / (2 * var)).tolist(),
                xt=(((V - VX)[mask] ** 2).mean(0) / (2 * var)).tolist(),
                n=int(mask.sum()),
            )

        shares.append(
            dict(
                t=t,
                all=share_of(np.ones_like(snr, bool)),
                slot=[
                    share_of(np.broadcast_to(np.arange(n_slots) == s, snr.shape))
                    for s in range(n_slots)
                ],
                snr=[share_of(snr_bin == i) for i in range(len(SNR_EDGES) - 1)],
            )
        )

        # each slot's squared Jacobian per bin, as shares of its total: own band, other bands, rest; enrichment per ring
        M = np.concatenate(jac_maps)
        n = len(M)
        total = M.sum(-1)
        split = (
            np.stack(
                [(M * own[:n, :, None]).sum(-1), (M * others[:n, :, None]).sum(-1)], -1
            )
            / total[..., None]
        )
        split = np.concatenate([split, 1 - split.sum(-1, keepdims=True)], -1)
        density = np.stack(
            [(M * (ring[:n, :, None] == r)).sum(-1) for r in range(len(rings) - 1)], -1
        )
        enrichment = (
            density
            / total[..., None]
            / np.maximum(ring_counts[:n, :, None], 1)
            * n_bins
        )
        present = ring_counts[:n, :, None] > 0
        jac.append(
            dict(
                t=t,
                split=split.mean(0).tolist(),
                split_snr=[
                    (
                        split[snr_bin[:n] == i].mean(0).tolist()
                        if (snr_bin[:n] == i).any()
                        else None
                    )
                    for i in range(len(SNR_EDGES) - 1)
                ],
                enrichment=(
                    np.where(present, enrichment, 0).sum(0)
                    / np.maximum(present.sum(0), 1)
                )
                .mean(0)
                .tolist(),
                # one observation, summed over coordinates, as each slot's share per `stride` bins
                example=reduce(
                    M[0].sum(1) / M[0].sum((1, 2))[:, None],
                    "s (f k) -> s f",
                    "sum",
                    k=stride,
                )
                .round(5)
                .tolist(),
            )
        )
        print(f"t {t}: {time.perf_counter() - started:.0f} s", flush=True)

    n_jac = len(M)
    baseline = np.stack([own[:n_jac].mean(-1), others[:n_jac].mean(-1)], -1)
    record = dict(
        run=cfg.run,
        config=cfg.name,
        commit=cfg.commit,
        epoch=epoch,
        steps=epoch * cfg.log_interval,
        total_steps=cfg.total_steps,
        weights=cfg.weights,
        bf16=cfg.bf16,
        device=jax.devices()[0].device_kind,
        seconds=round(time.perf_counter() - started),
        coordinates=COORDINATES,
        t_grid=list(T_GRID),
        snr_edges=list(SNR_EDGES),
        window_bins=n_bins,
        patch=flow.patch,
        observations=len(snr),
        jacobian_observations=n_jac,
        band_power=BAND_POWER,
        rings=rings,
        shares=shares,
        jacobian=jac,
        baseline=np.concatenate([baseline, 1 - baseline.sum(-1, keepdims=True)], -1)
        .mean(0)
        .tolist(),
        example=dict(
            snr=snr[0].tolist(),
            f0_bin=f0_bin[0].tolist(),
            own=reduce(own[0], "s (f k) -> s f", "max", k=stride).astype(int).tolist(),
            stride=stride,
        ),
    )
    with open(out, "w") as fh:
        json.dump(record, fh)

    row = lambda label, values: print(
        f"{label:<22}" + " ".join(f"{v:7.3f}" for v in values)
    )
    print(
        f"{cfg.name} {cfg.run} epoch {epoch} ({cfg.weights}), {len(snr)} obs, {n_jac} Jacobians -> {out}"
    )
    print(f"{'':<22}" + " ".join(f"{n:>7s}" for n in COORDINATES))
    for s, j in zip(shares, jac):
        row(f"t {s['t']} data share", s["all"]["data"])
        row(f"t {s['t']} x_t share", s["all"]["xt"])
        row(f"t {s['t']} own band", np.asarray(j["split"])[:, :, 0].mean(0))
    print(
        "own-band bin fraction per slot", np.asarray(record["baseline"])[:, 0].round(3)
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).stem if len(sys.argv) > 1 else None)
