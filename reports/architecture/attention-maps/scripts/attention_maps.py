"""Where a trained LisaFlow's source tokens attend: per block and head, against path time t and SNR.

The MMDiT does not expose its attention weights, so its decode is swapped (in memory only) for a
python loop over the same blocks that recomputes each block's softmax from the block's own
modulation, q/k projections, rms norm, qk gain and RoPE, then runs the block's real decode. The
recomputed weights are checked against it: the x stream they imply must match the block's.
On fresh training observations at fixed t (one network call), each
source slot's mass on itself, the other slots, the sinks and the data tokens; the data mass on the
tokens holding its own source's band (95% of its power), the other sources' bands and the rest;
entropies; and one observation's full map. build_page.py, beside this script, makes the page.

Run from the repo root on the run's commit (slurm/run-src.sh), as attention_maps.sbatch does:
    LISA_RUN=<run> uv run --extra cuda --frozen python reports/architecture/attention-maps/scripts/attention_maps.py [tag]
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
from einops import rearrange, reduce
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
from canna.networks import MMDiT
from canna.networks.mmdit import MMDiTBlock, rotate

REPORT = Path(__file__).parents[1]
T_GRID = (0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
SNR_EDGES = (7.0, 20.0, 60.0, 200.0, 1000.0)
BAND_POWER = 0.95
BATCHES, BATCH = 4, 16
# the example map's weights as integers in these units
EXAMPLE_UNIT = 1e-4

# each block's weights, appended while a tapped flow is traced
TAP: list = []


def block_attention(
    block: MMDiTBlock,
    x: Float[Array, "N D"],
    c: Float[Array, " D"],
    kv: tuple[Float[Array, "M H K"], Float[Array, "M H K"]],
    rope,
    s: Float[Array, "R D"] | None,
) -> tuple[Float[Array, "H Q U"], Float[Array, "N D"]]:
    """MMDiTBlock.decode's softmax over [x, sinks, data] keys for the x and sink queries, and the x it implies."""
    attention = block.attention
    n = x.shape[-2]
    hx, gate = block.mod_x_attn(x, c)
    q, k, v = attention.heads(attention.qkv_proj_x, hx, rope)
    if s is not None:
        hs, _ = block.mod_s_attn(s, c)
        qkv_s = attention.heads(attention.qkv_proj_s, hs)
        q, k, v = (jnp.concat([a, b]) for a, b in zip((q, k, v), qkv_s))
    out_dtype = v.dtype
    q, k, v = (a.astype(attention.attention_dtype or out_dtype) for a in (q, k, v))
    logits = jnp.concat(
        [
            jnp.einsum("nhk,mhk->hnm", q, k, preferred_element_type=jnp.float32),
            jnp.einsum("nhk,mhk->hnm", q, kv[0], preferred_element_type=jnp.float32),
        ],
        axis=-1,
    )
    weights = jax.nn.softmax(logits * q.shape[-1] ** -0.5, axis=-1)

    # the x stream's update from these weights, as decode makes it
    w = weights[:, :n].astype(v.dtype)
    own = k.shape[0]
    h = jnp.einsum("hnm,mhk->nhk", w[..., :n], v[:n]) + jnp.einsum(
        "hnm,mhk->nhk", w[..., own:], kv[1]
    )
    if rope is not None:
        h = rotate(h, rope[0], -rope[1])
    h = h + jnp.einsum("hnm,mhk->nhk", w[..., n:own], v[n:own])
    x = (
        x
        + jax.vmap(attention.out_proj_x)(
            rearrange(h.astype(out_dtype), "n h d -> n (h d)")
        )
        * gate
    )
    hx, gate = block.mod_x_mlp(x, c)
    return weights, x + jax.vmap(block.mlp_x)(hx) * gate


class TappedMMDiT(eqx.Module):
    """MMDiT's encode and decode, decode as a python loop over the blocks that records each block's attention in TAP."""

    blocks: MMDiTBlock
    last: MMDiTBlock

    def encode(self, y, c, rope=None):
        return MMDiT.encode(self, y, c, rope)

    def decode(self, x, c, cache, rope=None, s=None):
        params, static = eqx.partition(self.blocks, eqx.is_array)
        n_layers = cache[0].shape[0]
        for i in range(n_layers):
            block = (
                self.last
                if i == n_layers - 1
                else eqx.combine(jax.tree.map(lambda a: a[i], params), static)
            )
            kv = (cache[0][i], cache[1][i])
            weights, x_implied = block_attention(block, x, c, kv, rope, s)
            x, s = block.decode(x, c, kv, rope, s)
            x32 = x.astype(jnp.float32)
            gap = jnp.max(jnp.abs(x_implied.astype(jnp.float32) - x32))
            TAP.append((weights, gap / jnp.max(jnp.abs(x32))))
        return x


def tapped(
    flow: LisaFlow,
    plain: LisaFlow,
    xt: Float[Array, "S X"],
    t: Float[Array, ""],
    y: Float[Array, "F 6"],
    f: Float[Array, ""],
) -> tuple[Float[Array, "L H S U"], Float[Array, " L"], Float[Array, ""]]:
    """One network call at (xt, t): every block's source-query weights, the relative x-check per block, the relative output check."""
    cache = flow.encode(y, f)
    TAP.clear()
    out = flow(xt, t, y, f, cache)
    weights = jnp.stack([w for w, _ in TAP])[:, :, : xt.shape[0]]
    checks = jnp.stack([c for _, c in TAP])
    TAP.clear()
    reference = plain(xt, t, y, f, plain.encode(y, f))
    return weights, checks, jnp.max(jnp.abs(out - reference)) / jnp.max(jnp.abs(reference))


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


def main(tag: str | None) -> None:
    cfg = config.load_run()
    out = REPORT / "data" / f"{tag or cfg.run}.json"
    problem, plain, epoch = load_trained(cfg)
    backbone = plain.backbone
    flow = eqx.tree_at(
        lambda m: m.backbone, plain, TappedMMDiT(backbone.blocks, backbone.last)
    )
    n_slots, patch = problem.n_sources, plain.patch
    n_sinks = 0 if plain.s_slot is None else plain.s_slot.shape[0]
    n_tokens = problem.window_bins // patch
    started = time.perf_counter()

    draws = jax.jit(
        jax.vmap(
            lambda key, t: train_sample(problem, key, jnp.full((1,), t)),
            in_axes=(0, None),
        )
    )
    bands = jax.jit(jax.vmap(partial(source_bands, problem)))

    @eqx.filter_jit
    def probe(flow, plain, xt, t, y, f):
        return jax.vmap(
            lambda xt, y, f: tapped(flow, plain, xt, t, y, f)
        )(xt, y, f)

    keys = [jr.split(jr.key(400 + k), BATCH) for k in range(BATCHES)]
    snr, power, f0_bin = (
        np.concatenate(a) for a in zip(*(map(np.asarray, bands(k)) for k in keys))
    )
    own_bins = band_masks(power)
    own = reduce(own_bins, "n s (m p) -> n s m", "max", p=patch)
    others = np.stack(
        [
            own[:, [j for j in range(n_slots) if j != s]].any(1) & ~own[:, s]
            for s in range(n_slots)
        ],
        1,
    )
    snr_bin = np.clip(np.digitize(snr, SNR_EDGES) - 1, 0, len(SNR_EDGES) - 2)
    rings = [0, 1]
    while rings[-1] < n_tokens:
        rings.append(2 * rings[-1])
    offset = np.abs(np.arange(n_tokens) - np.floor(f0_bin / patch)[..., None])
    ring = np.clip(np.searchsorted(rings, offset, side="right") - 1, 0, len(rings) - 2)
    ring_counts = np.stack([(ring == r).sum(-1) for r in range(len(rings) - 1)], -1)

    per_t, checks, outputs = [], [], []
    for t in T_GRID:
        parts = []
        for batch_keys in keys:
            b = draws(batch_keys, t)
            W, check, output = probe(flow, plain, b.xt[:, 0], jnp.asarray(t), b.y, b.f)
            parts.append(np.asarray(W, np.float64))
            checks.append(np.asarray(check))
            outputs.append(np.asarray(output))
        # W: [obs, block, head, slot, key] over keys [slots, sinks, data tokens]
        W = np.concatenate(parts)
        eye = np.eye(n_slots, dtype=bool)
        sources = W[..., :n_slots]
        self_mass = (sources * eye).sum(-1)
        data = W[..., n_slots + n_sinks :]
        data_mass = data.sum(-1)
        mass = np.stack(
            [
                self_mass,
                sources.sum(-1) - self_mass,
                W[..., n_slots : n_slots + n_sinks].sum(-1),
                data_mass,
            ],
            -1,
        )
        spread = lambda m: m[:, None, None]
        share = data / np.maximum(data_mass, 1e-30)[..., None]
        split = np.stack(
            [(share * spread(own)).sum(-1), (share * spread(others)).sum(-1)], -1
        )
        split = np.concatenate([split, 1 - split.sum(-1, keepdims=True)], -1)
        entropy = lambda p, n: -(p * np.log(np.maximum(p, 1e-30))).sum(-1) / np.log(n)
        ring_share = np.stack(
            [(share * spread(ring == r)).sum(-1) for r in range(len(rings) - 1)], -1
        )
        enrichment = ring_share / np.maximum(spread(ring_counts), 1) * n_tokens
        present = np.broadcast_to(spread(ring_counts) > 0, enrichment.shape)
        per_t.append(
            dict(
                t=t,
                mass=mass.mean(0).tolist(),
                split=split.mean(0).tolist(),
                split_snr=[
                    (
                        rearrange(split[..., 0], "n l h s -> n s l h")[snr_bin == i]
                        .mean(0)
                        .tolist()
                        if (snr_bin == i).any()
                        else None
                    )
                    for i in range(len(SNR_EDGES) - 1)
                ],
                entropy=entropy(W, W.shape[-1]).mean(0).tolist(),
                data_entropy=entropy(share, n_tokens).mean(0).tolist(),
                enrichment=(
                    np.where(present, enrichment, 0).sum((0, 3))
                    / np.maximum(present.sum((0, 3)), 1)
                ).tolist(),
                # one observation's data weights, mean over heads: [block, slot, token]
                example=np.rint(data[0].mean(1) / EXAMPLE_UNIT).astype(int).tolist(),
            )
        )
        print(f"t {t}: {time.perf_counter() - started:.0f} s", flush=True)

    baseline = np.stack([own.mean(-1), others.mean(-1)], -1)
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
        observations=len(snr),
        blocks=int(W.shape[1]),
        heads=int(W.shape[2]),
        slots=n_slots,
        sinks=n_sinks,
        tokens=n_tokens,
        patch=patch,
        t_grid=list(T_GRID),
        snr_edges=list(SNR_EDGES),
        band_power=BAND_POWER,
        rings=rings,
        example_unit=EXAMPLE_UNIT,
        # max |x| gap per block between the recomputed weights' x and decode's, and the tapped vs plain output, each over max |x|
        check_block=np.max(np.concatenate(checks), 0).tolist(),
        check_output=float(np.max(np.concatenate(outputs))),
        baseline=np.concatenate([baseline, 1 - baseline.sum(-1, keepdims=True)], -1)
        .mean(0)
        .tolist(),
        per_t=per_t,
        example=dict(
            snr=snr[0].tolist(),
            f0_token=np.floor(f0_bin[0] / patch).astype(int).tolist(),
            own=own[0].astype(int).tolist(),
        ),
    )
    with open(out, "w") as fh:
        json.dump(record, fh)

    print(
        f"{cfg.name} {cfg.run} epoch {epoch} ({cfg.weights}), {len(snr)} obs -> {out}"
    )
    print(
        f"check (relative to max |x|): block x gap {np.round(record['check_block'], 5)}, output gap {record['check_output']:.2e}"
    )
    print("own-band token fraction per slot", baseline[..., 0].mean(0).round(3))
    for r in per_t:
        m = np.asarray(r["mass"]).mean((1, 2))
        s = np.asarray(r["split"])[..., 0].mean((1, 2))
        e = np.asarray(r["entropy"]).mean((1, 2))
        print(
            f"t {r['t']}: per block [self, others, sinks, data] "
            + " | ".join(" ".join(f"{v:.2f}" for v in row) for row in m)
            + "  own-band share "
            + " ".join(f"{v:.2f}" for v in s)
            + "  entropy "
            + " ".join(f"{v:.2f}" for v in e)
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]).stem if len(sys.argv) > 1 else None)
