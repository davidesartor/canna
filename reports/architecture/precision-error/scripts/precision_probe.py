"""Per-block and end-to-end error of each bf16 mode against a float32 forward pass on the same trained params, in posterior widths."""

import json
import sys
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float

from canna.lisa import config as run_config
from canna.lisa.chart import physical_to_flow, sample_flow
from canna.lisa.injections import eval_injections, fisher_samples
from canna.lisa.network import LisaFlow
from canna.lisa.sample import sample_posterior
from canna.lisa.simulator import preprocess, sample_observation, source_snrs
from canna.lisa.train import load_trained
from canna.networks import mmdit

REPORT = Path(__file__).parents[1]
# (backbone dtype, attention dtype) per mode, as train.setup builds them; f64 checks the float32 reference's own floor
MODES = dict(
    none=(jnp.float32, None),
    attention=(jnp.float32, jnp.bfloat16),
    backbone=(jnp.bfloat16, jnp.bfloat16),
    f64=(jnp.float64, None),
)
REFERENCE = "none"
T_GRID = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)
N_STATES = 16
N_DRAWS = 64
MAD_SIGMA = 1.4826
RECORD: dict = {}


def recording_encode(self, y, c, rope=None):
    """MMDiT.encode, also recording the y stream after every block and the cached keys and values."""
    params, static = eqx.partition(self.blocks, eqx.is_array)

    def body(y, block):
        y, kv = eqx.combine(block, static).encode(y, c, rope)
        return y, (y, kv)

    y_out, (ys, (k, v)) = jax.lax.scan(body, y, params)
    _, (k_last, v_last) = self.last.encode(y_out, c, rope)
    k, v = jnp.concat([k, k_last[None]]), jnp.concat([v, v_last[None]])
    RECORD.update(y=jnp.concat([y[None], ys]), k=k, v=v)
    return k, v


def recording_decode(self, x, c, cache, rope=None, s=None):
    """MMDiT.decode, also recording the x stream entering each block and leaving the last."""
    params, static = eqx.partition(self.blocks, eqx.is_array)

    def body(carry, inputs):
        block, kv = inputs
        carry = eqx.combine(block, static).decode(carry[0], c, kv, rope, carry[1])
        return carry, carry[0]

    cached, cached_last = (
        jax.tree.map(lambda a: a[:-1], cache),
        jax.tree.map(lambda a: a[-1], cache),
    )
    (x_mid, s), xs = jax.lax.scan(body, (x, s), (params, cached))
    out = self.last.decode(x_mid, c, cached_last, rope, s)[0]
    RECORD.update(x=jnp.concat([x[None], xs, out[None]]))
    return out


@eqx.filter_jit
def encode_recorded(flow: LisaFlow, y: Float[Array, "F 6"], f: Float[Array, ""]):
    """The data stream's cache, with every block's y, k and v."""
    RECORD.clear()
    cache = flow.encode(y, f)
    return cache, {key: RECORD[key].astype(jnp.float64) for key in ("y", "k", "v")}


@eqx.filter_jit
def decode_recorded(flow: LisaFlow, x, t, y, f, cache):
    """The flow's mean m of x1 at states [N, S, 8], with the x stream at every block."""

    def one(x):
        RECORD.clear()
        m, _ = jnp.split(flow(x, t, y, f, cache), 2, axis=-1)
        return m.astype(jnp.float64), RECORD["x"].astype(jnp.float64)

    return jax.vmap(one)(x)


def as_mode(problem, cfg, flow: LisaFlow, mode: str) -> LisaFlow:
    """The same trained params in another precision mode's (static) structure."""
    dtype, attention_dtype = MODES[mode]
    skeleton = LisaFlow(
        problem,
        **cfg.network,
        dtype=dtype,
        attention_dtype=attention_dtype,
        param_dtype=jnp.float32,
        key=jr.key(0),
    )
    return jax.tree.unflatten(jax.tree.structure(skeleton), jax.tree.leaves(flow))


def token_relative_error(h: np.ndarray, ref: np.ndarray, axes: str) -> np.ndarray:
    """‖h − ref‖ / ‖ref‖ per token, flattening the feature axes named in `axes`."""
    diff = rearrange(h - ref, axes)
    return np.linalg.norm(diff, axis=-1) / np.linalg.norm(rearrange(ref, axes), axis=-1)


def robust_width(draws: np.ndarray) -> np.ndarray:
    """1.4826 · median absolute deviation over the leading axis, ignoring NaN."""
    centre = np.nanmedian(draws, axis=0)
    return MAD_SIGMA * np.nanmedian(np.abs(draws - centre), axis=0)


def quantiles(a: np.ndarray, axis) -> list:
    """Median and 90th percentile over `axis`."""
    return np.stack(
        [np.median(a, axis=axis), np.quantile(a, 0.9, axis=axis)], -1
    ).tolist()


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT / "data" / "run.json"
    cfg = run_config.load_run()
    problem, trained, epoch = load_trained(cfg)
    flows = {mode: as_mode(problem, cfg, trained, mode) for mode in MODES}
    latents, windows, _, _, noise_keys, prior_prec_p = eval_injections(
        problem, cfg.seed
    )

    # the recording methods must reproduce the scanned ones exactly
    original = mmdit.MMDiT.encode, mmdit.MMDiT.decode
    y0 = preprocess(
        problem,
        sample_observation(problem, noise_keys[0], latents[0], windows[0]),
        windows[0],
    )
    u_check = sample_flow(problem, noise_keys[0], windows[0])
    forward = eqx.filter_jit(
        lambda flow: jnp.split(flow(u_check, jnp.asarray(0.5), y0, windows[0]), 2, axis=-1)[0]
    )
    plain = forward(flows[REFERENCE])

    # attention's float32 logits may reject a float64 backbone: then there is no floor to show
    try:
        forward(flows["f64"])
    except Exception as error:
        print(f"[precision] f64 mode dropped: {type(error).__name__}: {error}", flush=True)
        del flows["f64"]
    mmdit.MMDiT.encode, mmdit.MMDiT.decode = recording_encode, recording_decode
    cache0, _ = encode_recorded(flows[REFERENCE], y0, windows[0])
    recorded, _ = decode_recorded(
        flows[REFERENCE],
        u_check[None],
        jnp.asarray(0.5),
        y0,
        windows[0],
        cache0,
    )
    patch_check = float(jnp.max(jnp.abs(recorded[0] - plain)))
    print(
        f"[precision] recording vs scanned forward: max |Δ| {patch_check:.2g}",
        flush=True,
    )

    encode_err = {m: {k: [] for k in ("y", "k", "v")} for m in flows}
    decode_err = {m: [] for m in flows}
    x1_err = {m: [] for m in flows}
    velocity_err = {m: [] for m in flows}
    e2e_disp = {m: [] for m in flows}
    e2e_width = {m: [] for m in flows}
    widths, snrs, token_ratio, token_rounding = [], [], [], []

    for j, (latent, f, key_n) in enumerate(zip(latents, windows, noise_keys)):
        y = preprocess(problem, sample_observation(problem, key_n, latent, f), f)
        x1 = physical_to_flow(problem, latent, f)
        fisher = fisher_samples(problem, latent, f, key_n, prior_prec_p)
        width = robust_width(
            np.asarray(
                jax.vmap(partial(physical_to_flow, problem), in_axes=(0, None))(
                    jnp.asarray(fisher), f
                )
            )
        )
        widths.append(width)
        snrs.append(np.asarray(source_snrs(problem, latent, f)))
        u0 = jax.vmap(partial(sample_flow, problem), in_axes=(0, None))(
            jr.split(key_n, N_DRAWS), f
        )

        # the data stream, per mode
        caches, enc = {}, {}
        for mode, flow in flows.items():
            caches[mode], enc[mode] = encode_recorded(flow, y, f)
        for mode in flows:
            for key, axes in (
                ("y", "l m d -> l m d"),
                ("k", "l m h k -> l m (h k)"),
                ("v", "l m h k -> l m (h k)"),
            ):
                encode_err[mode][key].append(
                    token_relative_error(
                        np.asarray(enc[mode][key]),
                        np.asarray(enc[REFERENCE][key]),
                        axes,
                    )
                )

        # x1 and the x stream at straight-path states between base draws and the truth
        dec_t = {m: [] for m in flows}
        x1_t = {m: [] for m in flows}
        vel_t = {m: [] for m in flows}
        for t in T_GRID:
            xt = (1 - t) * u0[:N_STATES] + t * x1
            t_arr = jnp.asarray(t)
            outs = {
                m: decode_recorded(flows[m], xt, t_arr, y, f, caches[m])
                for m in flows
            }
            ref_out, ref_x = (np.asarray(a) for a in outs[REFERENCE])
            for mode, (out, xs) in outs.items():
                out, xs = np.asarray(out), np.asarray(xs)
                dec_t[mode].append(
                    token_relative_error(xs, ref_x, "n l s d -> l (n s) d")
                )
                x1_t[mode].append(np.median(np.abs(out - ref_out), axis=0) / width)
                vel_t[mode].append(
                    np.median(
                        np.linalg.norm(out - ref_out, axis=-1)
                        / np.linalg.norm(ref_out - np.asarray(xt), axis=-1)
                    )
                )
        for mode in flows:
            decode_err[mode].append(np.stack(dec_t[mode]))
            x1_err[mode].append(np.stack(x1_t[mode]))
            velocity_err[mode].append(np.stack(vel_t[mode]))

        # end to end: the default sampler from the same base draws in every mode
        draws = {
            m: np.asarray(
                sample_posterior(flows[m], u0, y, f, cfg.ode_steps)
            )
            for m in flows
        }
        for mode in flows:
            e2e_disp[mode].append(
                np.median(np.abs(draws[mode] - draws[REFERENCE]), axis=0) / width
            )
            e2e_width[mode].append(
                robust_width(draws[mode]) / robust_width(draws[REFERENCE])
            )

        # can a bf16 token resolve a 1σ move? the float32 decode input at the truth, each coordinate moved by its width
        tokens = lambda x: decode_recorded(
            flows[REFERENCE],
            x[None],
            jnp.asarray(0.99),
            y,
            f,
            caches[REFERENCE],
        )[1][0, 0]
        base = np.asarray(tokens(x1))
        rounding = np.linalg.norm(
            base
            - np.asarray(
                jnp.asarray(base, jnp.float32).astype(jnp.bfloat16).astype(jnp.float64)
            ),
            axis=-1,
        )
        moved = np.stack(
            [
                np.linalg.norm(
                    np.asarray(
                        tokens(x1.at[:, c].add(jnp.asarray(width[:, c])))
                    )
                    - base,
                    axis=-1,
                )
                for c in range(x1.shape[-1])
            ],
            -1,
        )
        token_ratio.append(moved / rounding[:, None])
        token_rounding.append(rounding / np.linalg.norm(base, axis=-1))
        print(
            f"[{j + 1}/{len(latents)}] snr {np.round(snrs[-1], 1).tolist()} backbone Δx1/width (t=0.99) median {np.nanmedian(x1_err['backbone'][-1][3]):.3g}",
            flush=True,
        )
    mmdit.MMDiT.encode, mmdit.MMDiT.decode = original

    stack = lambda d: {m: np.stack(v) for m, v in d.items()}
    record = dict(
        run=cfg.run,
        config=cfg.name,
        commit=cfg.commit,
        epoch=int(epoch),
        total_steps=cfg.total_steps,
        weights=cfg.weights,
        run_bf16=cfg.bf16,
        device=jax.devices()[0].device_kind,
        platform=jax.default_backend(),
        matmul_precision=str(jax.config.jax_default_matmul_precision),
        modes=list(flows),
        reference=REFERENCE,
        t_grid=list(T_GRID),
        n_states=N_STATES,
        n_draws=N_DRAWS,
        ode_steps=cfg.ode_steps,
        patch_check=patch_check,
        snr=np.stack(snrs).tolist(),
        width=nan_list(np.stack(widths)),
        encode=quantile_tree(
            {m: {k: np.stack(v) for k, v in d.items()} for m, d in encode_err.items()}
        ),
        decode={
            m: quantiles(rearrange(v, "o t l n -> t l (o n)"), -1)
            for m, v in stack(decode_err).items()
        },
        x1_over_width={m: nan_list(v) for m, v in stack(x1_err).items()},
        velocity={m: np.stack(v).tolist() for m, v in velocity_err.items()},
        e2e_displacement={m: nan_list(v) for m, v in stack(e2e_disp).items()},
        e2e_width_ratio={m: nan_list(v) for m, v in stack(e2e_width).items()},
        token_ratio=nan_list(np.stack(token_ratio)),
        token_rounding=np.stack(token_rounding).tolist(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, allow_nan=False))
    print(f"[precision] wrote {out_path}")


def quantile_tree(errors: dict) -> dict:
    """Per mode and tensor, the median and 90th percentile per block over observations and tokens."""
    return {
        m: {k: quantiles(rearrange(v, "o l m -> l (o m)"), -1) for k, v in d.items()}
        for m, d in errors.items()
    }


def nan_list(a: np.ndarray) -> list:
    """Rounded to 4 significant digits, NaN and inf as None, for JSON."""
    a = np.asarray(a, np.float64)
    rounded = np.where(
        np.isfinite(a),
        np.vectorize(lambda v: float(f"{v:.4g}"))(np.nan_to_num(a)),
        np.nan,
    )
    return (
        [nan_list(row) for row in rounded]
        if rounded.ndim > 1
        else [None if np.isnan(v) else float(v) for v in rounded]
    )


if __name__ == "__main__":
    main()
