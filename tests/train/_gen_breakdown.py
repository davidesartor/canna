"""Per-operation timing of one NoisySinusoid batch draw. Run directly, not via pytest."""

import math
import os
import time
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
from wdm_transform.transforms import from_freq_to_wdm_band

import canna  # noqa: F401
from canna import problems

from _bench import run_config_args

RUN_CONFIG = os.environ.get("RUN_CONFIG", "NoisySinusoid-MMDiT-B")
N_TIMED = 20


def time_ms(fn, args: tuple) -> tuple[float, float]:
    """Time fn(*args) jitted. Inputs go in as arguments so XLA cannot constant-fold."""
    jitted = jax.jit(fn)
    t0 = time.monotonic()
    jax.block_until_ready(jitted(*args))
    compile_ms = (time.monotonic() - t0) * 1e3
    t0 = time.monotonic()
    for _ in range(N_TIMED):
        out = jitted(*args)
    jax.block_until_ready(out)
    return compile_ms, (time.monotonic() - t0) / N_TIMED * 1e3


def main():
    args = run_config_args(RUN_CONFIG)
    problem = getattr(problems, args.problem["class"])(**args.problem["init_args"])
    batch = args.batch_size
    keys = jr.split(jr.key(0), batch)

    p = jax.vmap(problem.sample_physical)(keys)
    o = jax.vmap(problem.sample_observation)(keys, p)
    x0 = jax.vmap(problem.sample_point)(keys)
    x1 = jax.vmap(problem.chart.forward)(p)
    t = jr.uniform(jr.key(1), (batch,))

    # the grid setup below is copied from NoisySinusoid.preprocess, to time its pieces
    T = o.shape[-2]
    spectrum = jnp.fft.rfft(o, axis=-2)
    df = 1.0 / (T * problem.sampling_step)
    fmin, fmax = problem.freq_range
    kmin, kmax = math.floor(fmin / df), math.floor(fmax / df)
    band_bins = kmax - kmin + 1
    pd = problem.patch_downsample
    nt = pd * math.ceil(2 * band_bins / (problem.wdm_freq_bands * pd))
    half = nt // 2
    nf = (T // nt) - (T // nt) % 2
    nfreqs_fourier = nt * nf // 2 + 1
    block = problem.wdm_freq_bands * half
    mmin = max(0, (kmin - (block - band_bins) // 2) // half)
    kmin = mmin * half
    band = spectrum[..., kmin : kmin + block, :]

    @partial(jnp.vectorize, signature="(w,c)->(t,f,c)")
    @partial(jax.vmap, in_axes=-1, out_axes=-1)
    def to_wdm(channel):
        return from_freq_to_wdm_band(
            channel,
            df=df,
            nfreqs_fourier=nfreqs_fourier,
            kmin=kmin,
            nfreqs_wdm=nf,
            ntimes_wdm=nt,
            mmin=mmin,
            nf_sub_wdm=problem.wdm_freq_bands,
            a=1.0 / 3.0,
            d=1.0,
            backend="jax",
        )

    wdm = to_wdm(band)
    amp, freq, phase = jnp.split(p, 3, axis=-1)
    t_grid = jnp.arange(0, problem.t_obs, problem.sampling_step)
    angle = 2.0 * jnp.pi * freq * t_grid + phase
    noise_scale = jnp.sqrt(problem.noise_level / (2.0 * problem.sampling_step))

    # aggregates, kept out of the offender ranking so pieces are not double counted
    aggregates = {
        "train_sample (all)": (jax.vmap(problem.train_sample), (keys,)),
        "preprocess": (jax.vmap(problem.preprocess), (o,)),
        "sample_observation": (jax.vmap(problem.sample_observation), (keys, p)),
        "clean_signal": (jax.vmap(problem.clean_signal), (p,)),
    }

    ops = {
        "sample_physical": (jax.vmap(problem.sample_physical), (keys,)),
        "angle build": (
            lambda freq, phase: 2.0 * jnp.pi * freq * t_grid + phase,
            (freq, phase),
        ),
        "sin/cos + sum": (
            lambda amp, angle: jnp.stack(
                [
                    jnp.sum(amp * jnp.sin(angle), axis=-2),
                    jnp.sum(amp * jnp.cos(angle), axis=-2),
                ],
                axis=-1,
            ),
            (amp, angle),
        ),
        "noise draw": (
            lambda key: jr.normal(key, o.shape) * noise_scale,
            (jr.key(2),),
        ),
        "rfft": (lambda o: jnp.fft.rfft(o, axis=-2), (o,)),
        "band slice": (
            lambda spectrum: spectrum[..., kmin : kmin + block, :],
            (spectrum,),
        ),
        "to_wdm": (to_wdm, (band,)),
        "arcsinh": (jnp.arcsinh, (wdm,)),
        "chart.forward": (jax.vmap(problem.chart.forward), (p,)),
        "geodesic": (jax.vmap(problem.geometry.geodesic), (t, x0, x1)),
        "geodesic jacobian": (
            jax.vmap(jax.jacobian(problem.geometry.geodesic)),
            (t, x0, x1),
        ),
    }

    print(f"{RUN_CONFIG} batch={batch} on {jax.default_backend()}")
    print(f"{'op':>20} {'compile ms':>11} {'step ms':>9}")
    results = {}
    for name, (fn, fn_args) in {**aggregates, **ops}.items():
        compile_ms, step_ms = time_ms(fn, fn_args)
        results[name] = step_ms
        print(f"{name:>20} {compile_ms:>11.0f} {step_ms:>9.2f}", flush=True)

    total = results["train_sample (all)"]
    print(f"\nworst individual ops (share of one train_sample)")
    ranked = sorted(ops, key=lambda name: results[name], reverse=True)
    for name in ranked[:5]:
        # preprocess runs on both the noisy and the clean observation
        reps = 2 if name in ("rfft", "band slice", "to_wdm", "arcsinh") else 1
        print(
            f"{name:>20} {results[name]:>9.2f} ms x{reps} "
            f"{100 * reps * results[name] / total:>6.0f}%"
        )
    print(
        f"\n{'preprocess x2 (noisy + clean)':>36} {2 * results['preprocess']:>9.2f} ms"
    )
    print(
        f"{'as share of train_sample':>36} {200 * results['preprocess'] / total:>8.0f}%"
    )


if __name__ == "__main__":
    main()
