"""Shared plotting / IO / SNR helpers for the lisa data-inspection tests."""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
from einops import rearrange

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Scalar

from canna import lisa

PROBE_DIR = os.path.join(os.environ.get("OUTPUT_DIR", "outputs"), "inspect")

# source counts overlaid in the multi-source probes (data-gen only, so the
# match_sources n<=4 limit does not apply here)
SOURCE_COUNTS = [1, 2, 4, 8]


def map_chunks(fn, xs, chunk: int):
    """vmap ``fn`` over the leading axis of ``xs`` in chunks; concat as numpy pytree."""
    n = xs.shape[0]
    vfn = jax.jit(jax.vmap(fn))
    outs = [vfn(xs[i : i + chunk]) for i in range(0, n, chunk)]
    return jax.tree.map(lambda *a: np.concatenate([np.asarray(x) for x in a]), *outs)


def problem_name() -> str:
    return "simplified (1 month)" if lisa.SIMPLIFIED_PROBLEM else "full (1 year)"


def artifact_path(name: str) -> str:
    """Absolute path under outputs/inspect/, tagged by the active problem."""
    os.makedirs(PROBE_DIR, exist_ok=True)
    tag = "simplified" if lisa.SIMPLIFIED_PROBLEM else "full"
    return os.path.join(PROBE_DIR, f"{name}_{tag}")


def save_figure(fig, name: str) -> str:
    path = artifact_path(name) + ".pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {path}")
    return path


def save_text(name: str, text: str) -> str:
    path = artifact_path(name) + ".txt"
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"[txt] -> {path}")
    print(text)
    return path


def snr_of_signal(signal: Float[Array, "T C"]) -> Scalar:
    """Combined matched-filter optimal SNR of a channel-last clean A/E/T signal against the TDI 1.5 PSD."""
    signal = rearrange(signal, "t c -> c t")
    n = signal.shape[-1]
    spectra = jnp.fft.rfft(signal, axis=-1)
    psd = lisa.noise_psd(jnp.fft.rfftfreq(n, lisa.SAMPLING_STEP))
    integrand = jnp.where(psd > 0.0, jnp.abs(spectra) ** 2 / jnp.where(psd > 0.0, psd, 1.0), 0.0)
    return jnp.sqrt(2.0 * lisa.SAMPLING_STEP / n * jnp.sum(integrand))
