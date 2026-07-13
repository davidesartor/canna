"""WDM image dims (T, F, C) and patchified y-token counts across a range of nf.

Runs the real preprocess_datastream + Patchify on a mock (zeros) datastream and
reads back the shapes -- no formula assumptions. Observation length is set by
SIMPLIFIED_PROBLEM; run once per problem to cover both.
"""

import jax.numpy as jnp
import jax.random as jr

from canna import lisa, networks

from _helpers import problem_name, save_text

NF_CHOICES = [256, 512, 1024, 2048]


def test_wdm_dims():
    channels = len(lisa.CHANNEL_NAMES)
    patchify = networks.Patchify(channels=channels, dim=64, key=jr.key(0))
    mock = jnp.zeros((lisa.N_SAMPLES, channels))

    header = f"{'nf':>6} {'nt':>8} {'(T, F, C)':>18} {'patch (n_t,n_f)':>16} {'y_tokens':>10}"
    lines = [
        f"{problem_name()}: t_obs={lisa.T_OBS:,.0f}s  dt={lisa.SAMPLING_STEP:.4f}s  "
        f"n_samples={lisa.N_SAMPLES:,}",
        header,
    ]
    for nf in NF_CHOICES:
        y = lisa.preprocess_datastream(mock, nf=nf)
        t, f, c = y.shape
        gt, gf, d = patchify(y).shape
        assert c == channels and d == 64
        assert gt >= 1 and gf >= 1
        lines.append(
            f"{nf:>6} {t:>8} {f'({t}, {f}, {c})':>18} {f'({gt}, {gf})':>16} {gt * gf:>10,}"
        )
    save_text("wdm_dims", "\n".join(lines))
