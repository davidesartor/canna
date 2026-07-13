"""Prior distributions: normalized u (should be ~Uniform[0,1]) and physical params."""

import os

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from canna import lisa

from _helpers import problem_name, save_figure, save_text

LOG_PARAMS = {"f0", "fdot", "A"}  # log-uniform priors: show on a log axis
N_SOURCES = int(os.environ.get("PRIOR_SOURCES", 4096))  # priors are per-source iid


def test_param_priors():
    # priors need no waveforms: draw u directly and map through the inverse CDF
    u_full = jr.uniform(jr.key(int(os.environ.get("SEED", 0))), (N_SOURCES, 8))
    params_full = np.asarray(lisa.prior_inverse_cdf(u_full))
    u = np.asarray(u_full)
    if lisa.SIMPLIFIED_PROBLEM:
        u, params_full = u[:, lisa.MASK], params_full[:, lisa.MASK]
    params = params_full
    names = lisa.PARAMETER_NAMES

    assert np.isfinite(u).all() and np.isfinite(params).all()
    assert (u >= 0.0).all() and (u <= 1.0).all()

    p = len(names)
    fig, axes = plt.subplots(2, p, figsize=(3 * p, 6), squeeze=False)
    for j, name in enumerate(names):
        axes[0, j].hist(u[:, j], bins=40, color="steelblue")
        axes[0, j].set_title(f"u[{name}]")
        col = params[:, j]
        if name in LOG_PARAMS:
            bins = np.logspace(np.log10(col.min()), np.log10(col.max()), 40)
            axes[1, j].set_xscale("log")
        else:
            bins = 40
        axes[1, j].hist(col, bins=bins, color="indianred")
        axes[1, j].set_title(name)
    fig.suptitle(f"param priors — {problem_name()} ({N_SOURCES} sources)")
    save_figure(fig, "param_priors")

    lines = [f"param priors — {problem_name()} ({N_SOURCES} sources)"]
    for j, name in enumerate(names):
        c = params[:, j]
        lines.append(
            f"{name:>6}: u[mean={u[:, j].mean():.3f} std={u[:, j].std():.3f}]  "
            f"phys[min={c.min():.3g} med={np.median(c):.3g} max={c.max():.3g}]"
        )
    save_text("param_priors", "\n".join(lines))
