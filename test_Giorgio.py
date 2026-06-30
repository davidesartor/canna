"""Test the trained flow-matching network on a known Galactic-Binary injection.

Clean rewrite of ``test.py``. Differences:

* The network is conditioned on the **WDM image of the datastream** (real
  inference), not on a noisy copy of the parameters.
* Posterior samples are drawn with :func:`src.gb_problem.sample_flow`, which
  integrates the velocity field without the ``x % 1`` periodic-wrap hack of
  ``MMDiT.push`` (that hack corrupts the non-periodic f0/fdot/A/psi axes).
* The corner plot is in interpretable units ``[log10 f0, log10 fdot, log10 A, psi]``
  with the injected truth overlaid.

Usage:
    python test_Giorgio.py --n-samples 2000
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

jax.config.update("jax_enable_x64", True)

from src import lisa, gb_problem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoint_Giorgio.eqx")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--t-obs", type=float, default=lisa.MONTH_s)
    p.add_argument("--out", type=str, default="test_Giorgio_corner.pdf")
    args = p.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    key = jr.key(args.seed)

    # --- known injection (within the training ranges) ---------------------
    phys_true = jnp.array([2.0e-3, 1.0e-16, 1.0e-22, float(jnp.pi / 4)])  # f0, fdot, A, psi
    u_true = gb_problem.physical_to_u(phys_true)
    print("injection (physical):", np.asarray(phys_true))
    print("injection (unit cube):", np.asarray(u_true))

    # --- datastream + WDM conditioning (same pipeline as training) --------
    key, key_noise = jr.split(key)
    params = gb_problem.full_params(phys_true)
    signal = lisa.clean_signal(params, t_obs=args.t_obs, dt=gb_problem.DT,
                               n=gb_problem.N_SLOW, ncrop=gb_problem.NCROP)
    noise = lisa.sample_noise(key_noise, t_obs=args.t_obs, dt=gb_problem.DT,
                              ncrop=gb_problem.NCROP)
    y = gb_problem.datastream_to_y(signal + noise)
    print("conditioning y shape:", y.shape)

    # --- build flow + load checkpoint -------------------------------------
    key, key_init = jr.split(key)
    flow = gb_problem.build_flow(
        y_dim=y.shape[-1], hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads, key=key_init,
    )
    flow = eqx.tree_deserialise_leaves(args.checkpoint, flow)

    # --- sample the posterior ---------------------------------------------
    @eqx.filter_jit
    def sample(x0, y):
        return jax.vmap(lambda xi: gb_problem.sample_flow(flow, xi, y, args.ode_steps))(x0)

    key, key_x0 = jr.split(key)
    x0 = jr.uniform(key_x0, (args.n_samples, 1, gb_problem.X_DIM))
    u = sample(x0, y).block_until_ready()
    u = np.asarray(u)[:, 0, :]   # (N, 4) unit cube
    phys = np.asarray(gb_problem.u_to_physical(jnp.asarray(u)))  # (N, 4) physical

    # --- corner plot in [log10 f0, log10 fdot, log10 A, psi] --------------
    plot = np.column_stack([np.log10(phys[:, 0]), np.log10(phys[:, 1]),
                            np.log10(phys[:, 2]), phys[:, 3]])
    truth = [float(np.log10(phys_true[0])), float(np.log10(phys_true[1])),
             float(np.log10(phys_true[2])), float(phys_true[3])]
    labels = [r"$\log_{10} f_0$", r"$\log_{10}\dot f$", r"$\log_{10} A$", r"$\psi$"]

    plt.close("all")
    fig = corner.corner(plot, labels=labels, truths=truth, truth_color="black",
                        color="C1", show_titles=True, title_fmt=".3f",
                        range=[(np.log10(gb_problem.RANGES["f0"][0]), np.log10(gb_problem.RANGES["f0"][1])),
                               (np.log10(gb_problem.RANGES["fdot"][0]), np.log10(gb_problem.RANGES["fdot"][1])),
                               (np.log10(gb_problem.RANGES["A"][0]), np.log10(gb_problem.RANGES["A"][1])),
                               gb_problem.RANGES["psi"]],
                        hist_kwargs={"density": True})
    fig.legend(handles=[mlines.Line2D([], [], color="C1", label="flow network"),
                        mlines.Line2D([], [], color="black", label="injected truth")],
               loc="upper right", fontsize=13, frameon=False)
    fig.suptitle("Flow-network posterior on a GB injection", y=1.02)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
