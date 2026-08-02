# CANNA

A framework for solving **inverse problems with conditional flow matching**:
given an observation `y`, sample the posterior `p(x | y)` over the parameters
that produced it.

A network learns a velocity field that transports a base sample to a posterior
sample along a geodesic.

## Layout

Each problem is a **self-contained package** — `point/`, `sinusoid/`, `lisa/` —
owning its `problem.py`, `network.py`, `train.py`, `eval.py` and `configs/`.
There is no abstract problem base and no shared trainer: a new inverse problem
is a new package, copied and edited. Only `networks/` (the reusable `nnx` blocks
and the `MLP`/`MMDiT` backbones) is shared.

- **The problem** (`problem.py`) — an `eqx.Module` holding the priors to draw
  parameters from, the simulator that turns them into an observation, and the
  preprocessing that makes it a network input. `train_sample(key)` draws
  `(p, o)`, forms the conditioning `y`, walks a geodesic from a base point to
  the whitened parameters, and returns the point and the velocity to match.
- **The geometry** (`geometries.py`, per package) — where the parameters live.
  Supplies `log_map` / `exp_map`, so the flow steps along the manifold rather
  than through the coordinates. Angles ride a circle (`Spherical(2)`), sky
  positions ride a sphere, and interchangeable sources form a `Set` whose
  posterior is permutation-invariant — no branch cuts, no poles, no label
  ambiguity. Flat parameters need no manifold at all, so `point/` has no
  geometry module.
- **The chart** (`physical_to_flow` / `flow_to_physical`) — how parameters are
  coordinatized for the flow, and back. It is the only bridge between the two,
  so the physics stays in the units it is written in and the flow only ever
  sees the manifold.

Stating the priors is most of the work of stating the geometry: a prior uniform
on a sphere is a parameter that lives on one, an angle drawn on a circle wraps,
and a block of interchangeable sources is a set. The geometry a problem needs is
largely **inferrable from the priors it puts on its parameters**, rather than
declared twice and kept in sync by hand.

## Usage

```bash
uv run python -m canna.<problem>.train --config <name>   # writes outputs/<problem>-<name>/
uv run python -m canna.<problem>.eval  --config <name>   # corner plots from that checkpoint
```

`<problem>` is `point`, `sinusoid` or `lisa`; `<name>` is a run `.yaml` in that
package's `configs/` (`B`, `XS`). Every CLI flag overrides the config.
