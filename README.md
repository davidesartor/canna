# CANNA

A general-purpose framework for solving **inverse problems with conditional flow
matching**: given an observation `y`, sample the posterior `p(x | y)` over the
parameters that produced it.

A network learns a velocity field that transports a base sample to a posterior
sample along a geodesic. Nothing in the trainer knows about any particular
physics — a problem supplies its priors, its simulator, and the geometry its
parameters live on, and the same machinery infers its posterior.

## The protocols

- **`Problem`** — the forward model. The priors to draw parameters from, the
  simulator that turns them into an observation, and the preprocessing that
  makes it a network input.
- **`Prior`** — a distribution over one parameter block. Stating the priors is
  most of the work of stating the geometry: a prior uniform on a sphere is a
  parameter that lives on one, an angle drawn on a circle wraps, and a block of
  interchangeable sources is a set. The geometry a problem needs is largely
  **inferrable from the priors it puts on its parameters**, rather than declared
  twice and kept in sync by hand.
- **`Geometry`** — where the parameters live. Supplies `log_map` / `exp_map`, so
  the flow steps along the manifold rather than through the coordinates. Angles
  ride a circle, sky positions ride a sphere, and interchangeable sources form a
  set whose posterior is permutation-invariant — no branch cuts, no poles, no
  label ambiguity.
- **`Chart`** — how a parameter block is coordinatized for the flow: the map
  from physical units onto its geometry's embedded coordinates, and back. It is
  the only bridge between the two, so the physics stays in the units it is
  written in and the flow only ever sees the manifold.

Composing these is the whole design: a new inverse problem is a new `Problem`,
not a new trainer.
