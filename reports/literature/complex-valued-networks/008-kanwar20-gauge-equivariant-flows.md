# Equivariant Flow-Based Sampling for Lattice Gauge Theory

Kanwar, Albergo, Boyda, Cranmer, Hackett, Racanière, Rezende & Shanahan, 2020, Physical Review
Letters 125, 121601 · [arXiv:2003.06413](https://arxiv.org/abs/2003.06413)

**Evidence: A.** Peer-reviewed (PRL); 193 citations in 6.6 yr (OpenAlex, 2026-10-19; Semantic
Scholar rate-limited).

## What they do

Normalizing flows for sampling U(1) lattice gauge theory in 2D, made **gauge-equivariant by
construction** so that the model density is exactly invariant under local U(1) phase rotations
of the link variables. Each gauge-equivariant coupling layer:

- picks a subset of plaquettes (products of four link phases around a square, which are
  gauge-invariant up to conjugation) to update, holding the rest frozen;
- applies an invertible "kernel" `h` to those plaquette phases, a mixture of non-compact
  projections (circle-valued, invertible maps of U(1)), whose parameters come from a CNN fed
  only gauge-invariant quantities from the frozen plaquettes;
- translates the plaquette update back into a link update, so the composite map commutes with
  every gauge transformation.

The prior is uniform (Haar) on the links, itself gauge-invariant. Training minimises reverse
KL `KL(q ‖ e^{−S}/Z)` (self-training: no data); samples are proposals in a Metropolis
independence chain, so estimates stay unbiased.

## What they find

- 16 × 16 lattice, seven couplings β = 1…7, 24 gauge-equivariant coupling layers, mixtures
  of 6 components; 100,000-sample ensembles per β.
- Integrated autocorrelation time of the topological charge at the largest β: τ_int ≈ 10 for
  the flow vs ≈ 4000 for heat bath and ≈ 15000 for HMC. Accounting for per-step cost
  (one flow proposal ≈ one HMC trajectory, heat bath half), the flow sampler is "roughly 1500
  times more efficient than HMC and 200 times more efficient than Heat Bath" for topological
  quantities.
- Equivariance as the enabling ingredient (qualitative, no table): "when the symmetry is
  enforced, we can successfully train flow-based models at a range of parameters approaching
  criticality, while without it, models of similar scale fail to learn the distributions under
  the same training procedure."

## Implementation details

- Variables are angles (link phases); the flow works directly on the circle via circle
  diffeomorphisms rather than embedding in ℝ².
- Training halted at a loss plateau; the authors state no extensive optimisation of the
  masking pattern, architecture or hyperparameters.

## Caveats

- The "non-equivariant flows fail" claim is stated without numbers or a described baseline.
- Efficiency figures exclude training cost; tiny lattice (16 × 16); a symmetry that is local
  (one phase per site) and exact. Self-trained (reverse KL), not amortised over observed data.
