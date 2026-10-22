# Orthogonality of decision boundaries in complex-valued neural networks

Nitta, 2004, Neural Computation 16(1):73–97 ·
[DOI:10.1162/08997660460734001](https://doi.org/10.1162/08997660460734001)

**Evidence: A.** Peer-reviewed (Neural Computation); 176 citations in 22 yr (Semantic Scholar,
2026-10-19; OpenAlex lists 169). Companion: Nitta 2003, "Solving the XOR
problem and the detection of symmetry using a single complex-valued neuron", Neural Networks
16(8):1101–1105,
[DOI:10.1016/S0893-6080(03)00168-0](https://doi.org/10.1016/S0893-6080(03)00168-0),
226 citations (Semantic Scholar) / 235 (OpenAlex).

## What they do

Analyses the decision boundaries of networks whose weights, thresholds, inputs and outputs
are all complex, using a split (Re/Im) sigmoid output. A single complex neuron outputs two
real numbers (Re and Im of `σ(w·z + b)`), each thresholded, so its decision boundary is a
pair of hypersurfaces `Re(w·z + b) = 0` and `Im(w·z + b) = 0`.

## What they find

From the abstract:

1. The two hypersurfaces of a single complex neuron **intersect orthogonally** and split the
   input space into four regions. Hence a single complex neuron (two-layer complex net)
   solves **XOR** and **symmetry detection**, which a single real neuron cannot, and it
   handles a fading-equalisation problem "with the best generalisation".
2. In three-layer complex nets trained by complex backprop, the learned boundaries also tend
   to be orthogonal, which the author links to better generalisation.
3. Complex-BP learns **several times faster** on average than real BP, with a smaller spread
   in learning speed.

## Implementation details

Split-sigmoid activations (sigmoid on Re and Im separately), complex backpropagation
(Nitta's Complex-BP). Geometrically (Cauchy–Riemann), `Re(w·z)` and `Im(w·z)` are the same
real weight vector rotated by 90° in each (Re, Im) plane, so their gradients are
perpendicular and of equal norm.

## Caveats

- Abstract-level only (full text paywalled): network sizes and how the real baselines were
  sized (whether parameter- or neuron-matched) are not verified here.
- The XOR result counts *neurons*: one complex neuron has 2 real outputs and 2× the real
  weights of one real neuron, so it is closer to two real neurons with tied weights; it is an
  inductive-bias statement, not a capacity win.
- Split activations are not phase-equivariant; low-dimensional toy problems.
