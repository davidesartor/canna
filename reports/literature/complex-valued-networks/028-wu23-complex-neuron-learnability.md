# Complex-valued neurons can learn more but slower than real-valued neurons via gradient descent

Wu, Zhang, Jiang & Zhou, 2023, NeurIPS 2023 ·
[proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ac4365b98bc242acd5ab974a05c68a8-Abstract-Conference.html)
· DOI:10.52202/075280-1030

**Evidence: B.** Peer-reviewed (NeurIPS); 20 citations in ~3 yr (Semantic Scholar,
2026-10-19).

## What they do

A learning-theory comparison of a **single complex-valued neuron** and real-valued neurons
and networks trained by gradient descent, in the teacher–student setting: how fast (if at
all) each learns a target real neuron or a target complex neuron.

## What they find

From the abstract:

- A complex neuron learns a **real** target neuron at a convergence rate `O(t^{-3})` and a
  **complex** target neuron at `O(t^{-1})`, `t` the number of GD iterations.
- A **two-layer real network of any finite width cannot learn** a single non-degenerate
  complex neuron — "complex neurons can learn more".
- Learning a real neuron with a complex neuron is **slower**: the rate is `Ω(t^{-3})`,
  exponentially slower than the real neuron's own linear rate `O(e^{-ct})` — "but slower".
- A JMLR 2026 extension (v27, no. 38; not read) reports that a phase reparameterisation
  restores a linear rate.

## Implementation details

Learning theory under gradient descent. Not verified here: the activation, the input
distribution, the "non-degenerate" condition and any experiments. No precision study, and
no optimiser other than GD in the abstract.

## Caveats

- Abstract-only (no arXiv version; proceedings PDF not parsed). Rates and claims above are
  the abstract's wording.
- A single-neuron result; it does not say how the gap scales with width or depth in a
  practical architecture, and the "cannot learn" statement concerns a specific real
  two-layer parametrisation of the target, not parameter-matched real networks in general.
