# SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics

Abbe et al., 2023, COLT 2023 · [arXiv:2302.11055](https://arxiv.org/abs/2302.11055)

**Evidence: A.** Peer-reviewed (Annual Conference Computational Learning Theory); 164 citations
in 3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

The setup is a multi-index target f*(x)=h*(Mx) with P=O(1) latent coordinates. Expand h* in the Hermite (or Walsh) basis. Leap(h*) is the smallest worst-case number of new coordinates, counted with Hermite multiplicity, that any ordering of the monomials has to add at once.
- The conjecture, proved for a subclass with layerwise projected SGD on 2-layer networks: online SGD on regular, isotropically initialised nets needs Θ̃(d^{max(Leap-1,1)}) steps.
- Dynamics go saddle to saddle. Each plateau has length set by the leap of the next monomial, and low-degree monomials act as an implicit curriculum for higher ones.

## What they find

- Table 1, samples needed:
  - z1⋯z_{2k}: Θ̃(d^{2k-1}).
  - z1⋯z_k + z1⋯z_{2k}: Θ̃(d^{k-1}).
  - Full staircase: Θ(d).
  - Kernels need Ω(d^{2k}) in all three cases.
- CSQ lower bound: d^{Leap} (Boolean) and d^{Leap/2} (Gaussian, proven).
- Fig 1: d=100, M=300, shifted sigmoid, step 0.4/d, target z1+z1⋯z5+z1⋯z9+z1⋯z14. The z1 term is picked up in ~d steps, and the leap-4/4/5 terms in ~d³, d³ and d⁴ steps.
- Appendix A (5-layer ResNets, width 1000):
  - He3(z1) (leap 3) at d=500 plateaus.
  - Adding He1(z1) (leap 1) removes the plateau entirely.
  - Saddles become more pronounced as d grows from 50 to 400.
  - Depth helps fitting once the support is found, but does not help finding it.
- Thm 1: the correlation with the support stays at its initial value for T/(polylog) steps, then jumps (Cor 1b). Early-stopped features cannot fit the target at all (risk ≥ 1 - log d/d^D).
- The authors conjecture n = Θ̃(d^{Leap/2}) for multi-pass ERM. Sample reuse helps, but the exponent stays.

## Implementation details

Adding a low-degree "stepping-stone" term to the target, or as an auxiliary output, lowers the leap.

## Caveats

The proofs are for 2-layer layerwise SGD with projections. The deep-network evidence is empirical and small-scale. Inputs are isotropic Gaussian or Boolean, unlike structured STFT tokens. Leap is defined for orthogonal polynomial expansions of the target in the input; mapping a conditional velocity target (a function of x_t and t as well) to a leap is heuristic.
