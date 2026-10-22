# FlowMM: Generating Materials with Riemannian Flow Matching

Miller et al., 2024, ICML 2024 · [arXiv:2406.04713](https://arxiv.org/abs/2406.04713)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 157 citations in
2.4 yr (Semantic Scholar, 2026-10-19).

## What they do

RFM (κ=1−t) on a product of flat tori (fractional coordinates, uniform base) × lattice lengths (log-normal base fit to data) × angles (logit-mapped from [60°,120°], uniform base). An EGNN-style GNN uses displacement edges and sinusoidal features of fractional coordinates. The loss is a per-dimension-normalised weighted sum.

## What they find

- CSP match rate: MP-20 61.39% vs DiffCSP 51.49%; MPTS-52 17.54 vs 12.19.
- CSP reaches its best match at ~50 integration steps (DiffCSP uses 1000).
- DNG: stability 4.32–4.65% vs DiffCSP 5.06%, at 250 steps and ~3.4× lower cost per stable structure.
- Ablations (App. B, 1 sample, proposed base, MP-20):
  - anti-annealing configurations span 56.82–61.26%.
  - **with the lattice base distribution / unconstrained transform ablated, several configurations collapse to 0% match.**

## Implementation details

- **The fractional-coordinate loss weight is large: λ̃_f = 300–1500 vs λ̃_l = 1** (λ̃_a = 300 for DNG). The swept range was 100–500 (CSP).
- **Inference anti-annealing:** the velocity is multiplied by s(t) = 1 + s′t on the torus coordinates only, with s′ = 1–10 per dataset. Applying it to the lattice hurt.
- AdamW, grad clip 0.5, width 512, 6 layers.

## Caveats

Anti-annealing is a sampling hack whose best value is task- and variable-dependent. It changes the sampled distribution, so there is no calibration guarantee. Inference variance only is reported (3 seeds).
