# NorMuon: Making Muon more efficient and scalable (Li et al., 2025)

[arXiv:2510.05491](https://arxiv.org/abs/2510.05491), preprint.

**Evidence: B.** Not peer-reviewed; 85 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

- Observe that Muon's orthogonalised updates leave very uneven per-neuron (row) norms.
- After orthogonalisation, keep an Adam-style second moment for each row (β₂ ≈ 0.95), and
  divide each row of the update by its root. Rescale so the update keeps Muon's overall size.

## What they find

- At 1.1B parameters: 21.7% more training efficiency than Adam, and 11.3% more than Muon.
- Row normalisation adds little memory: one scalar per row.

## Limits

- LLMs only, with no diffusion evidence either way.
