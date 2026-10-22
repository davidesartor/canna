# CMuon: Chunked Momentum Orthogonalization for Diffusion Transformers (Chen et al., 2026)

[arXiv:2608.02502](https://arxiv.org/abs/2608.02502), ECCV 2026.

**Evidence: B.** Peer-reviewed; 3 citations in 0.2 yr (Semantic Scholar, 2026-10-19).

## What they do

- Diagnose plain Muon on DiTs: it is fast early but plateaus late.
- Their explanation: fused matrices concatenate maps with different roles, and
  orthogonalising the fused matrix as one couples their update subspaces. Examples:
  - the QKV projection;
  - adaLN's shift/scale/gate.
- The fix: chunk the momentum into its constituent maps before Newton–Schulz, and
  orthogonalise each chunk on its own.

## What they find

- DiT-XL on ImageNet-256 reaches FID 1.18 in 200 epochs. They report this as more than 2×
  faster convergence than AdamW.
- Chunking removes the late plateau. It also makes training more stable than plain Muon.

## Limits

- Image DiTs only.
