# SOAP: Improving and Stabilizing Shampoo using Adam (Vyas et al., 2024)

[arXiv:2409.11321](https://arxiv.org/abs/2409.11321), ICLR 2025.

**Evidence: A.** Peer-reviewed; 238 citations in 2.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Run Adam in Shampoo's eigenbasis, refreshing the eigenbasis infrequently.

## What they find

- Faster than AdamW and Shampoo on LLM pretraining.
- In the diffusion benchmarks, SOAP is the best per step on the Navier–Stokes denoiser
  ([004](004-schaipp25-diffusion-benchmark.md)) and has the lowest CelebA loss
  ([005](005-bolatov26-diffusion-benchmark.md)).

## Limits

- No optax implementation; community JAX ports only.
