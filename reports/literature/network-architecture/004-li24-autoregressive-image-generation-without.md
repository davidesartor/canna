# Autoregressive Image Generation without Vector Quantization

Li et al., 2024, NeurIPS 2024 · [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 772 citations in 2.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

MAR. The transformer computes z once per token; a small adaLN-conditioned residual MLP runs every diffusion step. Training samples **4 t per z**.

## What they find

MAR-L, 400 epochs, head width sweep:
| Head width | Params | FID (no CFG) | FID (CFG) |
|---|---|---|---|
| 256 | 2M | 3.47 | 2.45 |
| 1024 | 21M | 2.85 | 1.97 |
| 1536 | 45M | 2.93 | 1.91 |
The head is about 5% of parameters and 10% of runtime.

## Implementation details

Multiple noise draws per expensive condition computation.

## Caveats

Per-token head with no mixing between tokens.
