# OminiControl2: Efficient Conditioning for Diffusion Transformers

Tan et al., 2025 · [arXiv:2503.08280](https://arxiv.org/abs/2503.08280)

**Evidence: A.** Peer-reviewed (2026 IEEE/CVF Conference on Computer Vision and Pattern
Recognition- FINDINGS Track (CVPRF)); 58 citations in 1.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Condition tokens sit in the joint MMDiT sequence behind an asymmetric mask: they never attend to the noisy tokens. Their per-layer K/V are computed at step 1 and reused at every later step.

## What they find

- FID with feature reuse: depth 26.88 → 28.72, deblur 16.38 → 18.64, canny 22.84 → 22.27, multi-condition 7.30 → 8.49.
- Latency 13.30 → 8.18 s. Multi-condition 31.63 → 9.78 s, i.e. condition overhead −84.7%.
- The full method is 3.8–5.9× faster.
- **Negative:** caching *without* training under the mask breaks quality (deblur FID 45.0 vs 16.4).

## Implementation details

Train with the mask so that the cache is exact. The paper does not say how the condition tokens are t-modulated.

## Caveats

FLUX + LoRA at 512² only. Small quality cost.
