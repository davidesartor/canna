# Back to Basics: Let Denoising Generative Models Denoise

Li & He, 2025, arXiv · [arXiv:2511.13720](https://arxiv.org/abs/2511.13720)

**Evidence: C.** Not peer-reviewed; 4 citations (OpenAlex, arXiv record only (Semantic Scholar
has no entry), 2026-10-19). Its claims are the authors' alone: a hypothesis, not a result.

## What they do

Have the network predict the clean sample x (x-prediction), but compute the loss in velocity space, v = (net(z_t) − z_t)/(1−t), with the 1−t denominator clipped at 0.05. Clean data lies on a low-dimensional manifold, so predicting x is easy for an under-wide network, whereas ε or v are full-dimensional and cannot be represented when the patch dimension exceeds the token width.

## What they find

- Toy: 2D data embedded in D ∈ {2, 8, 16, 512}, 5-layer 256-wide MLP. ε- and v-prediction degrade at D = 16 and fail at D = 512; x-prediction works throughout.
- Table 2a, ViT-B/16 at 256 px (FID), rows are loss space:
  | Loss | x-pred | ε-pred | v-pred |
  |---|---|---|---|
  | x-loss | 10.14 | 379 | 107.55 |
  | ε-loss | 10.45 | 394.58 | 126.88 |
  | v-loss | 8.62 | 372 | 96.53 |
- Table 2b, B/4 at 64 px (patch dim small): all nine combinations work, FID 3.46–6.20. The failure is specific to patch dim > width.
- Noise shift μ of logit-normal t (x-pred FID): 0.0 → 14.44, −0.4 → 9.79, −0.8 → 8.62, −1.2 → 8.99. No shift rescues ε- or v-prediction.
- EDM-style and linear preconditioners are worse than plain x-prediction: 28.94 and 39.50 (x-loss), 35.49 and 46.25 (v-loss).
- Architecture ablations (Table 4): baseline 7.48; +RoPE +qk-norm 6.69; +in-context class tokens 5.49.
- Headline FID: 256 px H/16 1.86, G/16 1.82; 512 px G/32 1.78.

## Implementation details

- z_t = t x + (1−t)ε; t logit-normal(μ = −0.8, σ = 0.8); clip 1−t at 0.05 in the v conversion.
- Bottleneck patch embedding to d′ = 128; noise scaled by image_size/256.
- Adam (0.9, 0.95), lr 2e-4 constant, batch 1024, EMA, 50-step Heun sampler.

## Caveats

Images with a Gaussian base only. The bottleneck/width argument is shown empirically, not proven.
