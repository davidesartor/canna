# PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior

Lee et al., 2021; ICLR 2022 · [arXiv:2106.06406](https://arxiv.org/abs/2106.06406)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 135
citations in 5.4 yr (Semantic Scholar, 2026-10-19).

## What they do

replace N(0,I) with N(μ(c),Σ(c)) computed by hand from the condition (vocoder: per-frame mel energy as diagonal std, clipped ≥0.1, μ=0; acoustic model: per-phoneme mean/var from aligned training data); loss ‖ε−ε_θ‖²_{Σ⁻¹}. Prop.: with linear ε_θ and det Σ=1 the attainable loss is ≤ baseline; Hessian better conditioned.

## What they find

vocoder MOS @1M steps T=50: 4.25 vs DiffWave 4.12; PriorGrad @500k (4.21) beats DiffWave @1M (4.12) → "2× fewer iterations"; T=6: 4.14 vs 4.01. Acoustic model @60k 4.04 vs baseline 3.84 (baseline @300k 3.91); T=2: 4.25 vs 2.80. Smaller model retains more quality (1.23M: 4.02 vs 3.90).

## Implementation details

**a trainable prior failed** (vocoder diverged; acoustic FAIL at 60k, 3.32 at 300k); alternative hand-designed condition statistics (V/UV, phoneme) were worse; std floor 0.1.

## Caveats

no ablation of Σ⁻¹ loss weighting vs prior alone; diagonal only; speech only; MOS CIs overlap in places.
