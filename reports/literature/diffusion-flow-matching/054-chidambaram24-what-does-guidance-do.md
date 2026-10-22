# What does guidance do? A fine-grained analysis in a simple setting

Chidambaram, Gatmiry, Chen, Lee, Lu, 2024; NeurIPS 2024, not on arXiv page · [arXiv:2409.13074](https://arxiv.org/abs/2409.13074)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 56 citations in 2.1 yr
(Semantic Scholar, 2026-10-19).

## What they do

Guidance is commonly motivated as sampling the tilt p(x)p(z|x)^(1+w). That is false because tilting and noising do not commute (the guidance term mixes scores across noise levels). The paper proves what guided probability-flow ODE sampling actually does for (i) a mixture of two compactly supported 1-D components and (ii) ½N(1,1)+½N(−1,1).

## What they find

- Thm 1 (compact support): as w grows, samples concentrate in (α₂(1−32/√ln w), α₂), the **far edge of the class support**, with probability ≥ 1−exp(−wα₁²/(512·ratio²)). These "archetypes" need not maximise the conditional likelihood.
- Thm 2 (Gaussians): samples drift to +∞ at rate √w, with P(x ≥ √(w+1)/4) ≥ 1−e^{−Ω(w)}.
- Thm 3: for any nonzero score error, a large enough w pushes samples **off the support** even with exact integration. The trajectory overshoots into the noised tails and cannot return.
- Experiments are qualitative or plot-only: 2-D uniform mixtures (w=3 is best), MNIST CFG with DDPM 400 steps, and ImageNet-256 classifier guidance where s≈5 is the largest scale with zero "support error" (pixels out of [0,255]) for tiger and peacock. Killer whale and porcupine need s<5, and basketball is non-monotone.

## Implementation details

Heuristic: pick the largest w for which the 1-D projected trajectory stays monotone (does not overshoot the support and come back). All experiments ran on one A5000.

## Caveats

1-D, two components, exact scores in the main theorems. The heuristic is empirical, and there are no FID numbers. The reverse SDE is only remarked on.
